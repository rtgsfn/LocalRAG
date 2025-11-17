# run_benchmark.py
import time
import pandas as pd
from multiprocessing import Pool, cpu_count
from itertools import product
from dataclasses import dataclass, asdict
import logging
import sys
import hashlib

from llama_index.core import (
    VectorStoreIndex, StorageContext, load_index_from_storage
)
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Importa le nostre factory e config
from strategies import (
    get_parser, get_chunker, get_embed_model,
    get_vector_store, get_reranker, get_llm
)
from config import TEST_FILE, EVAL_QUESTIONS

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger("run_single_experiment").setLevel(logging.INFO)


# --- 1. Definizione della Configurazione di Pipeline ---

@dataclass
class PipelineConfig:
    """Dataclass per contenere una singola combinazione di pipeline."""
    parser: str
    chunker: str
    embedder: str
    vector_store: str
    reranker: str
    llm: str


# --- 2. Caching dell'Ingestion ---

def get_ingestion_hash(parser_name, chunker_name, embedder_name):
    """Crea un hash unico per una pipeline di ingestion."""
    s = f"{parser_name}-{chunker_name}-{embedder_name}-{TEST_FILE}"
    return hashlib.md5(s.encode()).hexdigest()


def run_ingestion_pipeline(config: PipelineConfig, embed_model, collection_name: str):
    """
    Esegue PARSING e CHUNKING.
    Questa è la parte più lenta e viene "caching" su disco.
    """
    log = logging.getLogger("run_single_experiment")

    # 1. Parsing
    parser = get_parser(config.parser)
    documents = parser.load_data(TEST_FILE)

    # 2. Chunking
    # Passiamo l'embed_model al chunker semantico se richiesto
    if config.chunker == "semantic":
        chunker = get_chunker(config.chunker, embed_model_for_semantic=embed_model)
    else:
        chunker = get_chunker(config.chunker)

    nodes = chunker.get_nodes_from_documents(documents)

    # 3. Indicizzazione (Embedding + Storage)
    log.info(f"[{collection_name}] Indicizzazione di {len(nodes)} nodi...")

    # Colleghiamo il vector store persistente
    vector_store = get_vector_store(config.vector_store, collection_name)

    if vector_store:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
    else:  # Usa in-memory
        index = VectorStoreIndex(nodes, embed_model=embed_model)

    log.info(f"[{collection_name}] Indicizzazione completata.")
    return index


# --- 3. L'Esecutore del Singolo Esperimento ---

def run_single_experiment(config: PipelineConfig):
    """
    Esegue un singolo test (Retrieval + Sintesi) e ne valuta le performance.
    Riutilizza un indice di ingestion se esiste già.
    """
    log = logging.getLogger("run_single_experiment")
    collection_name = get_ingestion_hash(config.parser, config.chunker, config.embedder)
    log.info(f"--- INIZIO: {collection_name} | Reranker={config.reranker} | LLM={config.llm} ---")

    try:
        t_start = time.time()

        # --- A. Setup Componenti ---
        # (Impostiamo i Settings per questo specifico run)
        llm = get_llm(config.llm)
        embed_model = get_embed_model(config.embedder)
        reranker = get_reranker(config.reranker)

        # --- B. Ingestion / Caching ---
        # Controlla se l'indice esiste già
        try:
            # Tenta di caricare l'indice persistente
            vector_store = get_vector_store(config.vector_store, collection_name)
            if vector_store:
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = load_index_from_storage(storage_context, embed_model=embed_model)
                log.info(f"[{collection_name}] Indice caricato dalla cache.")
            else:
                raise ValueError("In-memory store, ricalcolo.")
        except Exception as e:
            # Se non esiste, lo crea
            log.warning(f"[{collection_name}] Indice non trovato, avvio ingestion... (Errore: {e})")
            index = run_ingestion_pipeline(config, embed_model, collection_name)

        ingestion_time = time.time() - t_start

        # --- C. Retrieval (Ricerca Ibrida + Reranking) ---
        t_start_eval = time.time()

        # 1. Creiamo i retriever (Ibrido come da tua immagine)
        dense_retriever = index.as_retriever(similarity_top_k=10)
        # BM25 (sparse) richiede i nodi, li prendiamo dall'index
        nodes = list(index.docstore.docs.values())
        sparse_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

        # 2. Fusione RRF (Usando QueryFusionRetriever)
        retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            similarity_top_k=10,  # Quanti risultati vuoi alla fine
            mode="reciprocal_rerank",  # <-- ECCO LA FUSIONE RRF
            num_queries=1,  # Imposta a 1 per disabilitare la generazione di query
            use_async=True,
        )

        # 3. Costruiamo il Query Engine
        query_engine = index.as_query_engine(
            retriever=retriever,
            node_postprocessors=[reranker] if reranker else [],
            llm=llm,
        )

        # --- D. Evaluation (Qualità RAG) ---
        relevancy_eval = RelevancyEvaluator(llm=llm)
        faithfulness_eval = FaithfulnessEvaluator(llm=llm)

        total_relevancy = 0
        total_faithfulness = 0

        for question in EVAL_QUESTIONS:
            response = query_engine.query(question)

            eval_rel = relevancy_eval.evaluate_response(query=question, response=response)
            total_relevancy += 1 if eval_rel.passing else 0

            eval_faith = faithfulness_eval.evaluate_response(response=response)
            total_faithfulness += 1 if eval_faith.passing else 0

        avg_relevancy = total_relevancy / len(EVAL_QUESTIONS)
        avg_faithfulness = total_faithfulness / len(EVAL_QUESTIONS)

        eval_time = time.time() - t_start_eval

        log.info(f"--- FINE: {collection_name} | Relevancy: {avg_relevancy:.2f} ---")

        return {
            **asdict(config),  # Converte il dataclass in dict
            "ingestion_cached": ingestion_time < 1.0,  # Stima se era in cache
            "avg_relevancy": round(avg_relevancy, 2),
            "avg_faithfulness": round(avg_faithfulness, 2),
            "query_time_s": round(eval_time, 2),
            "error": None
        }

    except Exception as e:
        log.error(f"!!! ERRORE in {collection_name}: {e}", exc_info=True)
        return {
            **asdict(config),
            "ingestion_cached": False,
            "avg_relevancy": 0,
            "avg_faithfulness": 0,
            "query_time_s": 0,
            "error": str(e)
        }


# Funzione wrapper per la parallelizzazione
def wrapper_run_single_experiment(config):
    return run_single_experiment(config)


# --- 4. Main Orchestrator ---

if __name__ == "__main__":
    print("Avvio del RAG Hyperparameter Benchmark...")

    # --- !! QUI DEFINISCI TUTTE LE COMBINAZIONI !! ---

    # Scegli da: ["simple_pdf", "nougat"]
    PARSERS = ["simple_pdf"]

    # Scegli da: ["fixed_window", "recursive", "hierarchical", "semantic"]
    CHUNKERS = ["recursive", "semantic"]

    # Scegli da: ["bge-large", "all-minilm", "bge-m3:ollama"]
    EMBEDDERS = ["all-minilm", "bge-large"]

    # Scegli da: ["qdrant", "weaviate", "memory"]
    VECTOR_STORES = ["qdrant"]

    # Scegli da: ["none", "bge-reranker", "ms-marco-minilm", "jina-reranker", "flashrank"]
    RERANKERS = ["none", "bge-reranker"]

    # Scegli da: ["llama3:8b", "mistral:7b", "gemma2:9b", "deepseek:7b"]
    LLMS = ["llama3:8b", "mistral:7b"]

    # --- !! ATTENZIONE ALLA COMPLESSITÀ !! ---
    # N. Esperimenti = P * C * E * V * R * L
    # Nell'esempio: 1 * 2 * 2 * 1 * 2 * 2 = 16 esperimenti

    # Genera tutte le configurazioni
    all_configs = [
        PipelineConfig(*p) for p in product(
            PARSERS, CHUNKERS, EMBEDDERS, VECTOR_STORES, RERANKERS, LLMS
        )
    ]

    print(f"Numero totale di esperimenti da eseguire: {len(all_configs)}")
    print("Inizio esecuzione...")

    # --- Esecuzione ---
    # ATTENZIONE: Se usi modelli GPU (Embedding, Reranker, LLM),
    # la parallelizzazione può causare errori CUDA Out-of-Memory.
    # Inizia con num_workers = 1 per sicurezza.

    EXECUTION_MODE = "serial"  # "serial" o "parallel"
    results = []

    if EXECUTION_MODE == "serial":
        for config in all_configs:
            results.append(wrapper_run_single_experiment(config))
    else:
        # Usa N-1 core CPU. NON gestisce la VRAM della GPU.
        num_workers = max(1, cpu_count() - 1)
        # num_workers = 1 # IMPOSTA A 1 SE USI GPU PESANTI
        print(f"Esecuzione in parallelo con {num_workers} worker...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(wrapper_run_single_experiment, all_configs)

    # --- 5. Analisi dei Risultati ---
    print("\n\n--- CONFRONTO UNICO RISULTATI ---")

    df = pd.DataFrame(results)

    # Miglior formattazione
    df = df.sort_values(by=["avg_relevancy", "avg_faithfulness"], ascending=False)

    print(df.to_markdown(index=False))

    # Salva i risultati
    df.to_csv("rag_hyper_benchmark_results.csv", index=False)
    print("\nRisultati salvati in 'rag_hyper_benchmark_results.csv'")