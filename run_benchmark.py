# run_benchmark.py
import time
import pandas as pd
from itertools import product
from dataclasses import dataclass, asdict
import logging
import sys
import hashlib
import numpy as np

from llama_index.core import (
    VectorStoreIndex, StorageContext, load_index_from_storage,
    QueryBundle
)
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index.retrievers.bm25 import BM25Retriever
# Import per RRF
from llama_index.core.retrievers import QueryFusionRetriever

# Importa le nostre factory e config
from strategies import (
    get_parser, get_chunker, get_embed_model,
    get_vector_store, get_reranker, get_llm,
    get_synthesizer  # <-- NUOVO IMPORT
)
from config import TEST_FILE, EVAL_QUESTIONS

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger("run_benchmark").setLevel(logging.INFO)
log = logging.getLogger("run_benchmark")


# --- 1. Definizione delle Configurazione ---

@dataclass
class IngestionConfig:
    parser: str
    chunker: str
    embedder: str
    vector_store: str


@dataclass
class QueryConfig:
    retriever_mode: str  # es. "rrf"
    reranker: str
    llm: str


# --- 2. Pipeline di Ingestione (con Caching) ---

def run_ingestion_pipeline(config: IngestionConfig, embed_model):
    """
    Esegue l'ingestione O la carica dalla cache.
    Restituisce l'indice e i risultati (tempo, num_chunks).
    """
    collection_name = hashlib.md5(str(asdict(config)).encode()).hexdigest()
    log.info(f"--- Ingestion: {collection_name} ---")

    t_start = time.time()

    try:
        # Tenta di caricare l'indice persistente
        vector_store = get_vector_store(config.vector_store, collection_name)
        if vector_store:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            log.info(f"[{collection_name}] Indice caricato dalla cache.")
            ingestion_time_s = 0.0  # Era in cache
            num_chunks = len(index.docstore.docs)
            was_cached = True
        else:
            raise ValueError("In-memory store, ricalcolo.")

    except Exception:
        log.warning(f"[{collection_name}] Indice non trovato, avvio ingestion...")

        # 1. Parsing
        parser = get_parser(config.parser)
        documents = parser.load_data(TEST_FILE)

        # 2. Chunking
        if config.chunker == "semantic":
            chunker = get_chunker(config.chunker, embed_model_for_semantic=embed_model)
        else:
            chunker = get_chunker(config.chunker)
        nodes = chunker.get_nodes_from_documents(documents)
        num_chunks = len(nodes)

        # 3. Indicizzazione (Embedding + Storage)
        vector_store = get_vector_store(config.vector_store, collection_name)

        if vector_store:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        else:  # Usa in-memory
            index = VectorStoreIndex(nodes, embed_model=embed_model)

        ingestion_time_s = time.time() - t_start
        was_cached = False
        log.info(f"[{collection_name}] Indicizzazione completata in {ingestion_time_s:.2f}s.")

    ingestion_metrics = {
        "ingestion_time_s": round(ingestion_time_s, 2),
        "num_chunks": num_chunks,
        "ingestion_cached": was_cached
    }

    return index, ingestion_metrics


# --- 3. Pipeline di Query (con Profiling) ---

def run_query_experiment(index: VectorStoreIndex, config: QueryConfig, llm, reranker):
    """
    Esegue un singolo test (Retrieval + Sintesi) e ne valuta le performance,
    cronometrando ogni singolo passaggio.
    """
    try:
        # --- A. Setup Componenti Query ---

        # 1. Retriever (RRF come da tua richiesta)
        dense_retriever = index.as_retriever(similarity_top_k=10)
        nodes_for_bm25 = list(index.docstore.docs.values())
        sparse_retriever = BM25Retriever.from_defaults(nodes=nodes_for_bm25, similarity_top_k=10)

        retriever = QueryFusionRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            similarity_top_k=10,
            mode="reciprocal_rerank",  # RRF
            num_queries=1,
            use_async=False,  # Mettiamo False per profiling più semplice
        )

        # 2. Synthesizer
        synthesizer = get_synthesizer(llm=llm)

        # 3. Evaluators
        relevancy_eval = RelevancyEvaluator(llm=llm)
        faithfulness_eval = FaithfulnessEvaluator(llm=llm)

        # Liste per salvare i tempi di ogni step
        step_times = {
            "retrieval_time_s": [],
            "reranking_time_s": [],
            "synthesis_time_s": []
        }
        total_relevancy = 0
        total_faithfulness = 0

        # --- B. Esecuzione e Profiling su ogni domanda ---
        for question in EVAL_QUESTIONS:
            query_bundle = QueryBundle(question)

            # --- STEP 1: RETRIEVAL (RRF) ---
            t_start_retrieval = time.time()
            retrieved_nodes = retriever.retrieve(query_bundle)
            t_end_retrieval = time.time()
            step_times["retrieval_time_s"].append(t_end_retrieval - t_start_retrieval)

            # --- STEP 2: RERANKING ---
            t_start_reranking = time.time()
            if reranker:
                processed_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=query_bundle)
            else:
                processed_nodes = retrieved_nodes  # Nessun reranking
            t_end_reranking = time.time()
            step_times["reranking_time_s"].append(t_end_reranking - t_start_reranking)

            # --- STEP 3: SYNTHESIS (LLM) ---
            t_start_synthesis = time.time()
            response = synthesizer.synthesize(query=question, nodes=processed_nodes)
            t_end_synthesis = time.time()
            step_times["synthesis_time_s"].append(t_end_synthesis - t_start_synthesis)

            # --- STEP 4: EVALUATION ---
            eval_rel = relevancy_eval.evaluate_response(query=question, response=response)
            total_relevancy += 1 if eval_rel.passing else 0

            eval_faith = faithfulness_eval.evaluate_response(response=response)
            total_faithfulness += 1 if eval_faith.passing else 0

        # --- C. Aggregazione Risultati ---
        avg_relevancy = total_relevancy / len(EVAL_QUESTIONS)
        avg_faithfulness = total_faithfulness / len(EVAL_QUESTIONS)

        # Calcola la media dei tempi per ogni step
        avg_times = {step: round(np.mean(times), 2) for step, times in step_times.items()}

        return {
            **avg_times,
            "avg_relevancy": round(avg_relevancy, 2),
            "avg_faithfulness": round(avg_faithfulness, 2),
            "error": None
        }

    except Exception as e:
        log.error(f"!!! ERRORE in {config}: {e}", exc_info=True)
        return {
            "retrieval_time_s": 0, "reranking_time_s": 0, "synthesis_time_s": 0,
            "avg_relevancy": 0, "avg_faithfulness": 0,
            "error": str(e)
        }


# --- 4. Main Orchestrator ---

if __name__ == "__main__":
    print("Avvio del RAG Hyperparameter Benchmark (con Profiling)...")

    # --- !! QUI DEFINISCI TUTTE LE COMBINAZIONI !! ---
    # (Come prima)
    PARSERS = ["simple_pdf", "nougat", "nougat_associative"]
    CHUNKERS = ["recursive", "semantic"]
    EMBEDDERS = ["all-minilm", "bge-large"]
    VECTOR_STORES = ["qdrant"]
    RERANKERS = ["none", "bge-reranker"]
    LLMS = ["llama3:8b", "mistral:7b"]

    # --- Creazione combinazioni ---
    ingestion_configs = [
        IngestionConfig(*p) for p in product(PARSERS, CHUNKERS, EMBEDDERS, VECTOR_STORES)
    ]
    query_configs = [
        QueryConfig(retriever_mode="rrf", reranker=r, llm=l) for r, l in product(RERANKERS, LLMS)
    ]

    print(f"Combinazioni Ingestion: {len(ingestion_configs)}")
    print(f"Combinazioni Query: {len(query_configs)}")
    print(f"Esperimenti totali da eseguire: {len(ingestion_configs) * len(query_configs)}")

    all_results = []

    # ATTENZIONE: Esecuzione solo seriale per semplicità
    # La parallelizzazione è complessa con questo nuovo setup di caching

    for ing_config in ingestion_configs:
        try:
            # --- FASE 1: INGESTION (Build o Cache) ---
            # Impostiamo l'embed_model per questa run
            embed_model = get_embed_model(ing_config.embedder)

            index, ingestion_metrics = run_ingestion_pipeline(ing_config, embed_model)

            ing_config_dict = asdict(ing_config)

            # --- FASE 2: QUERY (con profiling) ---
            for q_config in query_configs:
                log.info(
                    f"--- Esecuzione: [Ingestion: {ing_config.chunker}/{ing_config.embedder}] + [Query: {q_config.reranker}/{q_config.llm}] ---")

                # Setup componenti query
                llm = get_llm(q_config.llm)
                reranker = get_reranker(q_config.reranker)

                query_metrics = run_query_experiment(index, q_config, llm, reranker)

                # Combina i risultati
                full_report = {
                    **ing_config_dict,
                    **asdict(q_config),
                    **ingestion_metrics,
                    **query_metrics
                }
                all_results.append(full_report)

        except Exception as e:
            log.error(f"!!! ERRORE FATALE pipeline ingestion {ing_config}: {e}", exc_info=True)
            # Aggiungi un report d'errore
            all_results.append({**asdict(ing_config), "error": str(e)})

    # --- 5. Analisi dei Risultati ---
    print("\n\n--- CONFRONTO UNICO RISULTATI ---")

    df = pd.DataFrame(all_results)

    # Riorganizza le colonne per leggibilità
    base_cols = ["parser", "chunker", "embedder", "reranker", "llm"]
    quality_cols = ["avg_relevancy", "avg_faithfulness"]
    time_cols = ["ingestion_time_s", "retrieval_time_s", "reranking_time_s", "synthesis_time_s"]
    other_cols = [c for c in df.columns if c not in base_cols + quality_cols + time_cols]

    # Assicura che tutte le colonne esistano
    final_cols = [c for c in base_cols + quality_cols + time_cols + other_cols if c in df.columns]
    df = df[final_cols]

    df = df.sort_values(by=["avg_relevancy", "avg_faithfulness"], ascending=False)

    print(df.to_markdown(index=False))
    df.to_csv("rag_hyper_benchmark_results.csv", index=False)
    print("\nRisultati salvati in 'rag_hyper_benchmark_results.csv'")