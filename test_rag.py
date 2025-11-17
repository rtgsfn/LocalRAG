import hashlib
import sys
from dataclasses import asdict
import logging
from llama_index.core import (
    StorageContext, load_index_from_storage,
    QueryBundle
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

# Importa le nostre factory e config
from strategies import (
    get_embed_model, get_vector_store, get_reranker, get_llm,
    get_synthesizer
)
# Le dataclass ci servono per l'hashing
from run_benchmark import IngestionConfig

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

# --- 🚀 CONFIGURA LA TUA PIPELINE VINCENTE QUI ---
# (Modifica queste scelte in base ai risultati della tua ANALISI)

PARSER_CHOICE = "simple_pdf"
CHUNKER_CHOICE = "semantic"
EMBEDDER_CHOICE = "bge-large"
VECTOR_STORE_CHOICE = "qdrant"
RERANKER_CHOICE = "bge-reranker"
LLM_CHOICE = "llama3:8b"


# ---------------------------------------------------

def load_production_pipeline():
    """
    Carica la pipeline RAG "vincente" basandosi sulla cache
    generata dal benchmark.
    """
    print("--- Caricamento Pipeline RAG Vincente ---")

    # 1. Carica componenti
    try:
        print(f"LLM: {LLM_CHOICE}")
        llm = get_llm(LLM_CHOICE)

        print(f"Reranker: {RERANKER_CHOICE}")
        reranker = get_reranker(RERANKER_CHOICE)

        print(f"Embedder: {EMBEDDER_CHOICE}")
        embed_model = get_embed_model(EMBEDDER_CHOICE)

        print(f"Synthesizer: (default)")
        synthesizer = get_synthesizer(llm=llm)
    except Exception as e:
        print(f"ERRORE nel caricamento componenti: {e}")
        return

    # 2. Ricrea l'hash dell'indice per caricarlo dalla cache
    ing_config = IngestionConfig(
        parser=PARSER_CHOICE,
        chunker=CHUNKER_CHOICE,
        embedder=EMBEDDER_CHOICE,
        vector_store=VECTOR_STORE_CHOICE
    )
    collection_name = hashlib.md5(str(asdict(ing_config)).encode()).hexdigest()
    print(f"Caricamento indice: {collection_name} (da cache)")

    try:
        vector_store = get_vector_store(VECTOR_STORE_CHOICE, collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    except Exception as e:
        print("\n!!! ERRORE FATALE !!!")
        print(f"Impossibile caricare l'indice '{collection_name}' dalla cache.")
        print("Assicurati di aver ESEGUITO IL BENCHMARK ('run_hyper_benchmark.py')")
        print(f"con questa esatta combinazione di ingestion: {ing_config}")
        return

    # 3. Costruisci il Retriever (RRF)
    print("Costruzione Retriever (RRF)...")
    dense_retriever = index.as_retriever(similarity_top_k=10)
    nodes_for_bm25 = list(index.docstore.docs.values())
    sparse_retriever = BM25Retriever.from_defaults(nodes=nodes_for_bm25, similarity_top_k=10)

    retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        similarity_top_k=10,
        mode="reciprocal_rerank",  # RRF
        num_queries=1,
        use_async=False,
    )

    print("--- 🚀 Pipeline Pronta! ---")
    return retriever, reranker, synthesizer


def main():
    pipeline_components = load_production_pipeline()
    if not pipeline_components:
        return

    retriever, reranker, synthesizer = pipeline_components

    # Loop interattivo
    while True:
        print("\n-------------------------------------")
        try:
            question = input("Inserisci la tua domanda (o 'exit' per uscire): ")
        except EOFError:
            break

        if not question or question.lower() == 'exit':
            break

        print("\n...recupero chunk...")
        query_bundle = QueryBundle(question)
        retrieved_nodes = retriever.retrieve(query_bundle)

        if reranker:
            print("...riordino chunk...")
            processed_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle=query_bundle)
        else:
            processed_nodes = retrieved_nodes

        print("...genero risposta (LLM)...")
        response = synthesizer.synthesize(query=question, nodes=processed_nodes)

        print("\nRisposta RAG:")
        print(str(response))

        # Opzionale: mostra i chunk usati
        print("\nSorgenti usate:")
        for i, node in enumerate(processed_nodes[:3]):  # Mostra le top 3
            print(f"--- Sorgente {i + 1} (Score: {node.score:.4f}) ---")
            print(node.get_content()[:500] + "...")  # Stampa i primi 500 caratteri


if __name__ == "__main__":
    main()