# strategies.py
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    MarkdownNodeParser, HierarchicalNodeParser
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.readers.nougat_ocr import PDFNougatOCR
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from flashrank import Ranker, RerankRequest
from llama_index.core.postprocessor.types import BaseNodePostprocessor

import qdrant_client
import weaviate
from weaviate.embedded import EmbeddedOptions

from config import (
    DEVICE, OLLAMA_BASE_URL
)


# --- 1. Parser Factory ---
def get_parser(name: str):
    if name == "simple_pdf":
        return SimpleDirectoryReader()
    if name == "nougat":
        return PDFNougatOCR()
    raise ValueError(f"Parser '{name}' non supportato.")


# --- 2. Chunker Factory ---
def get_chunker(name: str, embed_model_for_semantic=None):
    if name == "fixed_window":
        return SentenceSplitter(chunk_size=512, chunk_overlap=50)
    if name == "recursive":
        return SentenceSplitter(chunk_size=512, chunk_overlap=50,
                                separator=["\n\n", "\n", ". ", " ", ""])
    if name == "hierarchical":
        return HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
    if name == "semantic":
        if not embed_model_for_semantic:
            raise ValueError("Semantic chunker richiede un embed_model.")
        return SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95,
            embed_model=embed_model_for_semantic
        )
    if name == "hybrid_markdown":
        return MarkdownNodeParser()
    raise ValueError(f"Chunker '{name}' non supportato.")


# --- 3. Embedding Factory (dalla tua immagine) ---
def get_embed_model(name: str):
    # Nota: Specter2 richiede addestramento/setup specifici, lo omettiamo

    # Modelli da Ollama (se li hai scaricati)
    if name == "bge-m3:ollama":
        return OllamaEmbedding(model_name="bge-m3", base_url=OLLAMA_BASE_URL)

    # Modelli locali via HuggingFace (scaricati automaticamente)
    if name == "bge-large":
        return HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=DEVICE)
    if name == "all-minilm":
        return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    raise ValueError(f"Embedding model '{name}' non supportato.")


# --- 4. Vector Store Factory (dalla tua immagine) ---
def get_vector_store(name: str, collection_name: str):
    if name == "qdrant":
        # MODIFICA QUI:
        # Usa un percorso locale invece di una URL
        client = qdrant_client.QdrantClient(path="./qdrant_db_storage")
        return QdrantVectorStore(client=client, collection_name=collection_name,
                                 enable_hybrid=True, batch_size=64)
    if name == "weaviate":
        client = weaviate.WeaviateClient(
            embedded_options=EmbeddedOptions(
                persistence_data_path="./weaviate_db_storage"
            )
        )
        return WeaviateVectorStore(weaviate_client=client, index_name=collection_name)
    if name == "memory":  # Baseline veloce
        return None  # Indica di usare un VectorStore in memoria

    raise ValueError(f"Vector store '{name}' non supportato.")


# --- 5. Reranker Factory (dalla tua immagine) ---
# Wrapper per FlashRank (non ha integrazione LlamaIndex nativa)
class FlashRankReranker(BaseNodePostprocessor):
    def __init__(self, top_n: int = 3):
        self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")  # Esempio
        self._top_n = top_n

    def _postprocess_nodes(self, nodes, query_bundle):
        passages = [{"id": n.node_id, "text": n.get_content()} for n in nodes]
        req = RerankRequest(query=query_bundle.query_str, passages=passages)
        results = self._ranker.rerank(req)

        # Mappa i risultati riordinati ai nodi originali
        id_to_node = {n.node_id: n for n in nodes}
        reranked_nodes = []
        for res in results[:self._top_n]:
            reranked_nodes.append(id_to_node[res["id"]])
        return reranked_nodes


def get_reranker(name: str, top_n: int = 3):
    if name == "none":
        return None  # Nessun reranker
    if name == "bge-reranker":
        # BAAI/bge-reranker-v2-m3
        return FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=top_n)
    if name == "ms-marco-minilm":
        # pip install sentence-transformers
        return SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2",top_n=top_n)
    if name == "jina-reranker":
        # jinaai/jina-reranker-v2-base-multilingual
        return JinaRerank(model="jinaai/jina-reranker-v2-base-multilingual", top_n=top_n)
    if name == "flashrank":
        return FlashRankReranker(top_n=top_n)

    raise ValueError(f"Reranker '{name}' non supportato.")


# --- 6. LLM Factory (dalla tua immagine) ---
def get_llm(name: str):
    # Nota: Assicurati di aver fatto 'ollama pull <name>' per questi
    if name == "llama3:8b":
        return Ollama(model="llama3:8b-instruct", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    if name == "mistral:7b":
        return Ollama(model="mistral:7b-instruct-v0.2", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    if name == "gemma2:9b":
        return Ollama(model="gemma2:9b-instruct", base_url=OLLAMA_BASE_URL, request_timeout=120.0)
    if name == "deepseek:7b":
        return Ollama(model="deepseek-llm:7b-chat", base_url=OLLAMA_BASE_URL, request_timeout=120.0)

    raise ValueError(f"LLM '{name}' non supportato.")