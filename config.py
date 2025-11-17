# config.py
import torch

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Configurazione caricata. Device di calcolo: {DEVICE}")

# --- Connessioni Servizi (da Docker) ---
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Dati di Test ---
# Usa lo stesso PDF del test precedente
TEST_FILE = "./test_doc.pdf"

# --- Domande di Valutazione ---
# (Fondamentali per giudicare la qualità della pipeline)
EVAL_QUESTIONS = [
    "What is the main topic of the document?",
    "What is a Transformer model?",
    "Explain the concept of 'self-attention'.",
    # Aggiungi domande più specifiche e difficili
]