⚙️ LocalRAG Benchmark Tool
Questo è un framework avanzato per il benchmark e il profiling di pipeline RAG (Retrieval-Augmented Generation) complete, ottimizzato per l'uso di modelli locali (tramite Ollama) e vector store embedded (Qdrant, Weaviate).

Lo scopo di questo tool non è essere un'applicazione RAG, ma trovare scientificamente la pipeline RAG migliore per i tuoi dati e i tuoi requisiti, testando ogni componente in isolamento.

✨ Features Principali
Totalmente Modulare: Testa e confronta ogni singolo pezzo della pipeline:

Parsing: (simple_pdf, nougat)

Chunking: (recursive, semantic, fixed_window)

Embedding: (all-minilm, bge-large, bge-m3)

Vector Store / Retrieval: (qdrant, weaviate, bm25, RRF)

Reranking: (none, bge-reranker, jina-reranker)

Generazione (LLM): (llama3:8b, mistral:7b, gemma2:9b)

Benchmarking "Head-to-Head": Esegui un'analisi di isolamento ("ablation study") per vedere l'impatto di un singolo componente (es. chunker) tenendo tutto il resto costante.

Profiling Dettagliato: Misura non solo il risultato finale, ma i costi e la qualità di ogni step:

ingestion_time_s

retrieval_time_s

reranking_time_s

synthesis_time_s

Valutazione Qualitativa (LLM-as-a-Judge): Misura automaticamente la qualità della risposta usando metriche che non richiedono un ground truth manuale:

avg_relevancy: La risposta è pertinente alla domanda?

avg_faithfulness: La risposta è fedele ai documenti (non si inventa nulla)?

Caching Intelligente: L'ingestion (parsing, chunking, embedding) è lenta. Lo script salva i risultati su disco e li riutilizza, permettendoti di testare decine di LLM e Reranker in pochi minuti.

Local-First: Progettato per girare interamente sulla tua macchina, senza Docker. Richiede solo l'app di Ollama.

🛠️ Installazione e Prerequisiti
Requisiti Fondamentali
Python 64-bit (OBBLIGATORIO): Questo progetto richiede Python 3.10, 3.11, o 3.12 (tassativamente a 64-bit). Versioni più vecchie (3.8) o sperimentali (3.13+) non sono supportate dalle librerie AI.

Ollama: Devi aver installato e in esecuzione Ollama (ollama.com) sul tuo computer.

Setup
Clona o Scarica questo Progetto:

Bash

git clone [TUA_REPO_URL]
cd LocalRAG-Benchmark-Tool
Crea un Ambiente Virtuale (con Python 3.10+):

Bash

python -m venv .venv
.\.venv\Scripts\activate
Installa le Dipendenze:

Bash

pip install -r requirements.txt
Scarica i Modelli Ollama: Questo script non scarica i modelli. Devi farlo tu manualmente per ogni modello che vuoi testare.

Bash

# Esempio per scaricare i modelli di base
ollama pull llama3:8b-instruct
ollama pull mistral:7b-instruct-v0.2
ollama pull all-minilm
ollama pull bge-reranker
🚀 Come Usarlo: Il Workflow in 3 Fasi
Il tuo workflow sarà sempre diviso in tre fasi: Benchmark, Analisi, Test.

Fase 1: Eseguire il Benchmark (Generazione Dati)
Questa è la fase lenta. Dici al tool quali "gare" combattere e lui genera un file CSV con tutti i risultati.

Configura config.py:

Imposta TEST_FILE (es. "./documento_test.pdf").

Scrivi le tue EVAL_QUESTIONS (le domande che userai per valutare).

Configura run_hyper_benchmark.py:

Scorri fino in fondo (if __name__ == "__main__":).

Modifica le liste per definire le combinazioni. Per un test di isolamento (es. testare solo i CHUNKERS), imposta le altre liste a un solo valore (la tua "baseline").

Esegui il Benchmark:

Bash

python run_hyper_benchmark.py
Lo script inizierà a lavorare. La prima esecuzione sarà molto lenta perché deve eseguire l'ingestion per ogni combinazione. Le successive saranno molto più veloci grazie alla cache.

Output: Un file rag_hyper_benchmark_results.csv pieno di dati.

Fase 2: Analizzare i Risultati (Trovare il Vincitore)
Ora che hai i dati, li analizzi per trovare la pipeline migliore.

Configura analyze_results.py:

Apri lo script e imposta la BASELINE_PIPELINE con i valori "standard" che vuoi usare come confronto fisso.

Assicurati che COMPONENTS_TO_ANALYZE contenga i componenti che vuoi valutare (es. "chunker", "llm").

Esegui l'Analisi (Istantaneo):

Bash

python analyze_results.py
Output: Una serie di report "head-to-head" stampati nel tuo terminale, che ti mostrano l'impatto di ogni componente in isolamento.

Esempio: Vedrai un report che ti dice che chunker="semantic" ha una avg_relevancy di 0.9, mentre chunker="recursive" ha 0.6 (tenendo tutto il resto bloccato).

Fase 3: Testare la Pipeline Vincente (Interattivo)
Dopo aver analizzato i report, hai trovato il tuo "vincitore". Ora puoi usarlo in un chatbot interattivo.

Configura test_rag.py:

Apri lo script e modifica le costanti in cima (es. CHUNKER_CHOICE = "semantic", LLM_CHOICE = "llama3:8b") con i componenti della tua pipeline vincente.

Esegui il Tester:

Bash

python test_rag.py
Output: Lo script caricherà la pipeline RAG dalla cache del benchmark e ti permetterà di farle domande in tempo reale.

📂 Descrizione dei File
Script Principali (Il Workflow)

run_hyper_benchmark.py: (FASE 1) Il motore del benchmark. Esegue tutti i test e produce il CSV.

analyze_results.py: (FASE 2) Il motore di analisi. Legge il CSV e stampa i report di isolamento.

test_rag.py: (FASE 3) Il tester interattivo. Carica la pipeline "vincente" e ti permette di usarla.

Moduli di Supporto

config.py: Il "pannello di controllo" globale. Contiene il file di test, le domande di valutazione e gli URL.

strategies.py: La "fabbrica" del progetto. Contiene la logica per istanziare ogni componente (get_llm, get_chunker, ecc.).

Helper (Utility)

export_chunks.py: Script opzionale per esportare tutti i chunk e i loro ID in un file JSONL, utile per facilitare la creazione di un ground truth manuale.

requirements.txt: Tutte le dipendenze Python.

⚖️ Licenza
Questo progetto è rilasciato sotto licenza MIT.