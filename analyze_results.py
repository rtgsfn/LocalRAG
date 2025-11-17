import pandas as pd
import warnings

# Ignora i warning futuri di pandas, se presenti
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE DELL'ANALISI ---

# Il file CSV generato dal tuo benchmark
INPUT_FILE = "rag_hyper_benchmark_results.csv"

# Le componenti che vuoi confrontare (devono corrispondere ai nomi delle colonne)
COMPONENTS_TO_COMPARE = [
    "parser",
    "chunker",
    "embedder",
    "reranker",
    "llm"
]

# Le metriche che vuoi vedere per ogni confronto
# (Qualità + Tempi di ogni step)
METRICS_TO_SHOW = [
    "avg_relevancy",
    "avg_faithfulness",
    "ingestion_time_s",
    "num_chunks",
    "retrieval_time_s",
    "reranking_time_s",
    "synthesis_time_s"
]


# --- FUNZIONE DI ANALISI ---

def analyze_component_impact(df, component_name, metrics):
    """
    Raggruppa i risultati per un singolo componente e calcola la media
    delle performance per ogni scelta di quel componente.

    Es: Confronta la media di 'chunker=semantic' vs 'chunker=recursive'
    """
    print("=" * 80)
    print(f"📊 REPORT DI ANALISI PER: '{component_name.upper()}'")
    print("=" * 80)

    try:
        # Raggruppa per il componente e calcola la media delle metriche
        # .agg('mean', numeric_only=True) gestisce eventuali errori di tipo
        analysis_df = df.groupby(component_name)[metrics].agg('mean', numeric_only=True)

        # Arrotonda per leggibilità
        analysis_df = analysis_df.round(3)

        # Ordina per la metrica di qualità più importante
        if "avg_relevancy" in metrics:
            analysis_df = analysis_df.sort_values(by="avg_relevancy", ascending=False)

        print(analysis_df.to_markdown(floatfmt=".3f"))
        print("\n")

    except KeyError:
        print(f"Errore: La colonna '{component_name}' non è stata trovata nel CSV.")
    except Exception as e:
        print(f"Errore durante l'analisi di '{component_name}': {e}")


# --- ESECUZIONE DELLO SCRIPT ---

if __name__ == "__main__":
    print("Avvio dell'analisi dei risultati del benchmark RAG...")

    try:
        # Carica il file CSV con i risultati
        results_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERRORE: File '{INPUT_FILE}' non trovato.")
        print(">> Assicurati di aver prima eseguito 'run_hyper_benchmark.py'!")
        exit()
    except Exception as e:
        print(f"ERRORE sconosciuto durante il caricamento del CSV: {e}")
        exit()

    # Esegui l'analisi per ogni componente che abbiamo definito
    for component in COMPONENTS_TO_COMPARE:
        if component in results_df.columns:
            analyze_component_impact(results_df, component, METRICS_TO_SHOW)
        else:
            print(f"Info: La colonna '{component}' non è nel CSV, saltata.\n")

    print("Analisi completata.")