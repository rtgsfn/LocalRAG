# analyze_results.py
import pandas as pd
import warnings
import itertools

# Ignora i warning futuri di pandas, se presenti
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURAZIONE DELL'ANALISI ---

INPUT_FILE = "rag_hyper_benchmark_results.csv"

# 1. DEFINISCI LA TUA PIPELINE DI BASELINE
#    Questi componenti rimarranno "bloccati" durante l'analisi
BASELINE_PIPELINE = {
    "parser": "simple_pdf",
    "chunker": "recursive",
    "embedder": "all-minilm",
    "reranker": "none",
    "llm": "llama3:8b"
}

# 2. DEFINISCI I COMPONENTI DA VARIARE (uno alla volta)
#    Questi sono i componenti che vuoi analizzare in isolamento.
#    Lo script li testerà uno per uno, sovrascrivendo la baseline.
COMPONENTS_TO_ANALYZE = [
    "chunker",
    "embedder",
    "reranker",
    "llm"
]

# 3. METRICHE DA MOSTRARE
METRICS_TO_SHOW = [
    "avg_relevancy",
    "avg_faithfulness",
    "retrieval_time_s",
    "reranking_time_s",
    "synthesis_time_s"
]


# --- FUNZIONE DI ANALISI ---

def analyze_component_isolation(df, component_to_vary, baseline_config):
    """
    Filtra il DataFrame per la baseline, poi mostra il confronto
    per il singolo componente che sta variando.
    """
    print("=" * 80)
    print(f"📊 REPORT DI ISOLAMENTO PER: '{component_to_vary.upper()}'")
    print("-" * 80)

    # 1. Costruisci i filtri della baseline
    #    Escludi il componente che stiamo variando
    filters = baseline_config.copy()
    if component_to_vary in filters:
        del filters[component_to_vary]

    # 2. Applica i filtri
    filtered_df = df.copy()
    for key, value in filters.items():
        if key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        print(f"ATTENZIONE: Nessun dato trovato per la baseline: {filters}")
        print("Assicurati di aver eseguito il benchmark per queste combinazioni.\n")
        return

    # 3. Estrai le metriche per il componente variabile
    try:
        # Colonne da mostrare: il componente e le metriche
        cols_to_show = [component_to_vary] + METRICS_TO_SHOW

        # Rimuovi colonne non esistenti se ce ne sono
        cols_to_show = [col for col in cols_to_show if col in filtered_df.columns]

        analysis_df = filtered_df[cols_to_show].drop_duplicates()

        # Ordina per la metrica di qualità più importante
        if "avg_relevancy" in analysis_df.columns:
            analysis_df = analysis_df.sort_values(by="avg_relevancy", ascending=False)

        print(f"Mostrando confronto con baseline fissa: {filters}\n")
        print(analysis_df.to_markdown(index=False, floatfmt=".3f"))
        print("\n")

    except KeyError as e:
        print(f"Errore: La colonna {e} non è stata trovata.")
    except Exception as e:
        print(f"Errore durante l'analisi di '{component_to_vary}': {e}")


# --- ESECUZIONE DELLO SCRIPT ---

if __name__ == "__main__":
    print("Avvio dell'analisi di isolamento (Ablation Study)...")

    try:
        results_df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERRORE: File '{INPUT_FILE}' non trovato.")
        print(">> Assicurati di aver prima eseguito 'run_hyper_benchmark.py'!")
        exit()

    # Esegui l'analisi per ogni componente che abbiamo definito
    for component in COMPONENTS_TO_ANALYZE:
        analyze_component_isolation(results_df, component, BASELINE_PIPELINE)

    print("Analisi completata.")