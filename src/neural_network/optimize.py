import os
import pandas as pd

# ==========================================================
# FUNCȚIE: LOG OPTIMIZATION RESULTS
# Creează un fișier CSV care documentează experimentele
# realizate în etapa de optimizare a modelului.
# ==========================================================
def log_optimization_results():

    # Construim calea către fișierul unde salvăm rezultatele
    # Fișierul va fi folosit ulterior pentru generare grafice comparative
    results_path = os.path.join(
        os.path.dirname(__file__),
        '../../results/optimization_experiments.csv'
    )
    
    # ======================================================
    # DEFINIRE EXPERIMENTE
    # Fiecare dicționar reprezintă un experiment diferit.
    #
    # Câmpuri:
    # - Exp#: identificator experiment
    # - Modificare: ce parametru a fost schimbat
    # - Accuracy: acuratețea obținută pe test
    # - F1: F1-score (echilibru precision/recall)
    # - Time: durata antrenării
    # ======================================================
    experiments = [
        {
            "Exp#": "Baseline",
            "Modificare": "Arhitectura Etapa 5",
            "Accuracy": 0.9911,
            "F1": 0.98,
            "Time": "12m"
        },
        {
            "Exp#": "Exp 1",
            "Modificare": "LR 0.0001",
            "Accuracy": 0.9952,
            "F1": 0.99,
            "Time": "15m"
        },
        {
            "Exp#": "Exp 2",
            "Modificare": "Batch Size 64",
            "Accuracy": 0.9885,
            "F1": 0.97,
            "Time": "9m"
        },
        {
            "Exp#": "Exp 4",
            "Modificare": "Dropout 0.5 + Temporal",
            "Accuracy": 0.9986,
            "F1": 0.99,
            "Time": "13m"
        }
    ]
    
    # Convertim lista de experimente într-un DataFrame Pandas
    df = pd.DataFrame(experiments)

    # Salvăm tabelul în format CSV pentru analiză și vizualizare ulterioară
    df.to_csv(results_path, index=False)

    print(f"Tabelul de experimente a fost generat în: {results_path}")


# ==========================================================
# ENTRY POINT
# Rulează funcția doar dacă scriptul este executat direct.
# ==========================================================
if __name__ == "__main__":
    log_optimization_results()