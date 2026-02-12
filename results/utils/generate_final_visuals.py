import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1. PATH FINDER
# Identifică automat rădăcina proiectului (folderul care conține 'src')
# pentru a permite rularea scriptului din orice subdirector.
# ==========================================================
def get_project_root():
    current_path = os.path.abspath(__file__)  # Pornim din locația scriptului
    while current_path != os.path.dirname(current_path):  # Urcăm în ierarhia folderelor
        current_path = os.path.dirname(current_path)
        if os.path.exists(os.path.join(current_path, 'src')):
            # Dacă există folderul 'src', considerăm că am găsit root-ul proiectului
            return current_path
    return None

ROOT = get_project_root()
sys.path.append(ROOT)  # Permite importuri relative din proiect

# ==========================================================
# 2. CONFIGURARE CĂI
# Definim locațiile fișierelor sursă și ale output-urilor grafice.
# ==========================================================
RESULTS_DIR = os.path.join(ROOT, 'results')
DOCS_DIR = os.path.join(ROOT, 'docs')
OPT_DIR = os.path.join(DOCS_DIR, 'optimization')

# Fișiere generate anterior în pipeline-ul de training/evaluare
HISTORY_CSV = os.path.join(RESULTS_DIR, 'training_history.csv')   # Curbe train/val
METRICS_JSON = os.path.join(RESULTS_DIR, 'test_metrics.json')     # Metrici finale test

# Setare stil global pentru toate graficele (consistență vizuală)
sns.set_theme(style="whitegrid")

def generate_real_plots():
    print("Generăm graficele folosind datele tale REALE...")

    # ==========================================================
    # 1. LOSS & LEARNING CURVES (din training_history.csv)
    # ==========================================================
    if os.path.exists(HISTORY_CSV):

        # Citim istoricul salvat în timpul antrenării
        df = pd.read_csv(HISTORY_CSV)
        
        # -------- A. Learning Curves (Accuracy) --------
        # Comparație între performanța pe train și validation
        plt.figure(figsize=(10, 5))
        plt.plot(df['train_acc'], label='Train Accuracy', lw=2)
        plt.plot(df['val_acc'], label='Val Accuracy', lw=2)
        plt.title('Real Learning Curves - Final Model', fontsize=14)
        plt.legend()

        # Salvare în folderul principal docs
        plt.savefig(os.path.join(DOCS_DIR, 'learning_curves_final.png'))
        plt.close()

        # -------- B. Loss Curves --------
        # Analizăm dacă modelul a suferit overfitting sau underfitting
        plt.figure(figsize=(10, 5))
        plt.plot(df['train_loss'], label='Train Loss', color='red', lw=2)
        plt.plot(df['val_loss'], label='Val Loss', color='orange', lw=2)
        plt.title('Real Loss Curve - Final Model', fontsize=14)
        plt.legend()

        plt.savefig(os.path.join(DOCS_DIR, 'loss_curve.png'))
        plt.close()

        print("Loss & Learning Curves generate.")

    # ==========================================================
    # 2. CONFUSION MATRIX (din test_metrics.json)
    # ==========================================================
    if os.path.exists(METRICS_JSON):

        # Citim metricile generate în etapa de evaluare
        with open(METRICS_JSON, 'r') as f:
            data = json.load(f)
        
        # Convertim matricea în array numpy pentru calcule
        cm = np.array(data['confusion_matrix'])

        # Calculăm procentajele pe fiecare linie (normalizare pe clasă reală)
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))

        # Construim etichete personalizate: valoare absolută + procent
        labels = np.array([
            [f"{cm[0,0]}\n({cm_perc[0,0]:.1%})", f"{cm[0,1]}\n({cm_perc[0,1]:.1%})"],
            [f"{cm[1,0]}\n({cm_perc[1,0]:.1%})", f"{cm[1,1]}\n({cm_perc[1,1]:.1%})"]
        ])

        # Heatmap pentru vizualizare intuitivă
        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap='Blues',
            xticklabels=['Pred Closed', 'Pred Open'],
            yticklabels=['Actual Closed', 'Actual Open']
        )

        plt.title('Confusion Matrix (Real Test Results)')
        plt.savefig(os.path.join(DOCS_DIR, 'confusion_matrix_optimized.png'))
        plt.close()

        print("Confusion Matrix reală generată.")

    # ==========================================================
    # 3. EVOLUȚIA PERFORMANȚEI PE STAGES DE OPTIMIZARE
    # ==========================================================
    exp_csv = os.path.join(RESULTS_DIR, 'optimization_experiments.csv')

    if os.path.exists(exp_csv):

        # Citim rezultatele experimentelor succesive
        exp_df = pd.read_csv(exp_csv)
        
        # -------- A. Comparare simplă Accuracy (Bar Plot) --------
        # Evidențiază îmbunătățirea progresivă a modelului
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Stage', y='Accuracy', data=exp_df, palette='Blues_d')
        plt.ylim(0.8, 1.0)  # Focus pe interval relevant
        plt.title('Accuracy Improvement per Stage')

        plt.savefig(os.path.join(OPT_DIR, 'accuracy_comparison.png'))
        plt.close()

        # -------- B. Evoluție multi-metrică (Accuracy & F1) --------
        # Transformăm datele pentru a putea afișa două metrici pe același grafic
        df_melted = exp_df.melt(
            id_vars='Stage',
            value_vars=['Accuracy', 'F1'],
            var_name='Metric',
            value_name='Value'
        )
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x='Stage',
            y='Value',
            hue='Metric',
            data=df_melted,
            marker='o',
            lw=3
        )

        plt.ylim(0.8, 1.0)
        plt.title('Model Performance Evolution (Accuracy & F1-Score)', fontsize=14)

        plt.savefig(os.path.join(DOCS_DIR, 'metrics_evolution.png'))
        plt.close()
        
        print("Diferențiere finalizată: Accuracy (Bar) vs Metrics Evolution (Line).")

# ==========================================================
# Entry Point
# Scriptul rulează doar dacă este executat direct.
# ==========================================================
if __name__ == "__main__":
    generate_real_plots()
    print("\nGATA! Toate graficele din /docs au fost generate cu succes!")
