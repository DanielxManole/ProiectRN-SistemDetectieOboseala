import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys

# Importăm modelul CNN definit în model.py
from model import DrowsinessCNN

# ================= 1. CONFIGURARE CĂI =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Directorul curent
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))  # Rădăcina proiectului

TEST_DIR = os.path.join(ROOT, 'data/processed/test')  # Folder test
MODEL_PATH = os.path.join(ROOT, 'models/optimized_model.pth')  # Model antrenat
OUTPUT_IMAGE = os.path.join(ROOT, 'docs/confusion_matrix_optimized.png')  # Confusion matrix
OUTPUT_JSON = os.path.join(ROOT, 'results/final_metrics.json')  # JSON metrici finale

# Creăm folderele de output dacă nu există
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

def evaluate():
    # Detectăm device-ul: GPU dacă există, altfel CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------- 1. Transformări -------------------
    # Trebuie să fie identice cu cele folosite la antrenament
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Redimensionăm imaginile
        transforms.Grayscale(num_output_channels=1),  # Convertim la grayscale
        transforms.ToTensor(),  # Transformăm în tensor PyTorch
        transforms.Normalize([0.5], [0.5])  # Normalizare valori [-1,1]
    ])

    # ------------------- 2. Încărcare Date -------------------
    if not os.path.exists(TEST_DIR):
        print(f"EROARE: Folderul de test nu există la {TEST_DIR}")
        return

    test_dataset = datasets.ImageFolder(TEST_DIR, transform)  # Încărcăm dataset
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch=1 pentru evaluare precisă
    classes = test_dataset.classes  # Lista claselor: ['Closed','Open']

    # ------------------- 3. Încărcare Model -------------------
    model = DrowsinessCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"EROARE: Modelul nu există la {MODEL_PATH}")
        return

    # Încărcăm greutățile antrenate
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Setăm modelul în modul eval (fără dropout)

    all_preds, all_labels = [], []  # Liste pentru predicții și label-uri adevărate

    # ------------------- 4. Inferență -------------------
    print(f"Evaluare finală pe: {device}")
    print(f"Analizez {len(test_dataset)} imagini de test...")
    
    with torch.no_grad():  # Dezactivăm gradientul, reduce memorie și viteză
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Alegem clasa cu cea mai mare probabilitate
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ------------------- 5. Calcul Metrici -------------------
    # Classification report pentru metrici detaliate
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)  # F1 mediu macro

    # Pregătim JSON-ul cu metrici importante
    metrics_to_save = {
        "test_accuracy": round(report['accuracy'], 4),
        "f1_macro": round(f1_macro, 4),
        "precision_closed": round(report['Closed']['precision'], 4),
        "recall_closed": round(report['Closed']['recall'], 4),
        "samples_tested": len(all_labels)
    }

    # ------------------- 6. Salvare JSON -------------------
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    # ------------------- 7. Salvare Confusion Matrix -------------------
    cm = confusion_matrix(all_labels, all_preds)  # Matrice de confuzie
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicție Model')
    plt.ylabel('Adevăr (Label)')
    plt.title('Confusion Matrix - Detecție Oboseală (Optimizat)')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    plt.close()  # Închidem plot-ul ca să nu consumăm memorie
    
    # ------------------- 8. Output în consolă -------------------
    print("-" * 50)
    print(f"Test Accuracy: {report['accuracy']:.4f}")
    print(f"Test F1-score (macro): {f1_macro:.4f}")
    print("-" * 50)
    print(f"✓ Confusion matrix saved to: docs/confusion_matrix_optimized.png")
    print(f"✓ Metrics saved to: results/final_metrics.json")
    print(f"✓ Evaluare finalizată cu succes!")

# ------------------- RUN -------------------
if __name__ == "__main__":
    evaluate()