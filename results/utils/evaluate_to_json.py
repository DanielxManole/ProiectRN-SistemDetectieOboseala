import torch
import os
import sys
import json
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# 1. PATH FINDER
# Caută automat rădăcina proiectului (folderul care conține 'src')
# pentru a permite rularea scriptului din orice subdirector.
# ==========================================================
def get_project_root():
    current_path = os.path.abspath(__file__)  # Pornim din locația fișierului curent
    while current_path != os.path.dirname(current_path):  # Urcăm în ierarhia de foldere
        current_path = os.path.dirname(current_path)
        if os.path.exists(os.path.join(current_path, 'src')):
            # Dacă găsim folderul 'src', considerăm că am ajuns la root
            return current_path
    return None  # Dacă nu este găsit, returnăm None

ROOT = get_project_root()
sys.path.append(ROOT)  # Permite importarea modulelor custom din proiect

from src.neural_network.model import DrowsinessCNN

# ==========================================================
# 2. CONFIGURARE GLOBALĂ
# Definirea căilor și a parametrilor principali
# ==========================================================
MODEL_PATH = os.path.join(ROOT, 'models', 'optimized_model.pth')   # Model antrenat
TEST_DIR = os.path.join(ROOT, 'data', 'processed', 'test')         # Set de test
SAVE_PATH = os.path.join(ROOT, 'results', 'test_metrics.json')     # Output metrici

# Selectare automată GPU dacă este disponibil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 3. PIPELINE DE PREPROCESARE
# IMPORTANT: trebuie să fie identic cu cel folosit la antrenare.
# ==========================================================
transform = transforms.Compose([
    transforms.Grayscale(),              # Conversie la 1 canal (modelul e antrenat pe grayscale)
    transforms.Resize((64, 64)),         # Redimensionare la input-ul așteptat
    transforms.ToTensor(),               # Conversie la tensor [0,1]
    transforms.Normalize((0.5,), (0.5,)) # Normalizare în intervalul [-1, 1]
])

def run_evaluation():
    print(f"Evaluăm modelul pe datele din: {TEST_DIR}")
    
    # ==========================================================
    # 1. ÎNCĂRCARE DATE
    # ImageFolder atribuie automat label-uri în ordine alfabetică.
    # Ex: 'closed' → 0, 'open' → 1 (dacă acestea sunt numele folderelor).
    # ==========================================================
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,   # Procesăm imaginile în batch-uri pentru eficiență
        shuffle=False    # Nu amestecăm datele la evaluare
    )
    
    # ==========================================================
    # 2. ÎNCĂRCARE MODEL
    # ==========================================================
    model = DrowsinessCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Dezactivăm dropout și batchnorm în mod training

    all_preds = []   # Lista pentru predicții
    all_labels = []  # Lista pentru etichetele reale

    # ==========================================================
    # 3. INFERENȚĂ PE TOT SETUL DE TEST
    # ==========================================================
    with torch.no_grad():  # Dezactivăm gradient computation (optimizare memorie + viteză)
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            # Alegem clasa cu scorul maxim (echivalent cu argmax pe logits)
            _, preds = torch.max(outputs, 1)
            
            # Mutăm rezultatele pe CPU și le convertim în numpy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ==========================================================
    # 4. CALCUL METRICI CU SCIKIT-LEARN
    # ==========================================================
    # classification_report calculează:
    # - precision
    # - recall
    # - f1-score
    # - accuracy
    report = classification_report(
        all_labels,
        all_preds,
        output_dict=True,
        target_names=['Closed', 'Open']
    )

    # Confusion Matrix: [[TP_closed, FP_closed],
    #                    [FN_open,   TP_open]]
    cm = confusion_matrix(all_labels, all_preds).tolist()

    # ==========================================================
    # 5. CONSTRUIRE STRUCTURĂ JSON
    # Rotunjim la 4 zecimale pentru claritate și consistență.
    # ==========================================================
    metrics_data = {
        "accuracy": round(report['accuracy'], 4),

        "precision_closed": round(report['Closed']['precision'], 4),
        "recall_closed": round(report['Closed']['recall'], 4),
        "f1_closed": round(report['Closed']['f1-score'], 4),

        "precision_open": round(report['Open']['precision'], 4),
        "recall_open": round(report['Open']['recall'], 4),
        "f1_open": round(report['Open']['f1-score'], 4),

        "confusion_matrix": cm,
        "total_test_samples": len(all_labels)
    }

    # ==========================================================
    # 6. SALVARE REZULTATE ÎN FORMAT JSON
    # Permite integrarea ușoară în documentație sau dashboard.
    # ==========================================================
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with open(SAVE_PATH, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    print(f"Fișier generat: {SAVE_PATH}")
    print(f"Acuratețe test: {metrics_data['accuracy']*100:.2f}%")

# ==========================================================
# Entry point – rulează evaluarea doar dacă fișierul este
# executat direct (nu importat ca modul).
# ==========================================================
if __name__ == "__main__":
    run_evaluation()
