import torch
import json
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================== 
# LOGICĂ PENTRU ADĂUGAREA CĂII CORESPUNZĂTOARE
# ============================================================== 
# Obține calea curentă și folderul root al proiectului
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
src_path = os.path.join(project_root, "src")

# Adaugă src în sys.path pentru a importa modulele locale
if src_path not in sys.path:
    sys.path.append(src_path)

# Încearcă să importi modelul DrowsinessCNN
try:
    from neural_network.model import DrowsinessCNN
    print("Modulul 'neural_network' a fost găsit în 'src'.")
except ImportError:
    print(f"EROARE: Nu am găsit 'neural_network' în {src_path}")
    sys.exit(1)

# ============================================================== 
# ⚙️ CONFIGURAȚII CĂI ȘI DEVICE
# ============================================================== 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CPU/GPU
MODEL_PATH = os.path.join(project_root, "models/optimized_model.pth")   # Model salvat
TEST_DATA_PATH = os.path.join(project_root, "data/processed/test")      # Date test
OUTPUT_JSON = os.path.join(project_root, "results/error_analysis.json") # Output JSON

# ============================================================== 
# FUNCȚIE PRINCIPALĂ DE ANALIZĂ
# ============================================================== 
def generate_analysis():
    # Verificări preliminare
    if not os.path.exists(MODEL_PATH):
        print(f"EROARE: Modelul lipsește din: {MODEL_PATH}")
        return
    if not os.path.exists(TEST_DATA_PATH):
        print(f"EROARE: Folderul de test lipsește din: {TEST_DATA_PATH}")
        print("Verifică dacă drumul este: data -> processed -> test")
        return

    # Încărcare Model
    model = DrowsinessCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()  # Mod evaluare (dezactivează Dropout/BatchNorm)

    # Transformări pentru test (ca la antrenare)
    transform = transforms.Compose([
        transforms.Grayscale(),             # 1 canal (alb-negru)
        transforms.Resize((64, 64)),       # Redimensionare
        transforms.ToTensor(),             # Tensor [C,H,W]
        transforms.Normalize([0.5], [0.5]) # Normalizare [-1,1]
    ])
    
    # Încărcare Date
    test_set = datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    classes = test_set.classes       # Lista clase: ["Closed", "Open"]
    errors = []                      # Listă pentru greșeli
    tp, tn, fp, fn = 0, 0, 0, 0     # Inițializare Confusion Matrix

    print(f"Analizăm {len(test_set)} imagini din {TEST_DATA_PATH}...")

    # Inferență și analiza greșelilor
    with torch.no_grad():
        for i, (image, label) in enumerate(loader):
            image, label = image.to(DEVICE), label.to(DEVICE)
            output = model(image)                           # Forward pass
            prob = torch.nn.functional.softmax(output, dim=1)  # Probabilități
            pred = torch.argmax(prob, dim=1).item()           # Predicția modelului
            actual = label.item()                             # Eticheta reală

            # Actualizare Confusion Matrix
            if pred == actual:
                if actual == 1: tp += 1  # True Positive (Open)
                else: tn += 1            # True Negative (Closed)
            else:
                if actual == 1: fn += 1  # False Negative (Open pred Closed)
                else: fp += 1            # False Positive (Closed pred Open)
                
                # Salvăm detalii pentru analiza erorilor
                img_path, _ = test_set.samples[i]
                errors.append({
                    "filename": os.path.basename(img_path),
                    "true_label": classes[actual],
                    "predicted": classes[pred],
                    "confidence": round(prob[0][pred].item(), 4)
                })

    # Calcul și salvare metrici în JSON
    accuracy = (tp + tn) / len(test_set) if len(test_set) > 0 else 0
    analysis_results = {
        "metrics": {
            "total_samples": len(test_set),
            "accuracy": round(accuracy, 5),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
        },
        "errors": errors
    }

    with open(OUTPUT_JSON, "w", encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=4)

    # 7️⃣ Output consolă
    print("-" * 30)
    print(f"Rezultat: {accuracy*100:.2f}% acuratețe")
    print(f"JSON generat în: results/error_analysis.json")

# ============================================================== 
# 🔧 EXECUȚIE SCRIPT
# ============================================================== 
if __name__ == "__main__":
    generate_analysis()