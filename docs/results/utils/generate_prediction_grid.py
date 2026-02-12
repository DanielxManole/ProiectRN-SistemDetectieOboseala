import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import random
import numpy as np

# ==========================================================
# 1. PATH FINDER INTELIGENT
# Caută automat rădăcina proiectului (folderul care conține 'src')
# pentru a permite rularea scriptului din orice locație.
# ==========================================================
def get_project_root():
    current_path = os.path.abspath(__file__)  # Pornim din locația fișierului curent
    while current_path != os.path.dirname(current_path):  # Urcăm în ierarhia de directoare
        current_path = os.path.dirname(current_path)
        if os.path.exists(os.path.join(current_path, 'src')):
            # Dacă găsim folderul 'src', considerăm că am ajuns la root-ul proiectului
            return current_path
    return None  # Dacă nu este găsit, returnăm None

ROOT = get_project_root()
if ROOT is None: 
    print("Nu am găsit rădăcina proiectului.")
    sys.exit(1)  # Oprim execuția pentru a evita erori ulterioare

# Adăugăm root-ul în sys.path pentru a putea importa module custom
sys.path.append(ROOT)

from src.neural_network.model import DrowsinessCNN

# ==========================================================
# 2. CONFIGURARE CĂI ȘI PARAMETRI GLOBALI
# ==========================================================

# Calea către modelul antrenat și optimizat
MODEL_PATH = os.path.join(ROOT, 'models', 'optimized_model.pth')

# Locația unde va fi salvată imaginea cu predicțiile
SAVE_PATH = os.path.join(ROOT, 'docs', 'example_predictions.png')

# Directorul setului de test
TEST_DIR = os.path.join(ROOT, 'data', 'processed', 'test')

# Subfolderele pentru cele două clase
CLOSED_PATH = os.path.join(TEST_DIR, 'closed')
OPEN_PATH = os.path.join(TEST_DIR, 'open')

# Selectăm automat GPU dacă este disponibil, altfel CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 3. PIPELINE DE PREPROCESARE
# Transformările trebuie să fie IDENTICE cu cele folosite la antrenare.
# ==========================================================
transform = transforms.Compose([
    transforms.Grayscale(),              # Asigurăm 1 canal (modelul e antrenat pe grayscale)
    transforms.Resize((64, 64)),         # Redimensionare la dimensiunea așteptată de model
    transforms.ToTensor(),               # Conversie în tensor PyTorch [0,1]
    transforms.Normalize((0.5,), (0.5,)) # Normalizare în intervalul [-1, 1]
])

def predict_and_plot():
    print(f"🔍 Scanăm TOT setul de test pentru a găsi punctele slabe...")
    
    # Inițializare model și încărcare greutăți
    model = DrowsinessCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Dezactivăm dropout/batchnorm în mod training

    # ==========================================================
    # 1. ÎNCĂRCARE COMPLETĂ SET DE TEST
    # ==========================================================
    all_files = []

    # Construim manual lista (path, label)
    for f in os.listdir(CLOSED_PATH):
        all_files.append((os.path.join(CLOSED_PATH, f), "Closed"))
    for f in os.listdir(OPEN_PATH):
        all_files.append((os.path.join(OPEN_PATH, f), "Open"))
    
    results = []

    # ==========================================================
    # 2. INFERENȚĂ PE ÎNTREG SETUL
    # ==========================================================
    for img_path, true_label in all_files:
        img_pil = Image.open(img_path).convert('L')  # Forțăm grayscale
        img_tensor = transform(img_pil).unsqueeze(0).to(device)  
        # unsqueeze(0) adaugă dimensiunea batch (1, C, H, W)
        
        with torch.no_grad():  # Dezactivăm calculul gradientului (optimizare performanță)
            output = model(img_tensor)

            # Aplicăm softmax pentru a obține probabilități
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            
            prob_closed = probs[0].item() 
            prob_open = probs[1].item()
            
            # Predicția finală este clasa cu probabilitatea mai mare
            pred_label = "Closed" if prob_closed > prob_open else "Open"
            
            # Confidența pe eticheta corectă (folosită pentru detectarea cazurilor dificile)
            conf_correct = prob_closed if true_label == "Closed" else prob_open
            
            # Salvăm toate informațiile relevante pentru analiză ulterioară
            results.append({
                'path': img_path,
                'true': true_label,
                'pred': pred_label,
                'conf': max(prob_closed, prob_open) * 100,  # Confidența maximă (%)
                'conf_correct': conf_correct,              # Probabilitatea clasei corecte
                'img': np.array(img_pil),
                'correct': pred_label == true_label        # Boolean pentru corectitudine
            })

    # ==========================================================
    # 3. ANALIZĂ PERFORMANȚĂ
    # Sortăm după probabilitatea pe clasa corectă (ascendent).
    # Astfel:
    # - Primele poziții = greșeli sau predicții foarte nesigure.
    # - Următoarele = cazuri limită (edge cases).
    # ==========================================================
    results.sort(key=lambda x: x['conf_correct'])
    
    worst_3 = results[:3]                     # Cele mai problematice imagini
    others_6 = random.sample(results[3:], 6)  # 6 exemple random pentru comparație
    
    final_selection = worst_3 + others_6

    # ==========================================================
    # 4. VIZUALIZARE GRAFICĂ (GRID 3x3)
    # ==========================================================
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Model Analysis: Weakest Predictions vs. Random Samples',
                 fontsize=20, fontweight='bold')
    
    axes = axes.flatten()

    for i, data in enumerate(final_selection):
        axes[i].imshow(data['img'], cmap='gray')
        axes[i].axis('off')
        
        # Verde = predicție corectă, Roșu = greșită
        color = 'green' if data['correct'] else 'red'
        
        # Primele 3 sunt cazuri dificile (edge cases)
        prefix = "EDGE CASE" if i < 3 else "STABLE"
        
        axes[i].set_title(
            f"{prefix}\nActual: {data['true']}\nPred: {data['pred']} ({data['conf']:.1f}%)",
            color=color,
            fontsize=12,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Creăm directorul dacă nu există
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Salvăm figura la rezoluție bună pentru documentație
    plt.savefig(SAVE_PATH, dpi=150)
    
    print("✅ Grid salvat!")

# ==========================================================
# Entry point al scriptului
# ==========================================================
if __name__ == "__main__":
    predict_and_plot()
