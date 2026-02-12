import torch
import sys
import os

# ==========================================================
# CONFIGURARE CĂI PROIECT
# Determinăm automat structura directoarelor pentru a permite
# rularea scriptului indiferent din ce locație este apelat.
# ==========================================================

# Directorul curent (unde se află acest fișier)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorul 'src'
SRC_DIR = os.path.dirname(CURRENT_DIR)

# Rădăcina proiectului (folderul părinte al lui 'src')
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Adăugăm SRC_DIR în sys.path pentru a putea importa module custom
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Importăm arhitectura modelului
from neural_network.model import DrowsinessCNN


# ==========================================================
# FUNCȚIE GENERARE MODEL NEANTRENAT
# Creează instanța modelului și salvează doar arhitectura +
# greutățile inițiale (random inițializate).
# ==========================================================
def generate():

    # Inițializăm modelul (greutăți random, conform PyTorch defaults)
    model = DrowsinessCNN()
    
    # Folderul în care salvăm modelul
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    
    # Creăm directorul dacă nu există
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # Calea finală de salvare
    SAVE_PATH = os.path.join(MODELS_DIR, 'untrained_model.pth')
    
    # Salvăm doar state_dict (recomandat în PyTorch)
    # Acesta conține doar parametrii modelului, nu întreaga clasă.
    torch.save(model.state_dict(), SAVE_PATH)

    print(f"Model neantrenat salvat la: {SAVE_PATH}")


# ==========================================================
# ENTRY POINT
# Scriptul va rula doar dacă este executat direct,
# nu dacă este importat ca modul.
# ==========================================================
if __name__ == "__main__":
    try:
        generate()
    except Exception as e:
        # Gestionare simplă a erorilor pentru debugging
        print(f"A apărut o eroare: {e}")
