import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# ================= CONFIGURARE =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Calea către datele brute colectate manual de utilizator
# Urcăm din folderul curent (src/data_acquisition) către proiect/data/raw
RAW_DIR = os.path.join(SCRIPT_DIR, '../../data/raw')

# Sursele originale: poze brute cu ochii deschiși și închiși
MY_OPEN_SRC = os.path.join(RAW_DIR, 'MyOpen')
MY_CLOSED_SRC = os.path.join(RAW_DIR, 'MyClosed')

# Destinația finală: folderele mari care conțin toate imaginile (online + tale)
FINAL_OPEN_DST = os.path.join(RAW_DIR, 'Open')
FINAL_CLOSED_DST = os.path.join(RAW_DIR, 'Closed')

# Factor de augmentare:
# Fiecare imagine originală va fi multiplicată de AUG_FACTOR ori
# Exemplu: 300 imagini × 10 = 3000 imagini per clasă
AUG_FACTOR = 10

# ================= AUGMENTĂRI =================
# Definim pipeline-ul de augmentare folosind torchvision.transforms
aug_pipeline = transforms.Compose([
    transforms.RandomRotation(15),               # Rotire aleatorie ±15°
    transforms.ColorJitter(brightness=0.3, 
                           contrast=0.3),        # Modificări ale luminozității/contrastului
    transforms.GaussianBlur(kernel_size=3),      # Blur ușor pentru diversitate
    transforms.RandomHorizontalFlip(p=0.5)      # Flip orizontal cu 50% șansă
])

# ================= FUNCTIE PRINCIPALĂ =================
def process_augmentation(src, dst, label):
    """
    Procesează imaginile din folderul src și le multiplică în dst
    folosind augmentări definite în aug_pipeline.

    Pas cu pas:
    1. Verifică dacă sursa există (altfel iese)
    2. Listează toate imaginile .jpg/.png/.jpeg
    3. Salvează originalul în folderul destinație
    4. Creează AUG_FACTOR copii augmentate pentru fiecare imagine
    """
    if not os.path.exists(src): 
        return

    files = [f for f in os.listdir(src) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Augmentare {label}: {len(files)} poze sursă...")

    for fname in files:
        # Citire imagine cu OpenCV
        img_path = os.path.join(src, fname)
        img_cv = cv2.imread(img_path)
        if img_cv is None: 
            continue  # Dacă imaginea nu poate fi citită, sărim peste ea
        
        # Conversie în PIL.Image pentru a folosi transformările Torchvision
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # --- 1. Salvăm originalul în folderul mare ---
        img_pil.save(os.path.join(dst, f"original_{fname}"))

        # --- 2. Generăm AUG_FACTOR copii augmentate ---
        for i in range(AUG_FACTOR):
            aug_img = aug_pipeline(img_pil)  # Aplică transformările
            aug_img.save(os.path.join(dst, f"myaug_{i}_{fname}"))

# ================= EXECUȚIE =================
if __name__ == "__main__":
    # Procesăm imaginile cu ochii deschiși
    process_augmentation(MY_OPEN_SRC, FINAL_OPEN_DST, "Ochi Deschiși")
    # Procesăm imaginile cu ochii închiși
    process_augmentation(MY_CLOSED_SRC, FINAL_CLOSED_DST, "Ochi Închiși")

    print("Gata! Pozele tale au fost multiplicate și adăugate în folderele Open/Closed.")