import os
import shutil
import random

# ==========================================================
# CONFIGURARE CĂI ȘI PARAMETRI
# ==========================================================

# Directorul unde se află scriptul curent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directorul sursă (date brute)
SOURCE_DIR = os.path.join(SCRIPT_DIR, '../../data/raw')

# Directorul destinație (date procesate și împărțite)
DEST_DIR = os.path.join(SCRIPT_DIR, '../../data/processed')

# Raportul de split: (train, validation, test)
# 80% train, 10% validation, 10% test
SPLIT_RATIO = (0.8, 0.1, 0.1)


# ==========================================================
# FUNCȚIE: CREARE STRUCTURĂ DIRECTOARE
# Șterge orice versiune anterioară și reconstruiește
# structura necesară pentru ImageFolder.
# ==========================================================
def create_dirs():

    # Dacă folderul processed există deja, îl ștergem complet
    # pentru a evita amestecarea datelor vechi cu cele noi
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)

    # Creăm structura:
    # processed/
    # ├── train/
    # ├── validation/
    # └── test/
    #     ├── Open/
    #     └── Closed/
    for split in ['train', 'validation', 'test']:
        for class_name in ['Open', 'Closed']:
            os.makedirs(os.path.join(DEST_DIR, split, class_name), exist_ok=True)


# ==========================================================
# FUNCȚIE: SPLIT DATASET
# Împarte imaginile pe clase în train/val/test conform
# proporțiilor definite în SPLIT_RATIO.
# ==========================================================
def split_data():

    classes = ['Open', 'Closed']

    for class_name in classes:

        # Calea către imaginile brute ale clasei curente
        src_path = os.path.join(SOURCE_DIR, class_name)

        # Selectăm doar fișiere imagine valide
        images = [
            f for f in os.listdir(src_path)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

        # Amestecăm imaginile pentru a evita bias de ordine
        random.shuffle(images)
        
        # Calculăm numărul de imagini pentru fiecare subset
        train_count = int(len(images) * SPLIT_RATIO[0])
        val_count = int(len(images) * SPLIT_RATIO[1])
        
        # Împărțire efectivă prin slicing
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count + val_count]
        test_imgs = images[train_count + val_count:]
        
        # Funcție internă pentru copiere imagini
        def copy_images(img_list, split_type):
            for img in img_list:
                src = os.path.join(src_path, img)
                dst = os.path.join(DEST_DIR, split_type, class_name, img)

                # Copiem fișierul fără a-l modifica
                shutil.copy(src, dst)
        
        print(f"Procesare clasa {class_name}...")

        # Copiere imagini în folderele corespunzătoare
        copy_images(train_imgs, 'train')
        copy_images(val_imgs, 'validation')
        copy_images(test_imgs, 'test')
        
        # Afișare statistici pentru verificare
        print(f"--> {class_name}: "
              f"Train={len(train_imgs)}, "
              f"Val={len(val_imgs)}, "
              f"Test={len(test_imgs)}")


# ==========================================================
# ENTRY POINT
# Rulează procesul complet de pregătire a datasetului.
# ==========================================================
if __name__ == "__main__":
    print("Începere preprocesare și splituire date...")

    create_dirs()   # Reconstruiește structura directoare
    split_data()    # Împarte și copiază imaginile

    print("Gata! Datele sunt organizate în data/processed.")