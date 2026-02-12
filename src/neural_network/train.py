import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pandas as pd
import time
from model import DrowsinessCNN

# ==========================================================
# CONFIGURARE GLOBALĂ TRAINING
# ==========================================================

TARGET_ACCURACY = 1.0   # Oprim antrenarea dacă atingem 100% acuratețe
MIN_EPOCHS = 10         # Număr minim de epoci înainte de a permite oprirea
MAX_EPOCHS = 50         # Limită superioară pentru prevenirea rulării excesive

# Determinare căi proiect
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(SCRIPT_DIR, '../../data/processed/train')
VAL_DIR = os.path.join(SCRIPT_DIR, '../../data/processed/validation')

# Locații salvare model și istoric
MODEL_PATH = os.path.join(SCRIPT_DIR, '../../models/optimized_model.pth')
HISTORY_PATH = os.path.join(SCRIPT_DIR, '../../results/training_history.csv')

# Creăm directoarele dacă nu există
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)


def train():

    # ==========================================================
    # 1. PREPROCESARE DATE
    # IMPORTANT: Transformările trebuie să fie identice cu cele
    # folosite ulterior la inferență și evaluare.
    # ==========================================================
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),                 # Redimensionare la dimensiunea input
        transforms.Grayscale(num_output_channels=1), # Conversie la 1 canal
        transforms.ToTensor(),                       # Conversie la tensor [0,1]
        transforms.Normalize([0.5], [0.5])           # Normalizare în interval [-1, 1]
    ])

    # Încărcare dataset folosind ImageFolder
    train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, data_transforms)

    # DataLoader gestionează batching și paralelizarea încărcării
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ==========================================================
    # 2. INITIALIZARE MODEL ȘI OPTIMIZARE
    # ==========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsinessCNN().to(device)

    # CrossEntropyLoss este standard pentru clasificare multi-clasă
    criterion = nn.CrossEntropyLoss()

    # Adam – optimizer adaptiv, convergență rapidă
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = []  # Salvăm metricile fiecărei epoci

    # Early Stopping configurare
    best_val_loss = float('inf')
    patience = 5
    trigger_times = 0
    
    start_total_time = time.time()

    print(f"Pornire antrenament pe: {str(device).upper()}")
    print("-" * 50)

    # ==========================================================
    # 3. LOOP PRINCIPAL DE ANTRENARE
    # ==========================================================
    for epoch in range(MAX_EPOCHS):

        start_epoch_time = time.time()
        
        # -------- TRAINING MODE --------
        model.train()
        train_loss, train_correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()     # Resetăm gradientul
            outputs = model(images)   # Forward pass
            loss = criterion(outputs, labels)

            loss.backward()           # Backpropagation
            optimizer.step()          # Actualizare greutăți

            train_loss += loss.item()

            # Calculăm numărul de predicții corecte
            train_correct += (
                (outputs.argmax(1) == labels)
                .type(torch.float)
                .sum()
                .item()
            )
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_dataset)

        # -------- VALIDATION MODE --------
        model.eval()
        val_loss, val_correct = 0, 0

        # Dezactivăm gradient computation pentru eficiență
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (
                    (outputs.argmax(1) == labels)
                    .type(torch.float)
                    .sum()
                    .item()
                )
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_dataset)
        
        epoch_duration = time.time() - start_epoch_time

        print(f"Epoch {epoch+1}: "
              f"TrainAcc={train_acc:.2%}, "
              f"ValAcc={val_acc:.2%}, "
              f"ValLoss={avg_val_loss:.4f}, "
              f"Timp={epoch_duration:.2f}s")
        
        # Salvăm istoricul pentru analiză ulterioară
        history.append([
            epoch+1,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            round(epoch_duration, 2)
        ])

        # ======================================================
        # 4. SALVARE MODEL + EARLY STOPPING
        # ======================================================

        # Salvăm modelul doar dacă val_loss s-a îmbunătățit
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model salvat (ValLoss minim: {best_val_loss:.4f})")
            trigger_times = 0
        else:
            trigger_times += 1

        # Permitem oprirea doar după MIN_EPOCHS
        if (epoch + 1) >= MIN_EPOCHS:

            # Oprire dacă atingem target accuracy
            if val_acc >= TARGET_ACCURACY:
                print(f"Acuratețe 100% atinsă la epoca {epoch+1}!")
                break

            # Early stopping dacă nu se mai îmbunătățește validarea
            if trigger_times >= patience:
                print(f"Early stopping la epoca {epoch+1} "
                      f"(Nu s-a mai îmbunătățit ValLoss)")
                break

    # ==========================================================
    # 5. FINALIZARE + SALVARE ISTORIC
    # ==========================================================
    total_duration = (time.time() - start_total_time) / 60
    print(f"\nGATA! Timp total: {total_duration:.2f} minute")

    # Salvăm istoricul într-un CSV pentru generare grafice
    df = pd.DataFrame(
        history,
        columns=[
            'epoch',
            'train_loss',
            'train_acc',
            'val_loss',
            'val_acc',
            'duration_sec'
        ]
    )

    df.to_csv(HISTORY_PATH, index=False)
    print(f"Istoric salvat în {HISTORY_PATH}")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    train()