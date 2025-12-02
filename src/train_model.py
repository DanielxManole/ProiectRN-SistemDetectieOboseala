import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. CONFIGURARE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../data/processed')
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, '../models')
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'drowsiness_model.pth')

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Rulez antrenarea pe: {device}")

# --- 2. TRANSFORMARE ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 3. DATALOADERS ---
print("Încărcare date...")
try:
    train_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_data = datasets.ImageFolder(os.path.join(DATA_DIR, 'validation'), transform=val_transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print("Eroare la încărcare date. Verifică folderele.")
    exit()

# --- 4. MODEL ---
class DrowsinessCNN(nn.Module):
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DrowsinessCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. ANTRENARE CU STOP LA 100% ---
print(f"Start Antrenare (Stop la 100%)...")

best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validare
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    avg_loss = running_loss/len(train_loader)
    
    print(f"Epoca {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acuratețe: {accuracy:.2f}%")
    
    # Salvare și Verificare Stop
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"    --> Model Salvat! ({accuracy:.2f}%)")
        
        # --- STOP LA PERFECȚIUNE ---
        if accuracy >= 100.0:
            print("\n!!! Acuratețe PERFECTĂ (100%)! Mă opresc aici. !!!")
            break

print(f"Gata! Modelul final e în models/drowsiness_model.pth")