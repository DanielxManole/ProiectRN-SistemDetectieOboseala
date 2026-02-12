import torch
import torch.nn as nn

class DrowsinessCNN(nn.Module):
    """
    CNN simplă pentru detectarea oboselii (ochi deschiși/închiși).
    Input: imagine grayscale 64x64
    Output: logits pentru 2 clase (Closed / Open)
    """
    def __init__(self):
        super(DrowsinessCNN, self).__init__()
        
        # ------------------- 1. Straturi Convoluționale -------------------
        # Extrage caracteristici vizuale din imagine
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # Convoluție: 1 canal -> 32 filtre, kernel 3x3
            nn.ReLU(),                        # Funcție de activare ReLU
            nn.MaxPool2d(2, 2),               # MaxPooling: reduce dimensiunea la jumătate (64->32)

            nn.Conv2d(32, 64, 3, padding=1),  # Convoluție: 32 -> 64 filtre
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # MaxPooling: 32x32 -> 16x16

            nn.Conv2d(64, 128, 3, padding=1), # Convoluție: 64 -> 128 filtre
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # MaxPooling: 16x16 -> 8x8
        )

        # ------------------- 2. Straturi Fully Connected -------------------
        # Clasifică caracteristicile extrase în 2 clase (Open / Closed)
        self.fc = nn.Sequential(
            nn.Flatten(),                       # Transformă tensorul 128x8x8 -> 8192
            nn.Linear(128 * 8 * 8, 128),       # Fully connected: 8192 -> 128
            nn.ReLU(),
            nn.Dropout(0.5),                    # Dropout pentru regularizare
            nn.Linear(128, 2)                   # Output final: 2 clase
        )

    # ------------------- Forward Pass -------------------
    def forward(self, x):
        """
        x: Tensor [batch_size, 1, 64, 64]
        return: Tensor [batch_size, 2] (logits)
        """
        return self.fc(self.conv(x))