import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
import winsound
import mediapipe as mp
import threading

# --- CONFIGURARE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, '../models/drowsiness_model.pth')
IMG_SIZE = 64

# SENSIBILITATEA ALARMEI
ALARM_THRESHOLD = 10   
BLINK_RECOVERY = 4     

# PRAGUL DE DECIZIE 
CLOSED_THRESHOLD = 0.10

# Inițializare MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNCȚIE NON-BLOCANTĂ PENTRU SUNET ---
def play_alarm_sound():
    winsound.Beep(3000, 100)

# --- MODELUL ---
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

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("EROARE: Nu găsesc modelul!")
        sys.exit(1)
    model = DrowsinessCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    return model

# --- PREDICTIE ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_eye_prob(eye_img, model):
    try:
        if eye_img.size == 0: return 0.0
        eye_img = cv2.equalizeHist(eye_img)
        eye_pil = Image.fromarray(eye_img)
        input_tensor = transform(eye_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            closed_prob = probs[0][0].item() 
        return closed_prob
    except:
        return 0.0

def get_eye_crop(frame, landmarks, eye_indices, padding=4): 
    h, w = frame.shape[:2]
    x_min, y_min, x_max, y_max = w, h, 0, 0

    for idx in eye_indices:
        pt = landmarks[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    eye_crop = frame[y_min:y_max, x_min:x_max]
    return eye_crop, (x_min, y_min, x_max, y_max)

LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

# --- MAIN ---
def main():
    model = load_model()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    drowsy_score = 0
    
    print("\n--- DeepGuard: Final Version (No Lag) ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        results = face_mesh.process(rgb_frame)
        
        is_drowsy_frame = False
        eyes_found = False
        
        l_prob_txt = "0%"
        r_prob_txt = "0%"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eyes_found = True
                
                l_img, l_rect = get_eye_crop(gray_frame, face_landmarks.landmark, LEFT_EYE_IDXS)
                l_closed_prob = predict_eye_prob(l_img, model)
                
                r_img, r_rect = get_eye_crop(gray_frame, face_landmarks.landmark, RIGHT_EYE_IDXS)
                r_closed_prob = predict_eye_prob(r_img, model)

                l_prob_txt = f"{int(l_closed_prob*100)}%"
                r_prob_txt = f"{int(r_closed_prob*100)}%"

                l_is_closed = l_closed_prob > CLOSED_THRESHOLD
                r_is_closed = r_closed_prob > CLOSED_THRESHOLD

                col_l = (0, 0, 255) if l_is_closed else (0, 255, 0)
                col_r = (0, 0, 255) if r_is_closed else (0, 255, 0)
                
                cv2.rectangle(frame, (l_rect[0], l_rect[1]), (l_rect[2], l_rect[3]), col_l, 1)
                cv2.rectangle(frame, (r_rect[0], r_rect[1]), (r_rect[2], r_rect[3]), col_r, 1)
                
                cv2.putText(frame, l_prob_txt, (l_rect[0], l_rect[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_l, 1)
                cv2.putText(frame, r_prob_txt, (r_rect[0], r_rect[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_r, 1)

                if l_is_closed and r_is_closed:
                    is_drowsy_frame = True

        # --- UPDATE SCOR ---
        if eyes_found:
            if is_drowsy_frame:
                drowsy_score += 1
            else:
                drowsy_score -= BLINK_RECOVERY
        
        if drowsy_score < 0: drowsy_score = 0
        
        # --- ALARMĂ ---
        cv2.putText(frame, f"SOMN: {drowsy_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        max_bar = 200
        current_bar = min(int(drowsy_score * 10), max_bar)
        
        bar_color = (0, 255, 0)
        if drowsy_score > ALARM_THRESHOLD / 2: bar_color = (0, 255, 255)
        if drowsy_score > ALARM_THRESHOLD: bar_color = (0, 0, 255)

        cv2.rectangle(frame, (10, 40), (10 + current_bar, 60), bar_color, -1)
        cv2.rectangle(frame, (10, 40), (10 + max_bar, 60), (255, 255, 255), 1)

        if drowsy_score > ALARM_THRESHOLD:
            cv2.putText(frame, "!!! TREZESTE-TE !!!", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            t = threading.Thread(target=play_alarm_sound)
            t.daemon = True
            t.start()

        cv2.imshow('DeepGuard Final', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()