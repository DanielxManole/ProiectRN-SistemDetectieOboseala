# ==============================================================
# SISTEM DETECȚIE OBOSEALĂ ȘOFER
# --------------------------------------------------------------
# Sistem hibrid bazat pe:
#   1. EAR (Eye Aspect Ratio) - metodă geometrică
#   2. CNN (Convolutional Neural Network) - clasificare imagine
#
# Logica finală este de tip OR (SAU):
#   Dacă oricare metodă indică ochi închis → crește scorul.
#
# Include:
#   - smoothing temporal pe mai multe frame-uri
#   - warmup inițial pentru stabilizare
#   - detecție dual/single eye
#   - alarmă sonoră pe thread separat
# ==============================================================

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
import time
from collections import deque

# --- IMPORT MODEL ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SRC_PATH not in sys.path: sys.path.append(SRC_PATH)

from neural_network.model import DrowsinessCNN


MODEL_PATH = os.path.join(CURRENT_DIR, '../../models/optimized_model.pth')

SWAP_INDEX = 1              # Dacă modelul a fost antrenat cu etichete inversate, setează la 0
EAR_THRESHOLD = 0.16        # Sub această valoare = ochi închis (Prag geometric sub care ochiul este considerat închis)
CNN_THRESHOLD = 0.85        # Peste această valoare = ochi închis (Probabilitatea minimă pentru clasa "closed" peste care modelul declară ochiul închis)
ALARM_LIMIT = 40            # Scor necesar pentru alarmă (mai mic = mai sensibil) (Previne alarmele false la clipit natural)

# Setări avansate
USE_SINGLE_EYE = True       # Permite detecție pe un singur ochi
SINGLE_EYE_CONFIDENCE = 0.7 # Cât de mult să aibă încredere în un ochi
SMOOTH_FRAMES = 12           # Câte frame-uri pentru filtrare
MIN_EYE_WIDTH = 14          # Pixeli minim pentru ochi valid

# Mod diagnostic
SHOW_DEBUG_INFO = False      # Arată valori EAR/CNN pe ecran
DEBUG_TO_CONSOLE = False    # Print în consolă

# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EyeTracker:
    """
    Tracker pentru un ochi individual cu smoth filtering.

    Rol:
    - Păstrează un istoric al valorilor EAR și CNN
    - Aplică filtrare temporală (median/mean smoothing)
    - Reduce zgomotul și variațiile bruște între frame-uri

    """
    def __init__(self, history_size=SMOOTH_FRAMES):
        self.ear_history = deque(maxlen=history_size)
        self.cnn_history = deque(maxlen=history_size)
        
    def add_measurement(self, ear, cnn_prob):
        self.ear_history.append(ear)
        self.cnn_history.append(cnn_prob)
        
    def get_smooth_ear(self):
        if not self.ear_history: return 0.3
        return np.median(list(self.ear_history))
    
    def get_smooth_cnn(self):
        if not self.cnn_history: return 0.0
        return np.mean(list(self.cnn_history))
    
    def is_closed(self, ear_thresh, cnn_thresh, warmup_mode=False, relax_threshold=False):
        """
        Determină dacă ochiul este închis folosind:
        - EAR (metrică geometrică)
        - CNN (probabilitate clasificare)
        
        warmup_mode:
            În perioada de inițializare nu permitem detectare.
            
        relax_threshold:
            Folosit în single-eye mode (profil), unde condițiile sunt mai dificile.

        """
        # În perioada de warmup, nu declara niciodată închis
        if warmup_mode:
            return False
        
        ear = self.get_smooth_ear()
        cnn = self.get_smooth_cnn()
        
        # Relaxează threshold-urile pentru single eye mode (condiții mai grele)
        if relax_threshold:
            ear_thresh *= 0.85  # Mai strict pentru EAR (mai mic = mai închis)
            cnn_thresh *= 1.15  # Mai permisiv pentru CNN (mai mare = mai ușor să declare închis)
        
        # Logică OR: oricare dintre ele poate declara închis
        ear_says_closed = ear < ear_thresh
        cnn_says_closed = cnn > cnn_thresh
        
        return ear_says_closed or cnn_says_closed


class DrowsinessDetectionSystem:
    """
    Sistem principal de detecție oboseală.
    
    Responsabilități:
    - Încarcă modelul CNN
    - Inițializează FaceMesh (MediaPipe)
    - Procesează fiecare frame video
    - Calculează scor oboseală
    - Controlează alarma

    """
    def __init__(self):
        print("Initializing Drowsiness Detection System...")
        self.model = self._load_model()
        print("Model loaded")
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        print("Face detection initialized")
        
        self.left_tracker = EyeTracker()
        self.right_tracker = EyeTracker()
        
        self.score = 0
        self.is_drowsy = False
        self.running = True
        self.frame_count = 0
        self.detection_mode = "NONE"
        
        # Warmup pentru a preveni detectări false la start
        self.warmup_frames = SMOOTH_FRAMES * 10  # 120 frames warmup
        self.is_warming_up = True
        
        # Statistici
        self.total_detections = 0
        self.alarm_count = 0
        
        # Start alarm thread
        threading.Thread(target=self._alarm_worker, daemon=True).start()
        print("Alarm system ready\n")

    def _load_model(self):
        m = DrowsinessCNN().to(device)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        m.eval()
        return m

    def _alarm_worker(self):
        """
        Thread separat pentru alarmă.
        
        Avantaj:
        - Nu blochează thread-ul principal video
        - Permite beep repetitiv cât timp sistemul
          este în stare de oboseală

        """
        while self.running:
            if self.is_drowsy:
                winsound.Beep(2500, 200)
                time.sleep(0.15)
            else:
                time.sleep(0.3)

    def compute_ear(self, coords):
        """
        Calculează Eye Aspect Ratio (EAR).
        
        Formula:
            EAR = (dist_vertical_1 + dist_vertical_2)
                  ------------------------------------
                        (2 * dist_horizontal)
        
        EAR scade semnificativ când ochiul este închis.
        """
        try:
            def dist(p1, p2): 
                return np.linalg.norm(np.array(p1) - np.array(p2))
            
            # Landmarks: 0,3=orizontal, 1,2,4,5=vertical
            vertical = dist(coords[1], coords[5]) + dist(coords[2], coords[4])
            horizontal = dist(coords[0], coords[3])
            
            if horizontal < 1e-6: 
                return 0.3
            
            return vertical / (2.0 * horizontal)
        except:
            return 0.3

    def extract_eye_region(self, gray, pts, idxs):
        """
        Extrage regiunea ochiului din imagine.
        
        Include:
        - verificare lățime minimă (evită profil extrem)
        - padding suplimentar pentru context vizual
        - validare limite imagine
        """
        h, w = gray.shape[:2]
        try:
            coords = [(int(pts[i].x * w), int(pts[i].y * h)) for i in idxs]
            
            # Calculăm lățimea reală în pixeli
            min_x = min(c[0] for c in coords)
            max_x = max(c[0] for c in coords)
            eye_width = max_x - min_x
            
            # DACĂ OCHIUL E PREA ÎNGUST (Profil), ÎL IGNORĂM
            if eye_width < MIN_EYE_WIDTH: 
                return None, None
                
            eye_height = max(max(c[1] for c in coords) - min(c[1] for c in coords), 1)
            
            pad_x = int(0.2 * eye_width)
            pad_y = int(0.2 * eye_height)
            
            x1, y1 = max(0, min_x - pad_x), max(0, min(c[1] for c in coords) - pad_y)
            x2, y2 = min(w, max_x + pad_x), min(h, max(c[1] for c in coords) + pad_y)
                
            return (x1, y1, x2, y2), coords
        except:
            return None, None

    def analyze_eye(self, gray, pts, idxs, tracker, side_name, relax_threshold=False):
        """
        Analizează un ochi:
        - calculează EAR
        - extrage imagine 64x64
        - rulează inferență CNN
        - aplică smoothing temporal
        - returnează date complete pentru UI
        """
        box, coords = self.extract_eye_region(gray, pts, idxs)
        if box is None or coords is None:
            if DEBUG_TO_CONSOLE:
                print(f"{side_name} eye not detected")
            return None
        ear = self.compute_ear(coords)
        try:
            x1, y1, x2, y2 = box
            crop = cv2.resize(gray[y1:y2, x1:x2], (64, 64))
            tensor = transforms.ToTensor()(Image.fromarray(crop)).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                prob_closed = probs[SWAP_INDEX].item()
                prob_open = probs[1 - SWAP_INDEX].item()
        except:
            prob_closed = 0.0
            prob_open = 1.0
        tracker.add_measurement(ear, prob_closed)
        is_closed = tracker.is_closed(
            EAR_THRESHOLD, CNN_THRESHOLD, warmup_mode=False, relax_threshold=relax_threshold
        )
        if DEBUG_TO_CONSOLE:
            print(f"{side_name} - EAR: {ear:.3f}, Smooth EAR: {tracker.get_smooth_ear():.3f}, CNN closed: {prob_closed:.3f}, Closed: {is_closed}")
        return {
            'closed': is_closed,
            'box': box,
            'ear': ear,
            'ear_smooth': tracker.get_smooth_ear(),
            'cnn_closed': prob_closed,
            'cnn_open': prob_open,
            'cnn_smooth': tracker.get_smooth_cnn(),
            'side': side_name,
            'width': x2 - x1
        }
    
    def draw_eye_info(self, frame, data, color):
        """Desenare info pentru un ochi"""
        box = data['box']
        
        # Box colorat
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # Status text
        status = "CLOSED" if data['closed'] else "OPEN"
        side_label = 'R' if data['side'] == 'LEFT' else 'L'
        cv2.putText(frame, f"{side_label}: {status}",
                    (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Debug values (opțional)
        if SHOW_DEBUG_INFO:
            debug_text = f"EAR:{data['ear_smooth']:.2f} CNN:{data['cnn_smooth']:.2f}"
            cv2.putText(frame, debug_text, 
                       (box[0], box[3]+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def draw_ui(self, frame):
        """
        Desenează interfața grafică:
        - Header (status sistem)
        - Statistici sesiune
        - Progress bar scor oboseală
        - Threshold-uri curente
        
        Progress bar folosește gradient:
            Verde → Alert
            Galben → Atenție
            Roșu → Oboseală critică

        """
        h, w = frame.shape[:2]
        
        # === HEADER ===
        header_h = 70
        cv2.rectangle(frame, (0, 0), (w, header_h), (30, 30, 30), -1)
        
        # Status principal
        if self.detection_mode == "NO-FACE":
            status_text = "SEARCHING FOR DRIVER..."
            status_color = (150, 150, 150) # Gri (stare neutră)
        elif self.is_drowsy:
            status_text = "DROWSINESS DETECTED!"
            status_color = (0, 0, 255) # Roșu
        elif self.is_warming_up:
            status_text = "WARMING UP..."
            status_color = (0, 255, 255) # Galben
        else:
            status_text = "DRIVER ALERT" # Doar dacă vede fața și e ok
            status_color = (0, 255, 0) # Verde
        
        cv2.putText(frame, status_text, (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Info secundară
        cv2.putText(frame, f"Mode: {self.detection_mode}", 
                   (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        # Statistici
        stats_x = w - 250
        cv2.putText(frame, f"Frames: {self.frame_count}", 
                   (stats_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (150, 150, 150), 1)
        cv2.putText(frame, f"Alarms: {self.alarm_count}", 
                   (stats_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (150, 150, 150), 1)
        
        # === FOOTER - PROGRESS BAR ===
        bar_h = 35
        bar_y = h - bar_h - 15
        
        # Background
        cv2.rectangle(frame, (50, bar_y), (w-50, bar_y+bar_h), (50, 50, 50), -1)
        
        # Fill based on score
        fill_ratio = min(self.score / ALARM_LIMIT, 1.0)
        bar_width = int(fill_ratio * (w - 100))
        
        # Color gradient: verde -> galben -> roșu
        if fill_ratio < 0.5:
            bar_color = (0, 255, 0)  # verde
        elif fill_ratio < 0.8:
            bar_color = (0, 255, 255)  # galben
        else:
            bar_color = (0, 0, 255)  # roșu
        
        cv2.rectangle(frame, (50, bar_y), (50 + bar_width, bar_y+bar_h), 
                     bar_color, -1)
        
        # Score text
        score_text = f"{self.score}/{ALARM_LIMIT}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = w//2 - text_size[0]//2
        cv2.putText(frame, score_text, (text_x, bar_y+24), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # === SETTINGS INFO ===
        settings_y = header_h + 20
        if self.is_warming_up:
            warmup_progress = int((self.frame_count / self.warmup_frames) * 100)
            cv2.putText(frame, f"Warmup: {warmup_progress}% ({self.frame_count}/{self.warmup_frames} frames)", 
                       (15, settings_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Thresholds: EAR<{EAR_THRESHOLD:.2f} CNN>{CNN_THRESHOLD:.2f}", 
                       (15, settings_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (100, 100, 255), 1)
        
        return frame

    def run(self):
        """
        Loop principal aplicație.
        
        Pași:
        1. Captură frame
        2. Detectare landmarks față
        3. Analiză ochi stâng/drept
        4. Calcul scor oboseală
        5. Actualizare UI
        6. Verificare ieșire
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        L_EYE = [362, 385, 387, 263, 373, 380]
        R_EYE = [33, 160, 158, 133, 153, 144]

        print(f"🎥 Camera active - Warmup {self.warmup_frames} frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.is_warming_up and self.frame_count >= self.warmup_frames:
                self.is_warming_up = False
                print("✓ Warmup complete")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            left_data = None
            right_data = None
            both_closed = False
            confidence = 0

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Analizăm ambii ochi
                left_data = self.analyze_eye(gray, landmarks, L_EYE, self.left_tracker, "LEFT")
                right_data = self.analyze_eye(gray, landmarks, R_EYE, self.right_tracker, "RIGHT")

                # Verificăm validitatea rezultatelor
                l_valid = left_data is not None
                r_valid = right_data is not None

                if l_valid and r_valid:
                    self.detection_mode = "DUAL"
                    # În DUAL mode, alarmăm doar dacă AMBII ochi sunt închiși
                    both_closed = left_data['closed'] and right_data['closed']
                    confidence = 1.0
                elif l_valid or r_valid:
                    # Suntem în profil - folosim ochiul disponibil
                    visible_eye = left_data if l_valid else right_data
                    self.detection_mode = "SINGLE-" + visible_eye['side']
                    both_closed = visible_eye['closed']
                    confidence = SINGLE_EYE_CONFIDENCE
                else:
                    self.detection_mode = "NONE"
                    both_closed = False
                    confidence = 0

                # Desenează doar ce e valid
                if l_valid: frame = self.draw_eye_info(frame, left_data, (0,255,0) if not left_data['closed'] else (0,0,255))
                if r_valid: frame = self.draw_eye_info(frame, right_data, (0,255,0) if not right_data['closed'] else (0,0,255))
            else:
                self.detection_mode = "NO-FACE"
                both_closed = False
                confidence = 0
                # Opțional: resetăm scorul mai repede dacă șoferul a dispărut
                self.score = max(0, self.score - 5)

            # Scoring logic
            # ==================================================
            # LOGICA DE SCOR
            #
            # Dacă ambii ochi (sau unul în single mode) sunt închiși:
            #   → scorul crește gradual
            #
            # Dacă ochii sunt deschiși:
            #   → scorul scade progresiv
            #
            # Sistemul declară oboseală doar când scorul
            # atinge ALARM_LIMIT.
            #
            # Această metodă reduce:
            #   - alarme false la clipit
            #   - fluctuații scurte
            # ==================================================
            if not self.is_warming_up:
                if both_closed and confidence > 0:
                    increment = int(3 * confidence)
                    self.score = min(ALARM_LIMIT, self.score + increment)
                else:
                    self.score = max(0, self.score - 2)
                was_drowsy = self.is_drowsy
                self.is_drowsy = self.score >= ALARM_LIMIT
                if self.is_drowsy and not was_drowsy:
                    self.alarm_count += 1
                    print(f"🚨 ALARM #{self.alarm_count} triggered at frame {self.frame_count}")
            else:
                self.score = 0
                self.is_drowsy = False

            frame = self.draw_ui(frame)
            cv2.imshow('Retele Neuronale - Sistem Detectie Oboseala', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print("\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"Total frames: {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        print(f"Total alarms: {self.alarm_count}")
        if self.total_detections > 0:
            print(f"Alarm rate: {self.alarm_count/self.total_detections*100:.2f}%")
        print("="*60 + "\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DROWSINESS DETECTION SYSTEM")
    print("="*60)
    print("\nCurrent Configuration:")
    print(f"   EAR Threshold: {EAR_THRESHOLD}")
    print(f"   CNN Threshold: {CNN_THRESHOLD}")
    print(f"   Alarm Limit: {ALARM_LIMIT}")
    print(f"   Single Eye Mode: {'Enabled' if USE_SINGLE_EYE else 'Disabled'}")
    print(f"   Smooth Frames: {SMOOTH_FRAMES}")
    print(f"   Debug Info: {'Visible' if SHOW_DEBUG_INFO else 'Hidden'}")
    print("="*60 + "\n")
    
    try:
        system = DrowsinessDetectionSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()