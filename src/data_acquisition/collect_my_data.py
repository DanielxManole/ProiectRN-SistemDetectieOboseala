import cv2
import os
import time

# ==========================================================
# CONFIGURARE CĂI DIRECTOARE
# Determinăm automat locația scriptului și construim
# structura pentru salvarea imaginilor brute colectate.
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '../data/raw')

# Director pentru ochi deschiși
OPEN_DIR = os.path.join(DATA_DIR, 'MyOpen')

# Director pentru ochi închiși
CLOSED_DIR = os.path.join(DATA_DIR, 'MyClosed')

# Creăm directoarele dacă nu există deja
os.makedirs(OPEN_DIR, exist_ok=True)
os.makedirs(CLOSED_DIR, exist_ok=True)

# ==========================================================
# ÎNCĂRCARE MODELE HAAR CASCADE
# Folosim clasificatoare pre-antrenate pentru:
# - detectarea feței
# - detectarea ochilor
# ==========================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# ==========================================================
# INITIALIZARE CAMERA VIDEO
# Încercăm camera 0, dacă nu funcționează încercăm camera 1.
# CAP_DSHOW reduce latența pe Windows.
# ==========================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("\n--- COLECTARE DATE ORIGINALE ---")
print("1. Ține ochii DESCHIȘI și apasă tasta 'o'")
print("2. Ține ochii ÎNCHIȘI și apasă tasta 'c'")
print("3. Apasă 'q' pentru a ieși")

# Numărăm imaginile existente pentru a continua numerotarea
count_open = len(os.listdir(OPEN_DIR))
count_closed = len(os.listdir(CLOSED_DIR))

# ==========================================================
# LOOP PRINCIPAL
# Capturăm frame-uri în timp real până la apăsarea tastei 'q'
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Oglindire imagine pentru experiență naturală (mirror effect)
    frame = cv2.flip(frame, 1)

    # Conversie la grayscale pentru detecție mai eficientă
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Copie pentru desenarea dreptunghiurilor (fără a altera originalul)
    clone = frame.copy()

    # Detectare fețe în imagine
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    detected_eye = None  # Variabilă pentru ochiul detectat
    
    # ======================================================
    # DETECTARE OCHI ÎN INTERIORUL FEȚEI
    # ======================================================
    for (x, y, w, h) in faces:
        # Desenăm dreptunghi pe față
        cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 0, 0), 1)

        roi_gray = gray[y:y+h, x:x+w]
        
        # Detectare ochi în regiunea feței
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        for (ex, ey, ew, eh) in eyes:
            # Selectăm doar ochii din jumătatea superioară a feței
            # pentru a evita detectarea gurii sau a altor regiuni
            if ey < h / 2:
                cv2.rectangle(
                    clone[y:y+h, x:x+w],
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (0, 255, 0),
                    2
                )

                # Extragem zona ochiului pentru salvare
                detected_eye = roi_gray[ey:ey+eh, ex:ex+ew]
                break

    # Afișăm contorul de imagini salvate
    cv2.putText(
        clone,
        f"Open: {count_open} | Closed: {count_closed}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Colectare Date", clone)
    
    key = cv2.waitKey(1) & 0xFF

    # ======================================================
    # SALVARE IMAGINI
    # Salvăm doar dacă un ochi este detectat corect.
    # ======================================================
    if detected_eye is not None:
        # Timestamp pentru a evita suprascrierea fișierelor
        timestamp = int(time.time() * 1000)
        
        if key == ord('o'):
            path = os.path.join(OPEN_DIR, f"me_open_{timestamp}.jpg")
            cv2.imwrite(path, detected_eye)
            count_open += 1
            print(f"--> Salvat DESCHIS! (Total: {count_open})")
            
        elif key == ord('c'):
            path = os.path.join(CLOSED_DIR, f"me_closed_{timestamp}.jpg")
            cv2.imwrite(path, detected_eye)
            count_closed += 1
            print(f"--> Salvat ÎNCHIS! (Total: {count_closed})")

    # Dacă utilizatorul apasă tastă dar nu este detectat ochiul
    elif (key == ord('o') or key == ord('c')) and detected_eye is None:
        print("Nu văd ochiul! Apropie-te sau stai nemișcat.")

    # Ieșire din aplicație
    if key == ord('q'):
        break

# ==========================================================
# ELIBERARE RESURSE
# ==========================================================
cap.release()
cv2.destroyAllWindows()