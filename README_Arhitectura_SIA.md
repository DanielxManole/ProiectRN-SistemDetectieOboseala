# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Manole Daniel]  
**[Link Repository GitHub](https://github.com/DanielxManole/ProiectRN-SistemDetectieOboseala)**
**Data:** [11.12.2025]
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software
____________________________________________________________________________________________________________________________________________________________________________________________________
|                  **Nevoie reală concretă**                     |               **Cum o rezolvă SIA-ul vostru**                  |               **Modul software responsabil**                   |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Detectarea stării de somnolență (micro-somn) a șoferului în    | Monitorizare video continuă (30 FPS) → Clasificare stare ochi  | **Modul Achiziție (OpenCV)** + **Rețea Neuronală (CNN)**       |
| timp real pentru prevenirea accidentelor rutiere.              | (Deschis/Închis) cu latență de procesare < 100ms per cadru.    |                                                                |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Distincția precisă între clipitul natural și adormire, pentru  | Analiză temporală a secvenței video: declanșare alertă doar la | **Modul Logică de Control** (Scor Oboseală)                    |
| evitarea alarmelor false.                                      | depășirea unui prag de 10-15 cadre consecutive (≈ 0.5 secunde) |                                                                |
|                                                                | de ochi închiși.                                               |                                                                |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Alertarea imediată a șoferului în momentul detectării          | Generare semnal acustic strident și avertisment vizual         | **Modul Alertare (Threading + UI)**                            |
| pericolului, fără a bloca monitorizarea vizuală.               | (UI Roșu) pe un fir de execuție paralel (Threading), cu timp   |                                                                |
|                                                                | de reacție < 0.1 secunde.                                      |                                                                |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| Funcționare robustă indiferent de poziția capului sau          | Segmentare facială avansată (MediaPipe 468 puncte) și          | **Modul Preprocesare** + **MediaPipe**                         |
| iluminare variabilă.                                           | normalizare histogramă → Acuratețe de validare > 95% pe setul  |                                                                |
|                                                                | de testare.                                                    |                                                                |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Statistici Finale Dataset:**
* **Total imagini antrenare/validare/test:** 12,000 imagini (aprox.)
* **Imagini Originale (Proprii + Augmentate):** 6,000 imagini
* **Procent Contribuție Proprie:** **50%**

Pentru a asigura un grad ridicat de originalitate și robustețe a modelului, am adoptat o abordare hibridă în trei pași:

1.  **Achiziție Primară (Data Collection):**
    Am dezvoltat un script dedicat (`src/collect_my_data.py`) bazat pe detectoare Haar Cascade, cu care am achiziționat un set inițial de **600 de imagini brute** (300 Open / 300 Closed) având subiectul propriu în condiții reale de iluminare și poziționare.

2.  **Generare Sintetică (Data Augmentation Offline):**
    Deoarece datele brute erau insuficiente pentru Deep Learning, am implementat un pipeline de augmentare (`src/augment_data.py`) care a generat variații sintetice ale datelor proprii. Transformările aplicate au inclus:
    * Rotații aleatorii (+/- 15 grade) pentru simularea înclinării capului.
    * Ajustări de luminozitate și contrast (ColorJitter) pentru simularea condițiilor de zi/noapte.
    * Adăugare de zgomot Gaussian și Blur pentru simularea camerelor web de slabă calitate.
    * *Rezultat:* Multiplicarea datelor proprii de la 600 la **6,000 de observații unice**.

3.  **Integrare și Balansare:**
    Datasetul final a fost construit prin mixarea datelor generate anterior cu un subset aleatoriu din **MRL Eye Dataset**, respectiv **Kaggle**, păstrând o proporție echilibrată (50% Original / 50% Public) pentru a preveni bias-ul (supra-adaptarea pe o singură persoană) și a asigura generalizarea modelului.

____________________________________________________________________________________________________________________________________________________________________________________________________
|                   **Tip contribuție ales**                     |               **Implementare în Proiect (Dovada)**             |                 **Locație în Repository**                      |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| **Date achiziționate cu senzori proprii**                      | - Achiziție a **600 imagini brute** (300 Open / 300 Closed)    | `src/collect_my_data.py` `data/raw/MyOpen` `data/raw/MyClosed` |
|                                                                | folosind camera web a laptopului.<br>• Etichetare manuală în   |                                                                |
|                                                                | timp real (tasta 'o'/'c') prin script dedicat.                 |                                                                |
|                                                                | - Protocol: Iluminare variabilă (naturală/artificială),        |                                                                |
|                                                                | distanță 30-50cm față de senzor.                               |                                                                |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| **Date sintetice (Augmentare)**                                | - Generare a **5400 imagini sintetice** derivate din cele raw  | `src/augment_data.py` `data/raw/Open` (cu prefix `my_aug_`     |
|                                                                | - Metode: Rotație afină, Zgomot Gaussian, Blur, Expunere       |                                                                |
|                                                                | - Validare: Creșterea acurateței modelului de la 85% (doar     |                                                                |
|                                                                | date brute) la 99% (date augmentate).                          |                                                                |
|----------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|

**Detalii Protocol Achiziție (Dovada Experimentală):**
1.  **Setup:** Laptop cu cameră web integrată (720p), poziționat la nivelul ochilor.
2.  **Software:** Script propriu (`collect_my_data.py`) care utilizează Haar Cascades pentru a decupa automat regiunea de interes (ROI) a ochiului și a o salva doar la confirmarea manuală a utilizatorului.
3.  **Procesare:** Imaginile au fost salvate direct în format decupat (ROI), color, cu timestamp unic pentru a garanta trasabilitatea.

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

![Diagrama State Machine](docs/state_machine.svg)

**Arhitectura fluxului de date:**

```text
IDLE → INITIALIZE_SYSTEM (Load CNN Model, Warm-up Camera) → ACQUIRE_FRAME → 
DETECT_FACE_MESH (MediaPipe) → 
  ├─ [No Face Detected] → DISPLAY_FRAME → ACQUIRE_FRAME (loop)
  └─ [Face Detected] → EXTRACT_EYE_ROI → PREPROCESS (Grayscale, Resize 64x64) → 
                     RN_INFERENCE (CNN Prediction) → UPDATE_DROWSINESS_SCORE → 
                     CHECK_THRESHOLD (Score > 15?) → 
                       ├─ [Safe Condition] → DRAW_OVERLAY (Green UI) → DISPLAY_FRAME → 
                       │                     ACQUIRE_FRAME (loop)
                       └─ [Drowsy Detected] → TRIGGER_ALARM_STATE → 
                                            DRAW_WARNING (Red UI) → 
                                            START_AUDIO_THREAD (Non-blocking) → 
                                            ACQUIRE_FRAME (loop)
       ↓ [User Interrupt 'q']
     RELEASE_RESOURCES → STOP
```
**Legendă obligatorie (scrieți în README):**

### Justificarea State Machine-ului ales:

Am ales arhitectura de monitorizare continuă în timp real cu procesare secvențială pentru că proiectul nostru vizează siguranța rutieră și necesită o latență minimă între achiziția imaginii și decizia de alertare, precum și o filtrare temporală pentru evitarea alarmelor false (clipit vs. somn).

Stările principale sunt:
1. ACQUIRE_FRAME: Captura sincronă a fluxului video de la camera web la 30 FPS.
2. DETECT_FACE & PREPROCESS: Localizarea feței folosind MediaPipe și extragerea regiunii de interes (ROI) a ochiului, urmată de conversia în Grayscale și redimensionarea la 64x64 pixeli pentru a se potrivi cu intrarea rețelei neuronale.
3. RN_INFERENCE: Propagarea imaginii prin rețeaua CNN antrenată, care returnează o probabilitate (0.0 - 1.0) pentru clasa "Closed".
4. UPDATE_SCORE & LOGIC: Actualizarea unui contor intern (buffer temporal). Dacă ochiul este clasificat "Închis", scorul crește; dacă este "Deschis", scorul scade rapid (Blink Recovery).
5. TRIGGER_ALARM: Starea de alertă maximă care activează interfața grafică roșie și lansează sunetul de avertizare pe un fir de execuție paralel.

Tranzițiile critice sunt:
- [DETECT_FACE] → [ACQUIRE_FRAME]: Se întâmplă când șoferul își întoarce capul sau camera este obturată. Sistemul intră într-o buclă de așteptare (Skip Frame) fără a bloca aplicația sau a da crash.
- [CHECK_THRESHOLD] → [TRIGGER_ALARM]: Se întâmplă strict când variabila drowsy_score depășește pragul de 15 cadre consecutive. Aceasta este tranziția critică de siguranță care separă starea de veghe de cea de pericol.

Starea ERROR (sau fail-safe) este esențială pentru că în contextul utilizării la volan, condițiile de iluminare pot varia drastic (intrare în tunel, noapte). Dacă DETECT_FACE eșuează din cauza întunericului, sistemul nu trebuie să se oprească (crash), ci să reia ciclu de achiziție (ACQUIRE_FRAME) până la restabilirea vizibilității.

Bucla de feedback funcționează astfel: rezultatul inferenței curente (Ochi Închis/Deschis) actualizează variabila de stare drowsy_score (istoricul recent), care la rândul ei dictează comportamentul sistemului în cadrul următor (histerezis), prevenind oscilațiile rapide ale alarmei.

---

### 4. Scheletul Complet al Modulelor Software

Toate cele 3 module sunt implementate, integrate și rulează fără erori.

| **Modul** | **Implementare (Fișiere)** | **Status Funcționalitate (La predare)** |
|-----------|----------------------------|-----------------------------------------|
| **1. Data Logging / Acquisition** | `src/collect_my_data.py`<br>`src/augment_data.py` | **[x] FINALIZAT.** Scriptul de achiziție rulează stabil, iar pipeline-ul de augmentare a generat peste 6,000 de imagini (CSV-ul este înlocuit de structura de directoare standard `ImageFolder` compatibilă PyTorch). |
| **2. Neural Network Module** | `src/train_model.py`<br>`models/drowsiness_model.pth` | **[x] FINALIZAT.** Modelul CNN (Clasa `DrowsinessCNN`) este definit, compilat și salvat. Antrenamentul ajunge la convergență (Loss scăzut). |
| **3. Web Service / UI** | `src/webcam_detect.py` | **[x] FINALIZAT.** Aplicația desktop (bazată pe OpenCV) preia fluxul video, rulează inferența în timp real și afișează overlay-ul grafic + alerte sonore. |

#### Detalii per modul (Checklist de verificare):

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități implementate:**
- [x] **Codul rulează fără erori:** `python src/collect_my_data.py` deschide camera și salvează ROI-uri corecte.
- [x] **Compatibilitate:** Generează structura de directoare (`raw/MyOpen`, `raw/MyClosed`) compatibilă 100% cu scriptul de preprocesare din Etapa 3 (`torchvision.datasets.ImageFolder`).
- [x] **Originalitate:** Datasetul final conține 50% date proprii (originale + augmentate).
- [x] **Documentație:** Codul conține comentarii detaliate despre parametrii de augmentare (rotire +/- 15 grade, zgomot Gaussian).

#### **Modul 2: Neural Network Module**

**Funcționalități implementate:**
- [x] **Arhitectură Definită:** Clasa `DrowsinessCNN` (3 straturi Convoluționale + 2 straturi Fully Connected + Dropout) este definită explicit în `train_model.py`.
- [x] **Persistență:** Modelul este salvat automat în `models/drowsiness_model.pth` și reîncărcat cu succes de aplicația de detecție.
- [x] **Justificare:** Arhitectura aleasă este un CNN clasic, optimizat pentru viteză (inference time mic) și rezoluție scăzută (64x64), ideal pentru procesare real-time pe CPU.
- [x] **Status Antrenare:** Modelul este funcțional (weights inițializați și antrenați preliminar).

#### **Modul 3: Web Service / UI**

**Funcționalități implementate:**
- [x] **Input User:** Flux video live de la camera web (Selectabil ID 0 sau 1).
- [x] **Output:** 1. Bounding box colorat (Verde/Roșu) în jurul ochilor.
    2. Bară de progres pentru "Scorul de Oboseală".
    3. Mesaj de text "TREZESTE-TE" și alertă sonoră.
- [x] **Dovada:** Screenshot demonstrativ inclus în `docs/ui_demo.png`.

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [x] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [x] Cod generare/achiziție date funcțional și documentat
- [x] Dovezi contribuție originală: grafice + log + statistici în `docs/` (vezi `setup_achizitie.png`)
- [x] Diagrama State Machine creată și salvată în `docs/state_machine.png`
- [x] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [x] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/collect_my_data.py`)
- [x] Produce minimum 40% date originale din dataset-ul final (50% realizat)
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3 (Structură de foldere ImageFolder)
- [x] Documentație în `src/README.md` (sau folder dedicat) cu:
  - [x] Metodă de generare/achiziție explicată
  - [x] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [x] Justificare relevanță date pentru problema voastră
- [x] Fișiere în `data/raw/MyOpen` și `data/raw/MyClosed` conform structurii

### Modul 2: Neural Network
- [x] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [x] README în `src/neural_network/` (sau `src/`) cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [x] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [x] Screenshot demonstrativ în `docs/ui_demo.png`
- [x] README în `src/app/` (sau `src/`) cu instrucțiuni lansare (comenzi exacte)
