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

**Exemplu Date Generate (Before/After):**
* **Original:** Imagine clară, frontală.
* **Sintetic:** Aceeași imagine rotită cu 15 grade, cu zgomot de senzor adăugat (simulare condiții de noapte/vibrații).

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.*` (orice extensie)
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Stări tipice pentru un SIA:**
```
IDLE → ACQUIRE_DATA → PREPROCESS → INFERENCE → DISPLAY/ACT → LOG → [ERROR] → STOP
                ↑______________________________________________|
```

**Exemple concrete per domeniu de inginerie:**

#### A. Monitorizare continuă proces industrial (vibrații motor, temperaturi, presiuni):
```
IDLE → START_ACQUISITION → COLLECT_SENSOR_DATA → BUFFER_CHECK → 
PREPROCESS (filtrare, FFT) → RN_INFERENCE → THRESHOLD_CHECK → 
  ├─ [Normal] → LOG_RESULT → UPDATE_DASHBOARD → COLLECT_SENSOR_DATA (loop)
  └─ [Anomalie] → TRIGGER_ALERT → NOTIFY_OPERATOR → LOG_INCIDENT → 
                  COLLECT_SENSOR_DATA (loop)
       ↓ [User stop / Emergency]
     SAFE_SHUTDOWN → STOP
```

#### B. Clasificare imagini defecte producție (suduri, suprafețe, piese):
```
IDLE → WAIT_TRIGGER (senzor trecere piesă) → CAPTURE_IMAGE → 
VALIDATE_IMAGE (blur check, brightness) → 
  ├─ [Valid] → PREPROCESS (resize, normalize) → RN_INFERENCE → 
              CLASSIFY_DEFECT → 
                ├─ [OK] → LOG_OK → CONVEYOR_PASS → IDLE
                └─ [DEFECT] → LOG_DEFECT → TRIGGER_REJECTION → IDLE
  └─ [Invalid] → ERROR_IMAGE_QUALITY → RETRY_CAPTURE (max 3×) → IDLE
       ↓ [Shift end]
     GENERATE_REPORT → STOP
```

#### C. Predicție traiectorii robot mobil (AGV, AMR în depozit):
```
IDLE → LOAD_MAP → RECEIVE_TARGET → PLAN_PATH → 
VALIDATE_PATH (obstacle check) →
  ├─ [Clear] → EXECUTE_SEGMENT → ACQUIRE_SENSORS (LIDAR, IMU) → 
              RN_PREDICT_NEXT_STATE → UPDATE_TRAJECTORY → 
                ├─ [Target reached] → STOP_AT_TARGET → LOG_MISSION → IDLE
                └─ [In progress] → EXECUTE_SEGMENT (loop)
  └─ [Obstacle detected] → REPLAN_PATH → VALIDATE_PATH
       ↓ [Emergency stop / Battery low]
     SAFE_STOP → LOG_STATUS → STOP
```

#### D. Predicție consum energetic (turbine eoliene, procese batch):
```
IDLE → LOAD_HISTORICAL_DATA → ACQUIRE_CURRENT_CONDITIONS 
(vânt, temperatură, demand) → PREPROCESS_FEATURES → 
RN_FORECAST (24h ahead) → VALIDATE_FORECAST (sanity checks) →
  ├─ [Valid] → DISPLAY_FORECAST → UPDATE_CONTROL_STRATEGY → 
              LOG_PREDICTION → WAIT_INTERVAL (1h) → 
              ACQUIRE_CURRENT_CONDITIONS (loop)
  └─ [Invalid] → ERROR_FORECAST → USE_FALLBACK_MODEL → LOG_ERROR → 
                ACQUIRE_CURRENT_CONDITIONS (loop)
       ↓ [User request report]
     GENERATE_DAILY_REPORT → STOP
```

**Notă pentru proiecte simple:**
Chiar dacă aplicația voastră este o clasificare simplă (user upload → classify → display), trebuie să modelați fluxul ca un State Machine. Acest exercițiu vă învață să gândiți modular și să anticipați toate stările posibile (inclusiv erori).

**Legendă obligatorie (scrieți în README):**
```markdown
### Justificarea State Machine-ului ales:

Am ales arhitectura [descrieți tipul: monitorizare continuă / clasificare la senzor / 
predicție batch / control în timp real] pentru că proiectul nostru [explicați nevoia concretă 
din tabelul Secțiunea 1].

Stările principale sunt:
1. [STARE_1]: [ce se întâmplă aici - ex: "achiziție 1000 samples/sec de la accelerometru"]
2. [STARE_2]: [ce se întâmplă aici - ex: "calcul FFT și extragere 50 features frecvență"]
3. [STARE_3]: [ce se întâmplă aici - ex: "inferență RN cu latență < 50ms"]
...

Tranzițiile critice sunt:
- [STARE_A] → [STARE_B]: [când se întâmplă - ex: "când buffer-ul atinge 1024 samples"]
- [STARE_X] → [ERROR]: [condiții - ex: "când senzorul nu răspunde > 100ms"]

Starea ERROR este esențială pentru că [explicați ce erori pot apărea în contextul 
aplicației voastre industriale - ex: "senzorul se poate deconecta în mediul industrial 
cu vibrații și temperatură variabilă, trebuie să gestionăm reconnect automat"].

Bucla de feedback [dacă există] funcționează astfel: [ex: "rezultatul inferenței 
actualizează parametrii controlerului PID pentru reglarea vitezei motorului"].
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Streamlit, Gradio, FastAPI, Flask, Dash | WebVI sau Web Publishing Tool | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [ ] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [ ] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [ ] Include minimum 40% date originale în dataset-ul final
- [ ] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [ ] Arhitectură RN definită și compilată fără erori
- [ ] Model poate fi salvat și reîncărcat
- [ ] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [ ] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [ ] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
proiect-rn-[nume-prenume]/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/  # Date originale
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [ ] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [ ] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [ ] Cod generare/achiziție date funcțional și documentat
- [ ] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [ ] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [ ] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [ ] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [ ] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [ ] Produce minimum 40% date originale din dataset-ul final
- [ ] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [ ] Documentație în `src/data_acquisition/README.md` cu:
  - [ ] Metodă de generare/achiziție explicată
  - [ ] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [ ] Justificare relevanță date pentru problema voastră
- [ ] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [ ] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [ ] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [ ] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [ ] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [ ] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


