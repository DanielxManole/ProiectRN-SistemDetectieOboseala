# README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale
**Instituție:** POLITEHNICA București – FIIR
**Student:** Manole Daniel
**Link Repository GitHub:** [https://github.com/DanielxManole/ProiectRN-SistemDetectieOboseala](https://github.com/DanielxManole/ProiectRN-SistemDetectieOboseala)
**Data predării:** 15.01.2026

---

## PREREQUISITE – Verificare Etapa 4 (FINALIZAT)

- [x] **State Machine** definit în `docs/state_machine.svg`
- [x] **Contribuție 45.2% date originale** (6.600 imagini proprii/augmentate)
- [x] **Modul 1 (Data Logging)** funcțional (`collect_my_data.py`, `augment_data.py`)
- [x] **Modul 2 (RN)** Arhitectură definită în PyTorch
- [x] **Modul 3 (UI)** Aplicație webcam funcțională (inițial cu model dummy)
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare 

Am utilizat dataset-ul finalizat în Etapa 4, care conține **14.600 de eșantioane**. Ne-am folosit de script-ul realizat în Python aflat în `src/preprocessing/preprocess_split.py`.
- **Split:** 70% Train (10.220 imgs), 15% Validation (2.190 imgs), 15% Test (2.190 imgs).
- **Preprocesare:** Imaginile au fost redimensionate la 64x64 pixeli și convertite în Grayscale pentru a asigura robustețea la variații de culoare și eficiență computațională.

---

## Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|--------------------|-----------------|
| Learning rate | 0.001 | Valoare optimă pentru Adam, permite scăderea rapidă a erorii fără a oscila periculos |
| Batch size | 32 | Echilibru între stabilitatea gradientului și utilizarea memoriei RAM/GPU |
| Number of epochs | 10-50 | Suficient pentru convergență, utilizând Early Stopping pentru a preveni overfitting-ul |
| Optimizer | Adam | Algoritm adaptiv care gestionează automat rata de învățare pentru fiecare parametru |
| Loss function | CrossEntropyLoss | Funcția standard în PyTorch pentru clasificarea categorială, care penalizează eficient predicțiile incorecte |
| Activation functions| ReLU & Softmax | ReLU pentru straturile ascunse (evită vanishing gradient), iar Softmax pentru probabilități la ieșire (interpretarea rezultatelor sub formă de probabilități) |

**Justificare detaliată batch size:**
Am ales batch_size=32 pentru că avem N=14.600 samples → 14,600/32 ≈ 456 iterații/epocă. Acestea oferă echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Utilizare resurse (batch optim pentru a evita limitările de memorie pe CPU/RAM).
- Timp antrenare (batch 32 asigură convergență în 10-50 epoci pentru arhitectura noastră).
---

## Integrare UI și Demonstrație

Aplicația `src/app/webcam_detect.py` încarcă acum modelul antrenat `models/trained_model.pth`. 
- **Inferență:** Timp de răspuns ~18ms.
- **Feedback:** UI-ul se colorează în roșu și emite un semnal sonor când scorul de oboseală depășește pragul stabilit.

**Screenshot Inferență Reală:** Se găsește în `docs/screenshots/inference_real.png`
---

### Nivel 2 – Recomandat (85-90% din punctaj)

## Metrici Performanță (Test Set)

După antrenarea modelului, am obținut următoarele rezultate pe setul de date de test (date pe care modelul nu le-a văzut niciodată):
- **Acuratețe (Accuracy):** 96.4%
- **F1-score (macro):** 0.96
- **Precision (macro):** 0.96
- **Recall (macro):** 0.96
---

## Analiză Erori în Context Industrial

### 1. Pe ce clase greșește cel mai mult modelul?
Modelul tinde să confunde ocazional ochii închiși cu ochii deschiși în cazul persoanelor care poartă ochelari cu ramă groasă sau reflexii puternice. De asemenea, clipitul foarte rapid (cadre intermediare) este uneori clasificat eronat.

### 2. Ce caracteristici ale datelor cauzează erori?
Lumina foarte slabă (zgomot de imagine mare) și unghiurile extreme ale capului (profil) reduc vizibilitatea regiunii perioculare, făcând distincția între pleoapa închisă și deschisă dificilă pentru CNN-ul actual.

### 3. Ce implicații are pentru aplicația industrială?
- **False Negatives (critic):** un șofer adoarme, dar sistemul crede că are ochii deschiși, riscul fiind accidentul rutier.
- **False Positives:** alarme false în timpul condusului normal, care pot duce la frustrarea șoferului și dezactivarea sistemului. 
Prioritate: Maximizarea Recall-ului pentru clasa "Closed".

### 4. Ce măsuri corective propuneți?
1. Implementarea unui filtru temporal (ex: alerta se dă doar dacă minim 5 cadre consecutive sunt "Closed").
2. Adăugarea de imagini cu ochelari și reflexii în setul de antrenare.
3. Utilizarea iluminării Infraroșu (IR) pentru a elimina variațiile de lumină ambientală.
---

### Nivel 3 – Bonus (până la 100%)

| **Arhitectură** | **Straturi Conv** | **Accuracy** | **Latență** | **Status** |
|-----------------|-------------------|--------------|-------------|------------|
| Tiny-CNN | 2 | 94.20% | 9ms | Respins |
| CNN Personal | 3 | 99.93% | 14ms | Ales | 

**Justificare**: Am ales CNN-ul meu, deoarece saltul de aproximativ 5% în acuratețe este critic pentru un sistem de siguranță, în timp ce latența de 14ms rămâne mult sub pragul de alertă în timp real.

**Analiză erori (Top 5 exemple):**
1. Ochi încruntați - clasificați eronat ca fiind "Closed" din cauza luminii frontale puternice.
2. Ramă ochelari - umbrele lăsate de ramele groase pot induce incertitudine modelului.
3. Rotație cap - unghiurile de tip profil duc la pierderea vizibilității unuia dintre ochi.
4. Zgomot senzor - zgomotul digital afectează claritatea pleoapelor în cindoții de lumină slabă.
5. Cadre de tranziție - imagini surprinse exact în momentul deschiderii/închiderii ochiului (semi-închiși).
---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența modelului respectă riguros fluxul logic definit în State Machine-ul proiectului:

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Citire batch date din `data/train/` pentru antrenare |
| `PREPROCESS` | Resize la 64x64, Grayscale și Normalizare [-1, 1] (conform transformărilor de la antrenare) |
| `RN_INFERENCE` | Forward pass folosind greutățile salvate în trained_model.pth |
| `THRESHOLD_CHECK` | Verificarea probabilității returnate de model față de pragul de decizie (ex: >0.5 pentru ochi închiși) |
| `ALERT` | Incrementarea contorului de cadre consecutive și activarea alarmei sonore (winsound) |

---

**Completați pentru proiectul vostru:**
```
La nivelul modelului baseline, deși acuratețea generală este ridicată, am identificat următoarele confuzii principale între clasele Open și Closed:
- Ochi deschiși (Open) clasificați ca Închiși (Closed) - 2 cauze principale, anume ochii încruntați (din cauza luminii ambientale puternice, ochii vor fi clasificați eronat ca fiind închiși) și umbrele (arcadele sprâncenelor sau purtarea ochelarilor pot crea umbre în zona orbitală, modelul interpretând regiunea întunecată drept o pleoapă închisă).
- Ochi închiși (Closed) clasificați ca Deschiși (Open) - 2 cauze principale, anume tranzițiile (în timpul clipitului rapid, modelul captează cadre intermediare pe care nu le poate încadra cu certitudine în clasa corectă) și zgomotul digital (iluminare slabă, zgomotul din imagine distorsionează textura pleoapei, făcând-o să pară similară cu irisul).
```
---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [x] State Machine există și e documentat în `docs/state_machine.*`
- [x] Contribuție ≥40% date originale verificabilă în `data/generated/`
- [x] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [x] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [x] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [x] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [x] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [x] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [x] Tabel hiperparametri + justificări completat în acest README
- [x] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [x] Model salvat în `models/trained_model.pht`
- [x] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [x] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [x] UI face inferență REALĂ cu predicții corecte
- [x] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [x] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [x] Early stopping implementat și documentat în cod
- [x] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [x] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [x] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [x] Analiză erori în context industrial completată (4 întrebări răspunse)
- [x] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [x] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [-] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [x] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [x] `requirements.txt` actualizat cu toate bibliotecile noi
- [x] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [x] Cod nou comentat în limba română sau engleză (minimum 15%)
- [-] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [x] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [x] Fluxul de inferență respectă stările din State Machine
- [x] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [x] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [x] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [x] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [x] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [x] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [x] Push: `git push origin main --tags`
- [x] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. [x] **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. [x] **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. [x] **`results/training_history.csv`** - toate epoch-urile salvate

4. [x] **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. [x] **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. [x] **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. [x] **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---