# README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Manole Daniel
**Link Repository GitHub:** [https://github.com/DanielxManole/ProiectRN-SistemDetectieOboseala](https://github.com/DanielxManole/ProiectRN-SistemDetectieOboseala)

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** MRL Eye Dataset, un set de date benchmark utilizat pe scară largă pentru sistemele de monitorizare a stării de alertă a șoferilor.
* **Modul de achiziție:** [x] Fișier extern / [x] Generare programatică (extragerea regiunilor de interes - ROI - din stream video).
* **Sursă:** Kaggle - MRL Eye Dataset
* **Perioada / condițiile colectării:** datele acoperă scenarii diverse: subiecți cu/fără ochelari, condiții variate de iluminare și diferite unghiuri ale feței.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** ~14.600 imagini (echilibrat între clasele Open/Closed).
* **Număr de caracteristici (features):** 4096 (pentru inputul CNN: 64x64 pixeli) + 1 metrică geometrică (EAR).
* **Tipuri de date:** [x] Numerice (pixeli) / [x] Categoriale (etichete) / [x] Imagini.
* **Format fișiere:** [x] PNG / [x] JPG (re-formate la 64x64 pixeli).
* **Clase:** Clasa 0 (Open) - ochi deschiși, stare de alertă | Clasa 1 (Closed) - ochi închiși, stare de oboseală.

### 2.3 Descrierea fiecărei caracteristici

În contextul acestui proiect, caracteristicile sunt extrase din imaginea brută pentru a hrăni atât modelul neuronal, cât și logica de decizie hibridă:

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| Imagine ROI | numeric | pixel (int) | Matricea de pixeli a ochiului scalată la 64x64 | 0–255 (Grayscale) |
| EAR | numeric | ratio | Raportul dintre înălțimea și lățimea ochiului (Eye Aspect Ratio) | 0.0 – 0.50 |
| Label | categorial | - | Starea ochiului (clasa țintă) | Open, Closed |
| Landmarks | numeric | coordonate | Cele 6 puncte faciale (x, y) specifice fiecărui ochi | Proporțional cu rezoluția |

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie și Deviație Standard:** media pixelilor este egală cu aproximativ 115, indicând un set de date cu tonalități medii-închise. Deviația standard ridicată (sigma = 62) confirmă un contrast bun între pupilă și restul feței.
* **Min–Max:** valorile acoperă întreg spectrul [0, 255], asigurând o gamă dinamică completă.
* **Distribuții claselor:** dataset-ul este echilibrat - clasa 0 (open) = 50%, clasa 1 (closed) = 50%.
* **Identificarea outlierilor:** au fost identificate și eliminate cadrele cu zgomot extrem sau cele în care ochiul nu era vizibil deloc (decupaje eșuate în setul brut).

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă:** 0%, toate fișierele din directoarele sursă sunt valide și conțin date de tip imagine.
* **Detectarea valorilor inconsistente sau eronate:** s-au detectat reflexii IR puternice pe lentilele ochelarilor în aproximativ 12% din imagini, care pot masca detaliile pleoapelor.
* **Identificarea caracteristicilor redundante sau puternic corelate:** imaginile provenind din fluxuri video, există cadre succesive foarte similare. Astfel, s-a utilizat un split stratificat pentru a evita ca imagini aproape identice să apară simultan în Train și Test.

### 3.3 Probleme identificate

* Deși este spectru IR, intensitatea sursei externe variază, necesitând normalizare globală pentru uniformizare.
* Prezența ramelor groase de ochelari care „taie” din zona vizibilă a ochiului în anumite unghiuri.
* În condiții de lumină extrem de slabă, imaginile prezintă un nivel ridicat de zgomot alb, care poate afecta marginile irisului.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor:** s-au eliminat cadrele identice consecutive rezultate din stream-urile video, pentru a evita supra-antrenarea (overfitting) pe aceleași ipostaze.
* **Tratarea valorilor lipsă:**
  * Filtrare fișiere: eliminarea automată a imaginilor corupte sau care aveau dimensiuni sub pragul minim de 20x20 pixeli.
  * Bad crops: s-au eliminat manual sau algoritmic decupajele în care ochiul nu era centrat (ex: doar sprânceana sau doar obrazul).
* **Tratarea outlierilor:** s-au eliminat imaginile cu expunere extremă (complet albe sau complet negre) unde informația de textură era pierdută.

### 4.2 Transformarea caracteristicilor

* **Normalizare:** aplicarea transformării Normalize (mean=0.5, std=0.5), aducând valorile pixelilor din intervalul [0, 1] în intervalul [-1, 1] pentru a stabiliza procesul de antrenare a CNN-ului.
* **Conversie spațiu culori:** trecerea la Grayscale pentru a reduce numărul de parametri ai rețelei, culoarea nefiind un indicator relevant în spectrul Infraroșu.
* **Redimensionare:** scalarea tuturor imaginilor la rezoluția fixă de 64x64 pixeli prin interpolare biliniară.
* **Label Encoding:** conversia etichetelor text în format numeric, anume 0 pentru Open și 1 pentru Closed.

### 4.3 Structurarea seturilor de date

**Împărțire finală:**
* 70% – Train
* 15% – Validation
* 15% – Test

**Principii respectate:**
* Ne-am asigurat că proporția de ochi închiși/deschiși este identică în toate cele trei seturi.
* Imaginile aceluiași subiect nu sunt împărțite între seturi diferite; statisticile de normalizare au fost calculate doar pe setul de Train.

### 4.4 Salvarea rezultatelor preprocesării

* Imaginile transformate sunt stocate ca tensori PyTorch sau fișiere optimizate în `data/processed/`.
* Seturi train/val/test în foldere dedicate
* Toți parametrii critici (medie, deviație, rezoluție) sunt salvați în `config/preprocessing_params.pkl` pentru a fi încărcați identic în faza de inferență/test.

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – imaginile originale din MRL Eye Dataset (organizate în subfolderele Open/Closed).
* `data/processed/` – imaginile redimensionate la 64x64 și convertite în format grayscale.
* `data/train/`, `data/validation/`, `data/test/` – cele trei subseturi rezultate după split-ul stratificat (70/15/15).
* `config/preprocessing_params.pkl` – parametrii precum mediile, deviațiile și dimensiunile utilizate, necesari pentru inferență.
* `src/preprocessing/` – scripturile Python care automatizează pipeline-ul de preprocesare.
* `data/README.md` – documentația tehnică a setului de date

---

##  6. Stare Etapă (de completat de student)

- [x] Structură repository configurată
- [x] Dataset analizat (EDA realizată)
- [x] Date preprocesate
- [x] Seturi train/val/test generate
- [x] Documentație actualizată în README + `data/README.md`

---
