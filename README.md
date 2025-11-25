# Sistem de detectare a oboselii şoferului pe baza expresiilor faciale

## Descriere
**DeepGuard** este un sistem de vedere artificială (Computer Vision) în timp real, proiectat pentru a îmbunătăți siguranța rutieră prin detectarea semnelor de oboseală a șoferului. Sistemul utilizează **Rețele Neuronale Convoluționale (CNN)** pentru a analiza expresiile faciale (închiderea ochilor, căscat) și declanșează alerte pentru a preveni potențialele accidente.

Acest proiect este dezvoltat în cadrul cursului universitar de **Rețele Neuronale**.

## Funcționalități
* **Monitorizare în Timp Real:** Procesează fluxul video pentru a urmări fața șoferului.
* **Clasificare Deep Learning:** Utilizează un model CNN pentru a distinge între stările "Normal" (Treaz) și "Obosit" (Somnolent).
* **Eye Aspect Ratio (EAR):** Implementează o analiză geometrică pentru detectarea clipitului.
* **Integrare Platformă:** Backend Python pentru inferență AI, integrat cu **LabVIEW** pentru interfața grafică și logica de control.

## Tehnologii Utilizate
* **Limbaj:** Python 3.11
* **Biblioteci:** OpenCV, NumPy, Scikit-learn, Dlib
* **Framework DL:** PyTorch / TensorFlow (Keras)
* **Integrare:** NI LabVIEW (pentru GUI și Controlul Sistemului)

## 📂 Structura Proiectului
```
├── data/               # Seturi de date brute și procesate (Train/Val/Test)
├── src/                # Cod sursă pentru preprocesare și antrenare
├── requirements.txt    # Dependențe Python
└── README.md           # Descrierea proiectului
```
## Licență
- Acest proiect este realizat în scop educațional

## Contact
- daniel.manole@stud.fiir.upb.ro
- manoledaniel2004@gmail.com
