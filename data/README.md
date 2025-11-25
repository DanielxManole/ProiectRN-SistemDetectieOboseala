# Setul de Date - Monitorizarea Oboselii Șoferului

## Descriere
Acest director conține datele utilizate pentru antrenarea și validarea rețelei neuronale CNN. Setul de date este structurat pentru o problemă de clasificare binară:
- **Clasa 0:** normal (Ochii deschiși, care induc o stare normală)
- **Clasa 1:** obosit (Ochii închiși, care induc o stare de oboseală)

## Sursa Datelor
Datele utilizate provin din **MRL Eye Dataset**, un benchmark public disponibil pe platforma Kaggle.
Acest set de date conține imagini în spectru infraroșu cu ochii șoferilor în diverse condiții de iluminare și cu diverse accesorii (ochelari), fiind ideal pentru antrenarea rețelelor neuronale robuste.

Link sursă: [https://www.kaggle.com/datasets/taha07/mrl-eye-dataset](https://www.kaggle.com/datasets/taha07/mrl-eye-dataset)

## Structura Datelor
Datele au fost împărțite în trei subseturi pentru a asigura o evaluare corectă a modelului:

1. **Train (70% - 80%):** Utilizat pentru ajustarea ponderilor rețelei.
2. **Validation (10% - 15%):** Utilizat pentru tunarea hiperparametrilor și prevenirea overfitting-ului în timpul antrenării.
3. **Test (10% - 15%):** Utilizat pentru evaluarea finală a performanței modelului pe date nevăzute.

## Preprocesare
Imaginile au trecut prin următorii pași:
- Redimensionare la **64x64 pixeli** (sau 224x224, în funcție de model).
- Conversie în **Grayscale** (dacă se utilizează un singur canal) sau **RGB**.
- Normalizare (valorile pixelilor scalate între 0 și 1) - realizată la încărcarea în model.
