# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-08-27 20:14
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 18 | 0.1324 | 52.50% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5250 | Target > 0.52 |
| **RPS** | 0.1324 | Più basso = meglio |
| **Recall Home Win** | 75.44% | |
| **Recall Draw** | 14.11% | Classe più difficile |
| **Recall Away Win** | 56.29% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1324 | 52.50% |
| **Bet365** | 0.1722 | 36.73% |
| **Delta** | +0.0397 ✅ | +0.1577 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 1522 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €28714624699356773015292152203708419049229576199733248.00 |
| Profitto | €+28714624699356773015292152203708419049229576199733248.00 |
| ROI | +2871462469935677235067815431125048259732604605956096.0% |
| Stake totale | 2.321039593878632e+51% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5411    0.7544    0.6302       619
           D     0.3986    0.1411    0.2085       418
           A     0.5342    0.5629    0.5482       485

    accuracy                         0.5250      1522
   macro avg     0.4913    0.4862    0.4623      1522
weighted avg     0.4998    0.5250    0.4883      1522

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
