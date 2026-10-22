# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-22 11:10
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 32 | 0.1311 | 54.69% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5469 | Target > 0.52 |
| **RPS** | 0.1311 | Più basso = meglio |
| **Recall Home Win** | 75.14% | |
| **Recall Draw** | 13.64% | Classe più difficile |
| **Recall Away Win** | 59.87% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1311 | 54.69% |
| **Bet365** | 0.1311 | 54.69% |
| **Delta** | -0.0001 ⚠️ | +0.0000 ⚠️ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 448 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €632.53 |
| Profitto | €-367.47 |
| ROI | -36.8% |
| Stake totale | 702.5% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.002258 | 16/32 |
| away_promoted | 0.010321 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5620    0.7514    0.6430       181
           D     0.3261    0.1364    0.1923       110
           A     0.5875    0.5987    0.5931       157

    accuracy                         0.5469       448
   macro avg     0.4919    0.4955    0.4761       448
weighted avg     0.5130    0.5469    0.5148       448

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
