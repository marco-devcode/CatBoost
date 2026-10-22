# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-22 00:50
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 41 | 0.1315 | 55.80% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5580 | Target > 0.52 |
| **RPS** | 0.1315 | Più basso = meglio |
| **Recall Home Win** | 70.72% | |
| **Recall Draw** | 21.82% | Classe più difficile |
| **Recall Away Win** | 62.42% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1315 | 55.80% |
| **Bet365** | 0.1311 | 54.69% |
| **Delta** | -0.0004 ⚠️ | +0.0112 ✅ |

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
| Bankroll finale | €851.83 |
| Profitto | €-148.17 |
| ROI | -14.8% |
| Stake totale | 1461.8% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 41 feature |
|------|-------------|------|
| home_promoted | 0.010061 | 16/41 |
| away_promoted | 0.012544 | 17/41 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5689    0.7072    0.6305       181
           D     0.3871    0.2182    0.2791       110
           A     0.6087    0.6242    0.6164       157

    accuracy                         0.5580       448
   macro avg     0.5216    0.5165    0.5087       448
weighted avg     0.5382    0.5580    0.5393       448

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
