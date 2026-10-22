# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-22 06:05
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 41 | 0.1308 | 55.13% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5513 | Target > 0.52 |
| **RPS** | 0.1308 | Più basso = meglio |
| **Recall Home Win** | 71.82% | |
| **Recall Draw** | 20.00% | Classe più difficile |
| **Recall Away Win** | 60.51% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1308 | 55.13% |
| **Bet365** | 0.1311 | 54.69% |
| **Delta** | +0.0003 ✅ | +0.0045 ✅ |

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
| Bankroll finale | €1426.73 |
| Profitto | €+426.73 |
| ROI | +42.7% |
| Stake totale | 1687.3% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 41 feature |
|------|-------------|------|
| home_promoted | 0.008300 | 16/41 |
| away_promoted | 0.011387 | 17/41 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5628    0.7182    0.6311       181
           D     0.3548    0.2000    0.2558       110
           A     0.6129    0.6051    0.6090       157

    accuracy                         0.5513       448
   macro avg     0.5102    0.5078    0.4986       448
weighted avg     0.5293    0.5513    0.5312       448

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
