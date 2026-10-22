# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-22 08:13
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 41 | 0.1311 | 55.58% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5558 | Target > 0.52 |
| **RPS** | 0.1311 | Più basso = meglio |
| **Recall Home Win** | 75.69% | |
| **Recall Draw** | 17.27% | Classe più difficile |
| **Recall Away Win** | 59.24% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1311 | 55.58% |
| **Bet365** | 0.1311 | 54.69% |
| **Delta** | -0.0000 ⚠️ | +0.0089 ✅ |

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
| Bankroll finale | €695.03 |
| Profitto | €-304.97 |
| ROI | -30.5% |
| Stake totale | 792.1% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 41 feature |
|------|-------------|------|
| home_promoted | 0.002270 | 16/41 |
| away_promoted | 0.012533 | 17/41 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5708    0.7569    0.6508       181
           D     0.3878    0.1727    0.2390       110
           A     0.5849    0.5924    0.5886       157

    accuracy                         0.5558       448
   macro avg     0.5145    0.5073    0.4928       448
weighted avg     0.5308    0.5558    0.5279       448

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
