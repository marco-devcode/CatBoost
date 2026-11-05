# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-11-05 14:06
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 32 | 0.1294 | 52.11% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5211 | Target > 0.52 |
| **RPS** | 0.1294 | Più basso = meglio |
| **Recall Home Win** | 77.51% | |
| **Recall Draw** | 0.00% | Classe più difficile |
| **Recall Away Win** | 62.20% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1294 | 52.11% |
| **Bet365** | 0.1309 | 52.61% |
| **Delta** | +0.0015 ✅ | -0.0050 ⚠️ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 403 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1678.20 |
| Profitto | €+678.20 |
| ROI | +67.8% |
| Stake totale | 382.9% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.002896 | 16/32 |
| away_promoted | 0.019634 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5261    0.7751    0.6268       169
           D     0.0000    0.0000    0.0000       107
           A     0.5130    0.6220    0.5623       127

    accuracy                         0.5211       403
   macro avg     0.3464    0.4657    0.3964       403
weighted avg     0.3823    0.5211    0.4400       403

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
