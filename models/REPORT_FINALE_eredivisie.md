# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-10-22 06:20
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 41 | 0.1334 | 53.77% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5377 | Target > 0.52 |
| **RPS** | 0.1334 | Più basso = meglio |
| **Recall Home Win** | 73.91% | |
| **Recall Draw** | 11.65% | Classe più difficile |
| **Recall Away Win** | 62.81% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1334 | 53.77% |
| **Bet365** | 0.1306 | 52.73% |
| **Delta** | -0.0027 ⚠️ | +0.0104 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 385 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €919.14 |
| Profitto | €-80.86 |
| ROI | -8.1% |
| Stake totale | 864.0% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 41 feature |
|------|-------------|------|
| home_promoted | 0.013741 | 16/41 |
| away_promoted | 0.030224 | 17/41 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5385    0.7391    0.6230       161
           D     0.5000    0.1165    0.1890       103
           A     0.5429    0.6281    0.5824       121

    accuracy                         0.5377       385
   macro avg     0.5271    0.4946    0.4648       385
weighted avg     0.5296    0.5377    0.4941       385

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
