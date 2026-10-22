# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-10-22 01:16
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 41 | 0.1323 | 52.73% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5273 | Target > 0.52 |
| **RPS** | 0.1323 | Più basso = meglio |
| **Recall Home Win** | 72.05% | |
| **Recall Draw** | 12.62% | Classe più difficile |
| **Recall Away Win** | 61.16% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1323 | 52.73% |
| **Bet365** | 0.1306 | 52.73% |
| **Delta** | -0.0017 ⚠️ | +0.0000 ⚠️ |

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
| Bankroll finale | €1002.27 |
| Profitto | €+2.27 |
| ROI | +0.2% |
| Stake totale | 845.9% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 41 feature |
|------|-------------|------|
| home_promoted | 0.016356 | 16/41 |
| away_promoted | 0.028743 | 17/41 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5321    0.7205    0.6121       161
           D     0.5652    0.1262    0.2063       103
           A     0.5139    0.6116    0.5585       121

    accuracy                         0.5273       385
   macro avg     0.5371    0.4861    0.4590       385
weighted avg     0.5352    0.5273    0.4867       385

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
