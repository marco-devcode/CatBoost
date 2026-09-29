# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-09-29 02:05
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 17 | 0.1340 | 50.41% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5041 | Target > 0.52 |
| **RPS** | 0.1340 | Più basso = meglio |
| **Recall Home Win** | 74.03% | |
| **Recall Draw** | 9.28% | Classe più difficile |
| **Recall Away Win** | 53.45% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1340 | 50.41% |
| **Bet365** | 0.1922 | 34.33% |
| **Delta** | +0.0582 ✅ | +0.1608 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 367 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €4402630418781564829696.00 |
| Profitto | €+4402630418781564829696.00 |
| ROI | +440263041878156509184.0% |
| Stake totale | 1.6883630947968362e+20% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 17 feature |
|------|-------------|------|
| home_promoted | 0.021146 | 16/17 |
| away_promoted | 0.036448 | 17/17 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5044    0.7403    0.6000       154
           D     0.5000    0.0928    0.1565        97
           A     0.5041    0.5345    0.5188       116

    accuracy                         0.5041       367
   macro avg     0.5028    0.4558    0.4251       367
weighted avg     0.5031    0.5041    0.4571       367

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
