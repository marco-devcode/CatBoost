# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-09-29 02:53
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 17 | 0.1337 | 52.04% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5204 | Target > 0.52 |
| **RPS** | 0.1337 | Più basso = meglio |
| **Recall Home Win** | 75.97% | |
| **Recall Draw** | 10.31% | Classe più difficile |
| **Recall Away Win** | 55.17% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1337 | 52.04% |
| **Bet365** | 0.1922 | 34.33% |
| **Delta** | +0.0586 ✅ | +0.1771 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 367 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €5003525964071091306496.00 |
| Profitto | €+5003525964071091306496.00 |
| ROI | +500352596407109091328.0% |
| Stake totale | 2.2116813981609066e+20% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 17 feature |
|------|-------------|------|
| home_promoted | 0.019296 | 16/17 |
| away_promoted | 0.039289 | 17/17 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5247    0.7597    0.6207       154
           D     0.5263    0.1031    0.1724        97
           A     0.5120    0.5517    0.5311       116

    accuracy                         0.5204       367
   macro avg     0.5210    0.4715    0.4414       367
weighted avg     0.5211    0.5204    0.4739       367

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
