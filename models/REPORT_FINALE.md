# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-09-26 07:02
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 18 | 0.1347 | 53.27% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5327 | Target > 0.52 |
| **RPS** | 0.1347 | Più basso = meglio |
| **Recall Home Win** | 74.71% | |
| **Recall Draw** | 13.08% | Classe più difficile |
| **Recall Away Win** | 57.62% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1347 | 53.27% |
| **Bet365** | 0.1784 | 37.38% |
| **Delta** | +0.0437 ✅ | +0.1589 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 428 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €260593941538056160.00 |
| Profitto | €+260593941538055168.00 |
| ROI | +26059394153805516.0% |
| Stake totale | 1.6992625964947146e+16% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 18 feature |
|------|-------------|------|
| home_promoted | 0.014192 | 17/18 |
| away_promoted | 0.027331 | 18/18 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5292    0.7471    0.6195       170
           D     0.3590    0.1308    0.1918       107
           A     0.5839    0.5762    0.5800       151

    accuracy                         0.5327       428
   macro avg     0.4907    0.4847    0.4638       428
weighted avg     0.5059    0.5327    0.4986       428

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
