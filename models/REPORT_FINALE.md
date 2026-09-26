# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-09-26 08:55
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 18 | 0.1431 | 49.07% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.4907 | Target > 0.52 |
| **RPS** | 0.1431 | Più basso = meglio |
| **Recall Home Win** | 66.47% | |
| **Recall Draw** | 14.02% | Classe più difficile |
| **Recall Away Win** | 54.30% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1431 | 49.07% |
| **Bet365** | 0.1784 | 37.38% |
| **Delta** | +0.0353 ✅ | +0.1168 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 428 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €56302120813975984.00 |
| Profitto | €+56302120813974984.00 |
| ROI | +5630212081397498.0% |
| Stake totale | 5782118466367586.0% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 18 feature |
|------|-------------|------|
| home_promoted | 0.024907 | 17/18 |
| away_promoted | 0.029988 | 18/18 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5067    0.6647    0.5751       170
           D     0.2542    0.1402    0.1807       107
           A     0.5616    0.5430    0.5522       151

    accuracy                         0.4907       428
   macro avg     0.4409    0.4493    0.4360       428
weighted avg     0.4630    0.4907    0.4684       428

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
