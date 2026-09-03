# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-09-03 10:45
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 18 | 0.1377 | 50.46% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5046 | Target > 0.52 |
| **RPS** | 0.1377 | Più basso = meglio |
| **Recall Home Win** | 70.51% | |
| **Recall Draw** | 16.75% | Classe più difficile |
| **Recall Away Win** | 53.67% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1377 | 50.46% |
| **Bet365** | 0.1720 | 36.88% |
| **Delta** | +0.0343 ✅ | +0.1358 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 1532 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €467387350810715210781955146507744054764187737391104.00 |
| Profitto | €+467387350810715210781955146507744054764187737391104.00 |
| ROI | +46738735081071524193573629771670982594716571271168.0% |
| Stake totale | 3.2815646263355213e+49% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 18 feature |
|------|-------------|------|
| home_promoted | 0.030255 | 17/18 |
| away_promoted | 0.038425 | 18/18 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5500    0.7051    0.6180       624
           D     0.3196    0.1675    0.2198       418
           A     0.5127    0.5367    0.5244       490

    accuracy                         0.5046      1532
   macro avg     0.4608    0.4698    0.4541      1532
weighted avg     0.4752    0.5046    0.4794      1532

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
