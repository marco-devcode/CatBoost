# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-09-28 23:53
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 17 | 0.1445 | 48.36% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.4836 | Target > 0.52 |
| **RPS** | 0.1445 | Più basso = meglio |
| **Recall Home Win** | 65.88% | |
| **Recall Draw** | 14.95% | Classe più difficile |
| **Recall Away Win** | 52.32% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1445 | 48.36% |
| **Bet365** | 0.1784 | 37.38% |
| **Delta** | +0.0339 ✅ | +0.1098 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 428 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €31659481778259832.00 |
| Profitto | €+31659481778258832.00 |
| ROI | +3165948177825883.0% |
| Stake totale | 3764354427040675.0% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 17 feature |
|------|-------------|------|
| home_promoted | 0.030393 | 16/17 |
| away_promoted | 0.039172 | 17/17 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5045    0.6588    0.5714       170
           D     0.2712    0.1495    0.1928       107
           A     0.5374    0.5232    0.5302       151

    accuracy                         0.4836       428
   macro avg     0.4377    0.4438    0.4315       428
weighted avg     0.4578    0.4836    0.4622       428

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
