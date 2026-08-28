# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-08-28 13:45
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 18 | 0.1331 | 51.64% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5164 | Target > 0.52 |
| **RPS** | 0.1331 | Più basso = meglio |
| **Recall Home Win** | 73.18% | |
| **Recall Draw** | 14.11% | Classe più difficile |
| **Recall Away Win** | 56.49% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1331 | 51.64% |
| **Bet365** | 0.1722 | 36.73% |
| **Delta** | +0.0390 ✅ | +0.1491 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 1522 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €154797939204623887378943030377213739296727498752000.00 |
| Profitto | €+154797939204623887378943030377213739296727498752000.00 |
| ROI | +15479793920462388737894303037721373929672749875200.0% |
| Stake totale | 1.2311177593279722e+49% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 18 feature |
|------|-------------|------|
| home_promoted | 0.018447 | 17/18 |
| away_promoted | 0.028866 | 18/18 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5438    0.7318    0.6240       619
           D     0.3471    0.1411    0.2007       418
           A     0.5279    0.5649    0.5458       485

    accuracy                         0.5164      1522
   macro avg     0.4729    0.4793    0.4568      1522
weighted avg     0.4847    0.5164    0.4828      1522

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
