# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-09-26 07:20
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 18 | 0.1379 | 52.10% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5210 | Target > 0.52 |
| **RPS** | 0.1379 | Più basso = meglio |
| **Recall Home Win** | 70.59% | |
| **Recall Draw** | 19.63% | Classe più difficile |
| **Recall Away Win** | 54.30% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1379 | 52.10% |
| **Bet365** | 0.1784 | 37.38% |
| **Delta** | +0.0405 ✅ | +0.1472 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly ridotto al 25%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 428 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €348407528386190784.00 |
| Profitto | €+348407528386189760.00 |
| ROI | +34840752838618976.0% |
| Stake totale | 2.831677439916318e+16% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 18 feature |
|------|-------------|------|
| home_promoted | 0.025954 | 17/18 |
| away_promoted | 0.028659 | 18/18 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5333    0.7059    0.6076       170
           D     0.3889    0.1963    0.2609       107
           A     0.5503    0.5430    0.5467       151

    accuracy                         0.5210       428
   macro avg     0.4909    0.4817    0.4717       428
weighted avg     0.5032    0.5210    0.4994       428

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
