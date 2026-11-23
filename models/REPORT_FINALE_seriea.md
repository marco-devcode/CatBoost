# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-23 17:18
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 29 | 0.1284 | 56.94% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5694 | Target > 0.52 |
| **RPS** | 0.1284 | Più basso = meglio |
| **Recall Home Win** | 81.86% | |
| **Recall Draw** | 2.52% | Classe più difficile |
| **Recall Away Win** | 64.94% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1284 | 56.94% |
| **Bet365** | 0.1285 | 56.34% |
| **Delta** | +0.0001 ✅ | +0.0060 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 68 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1021.56 |
| Profitto | €+21.56 |
| ROI | +2.2% |
| Stake totale | 29.4% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.000000 | 16/29 |
| away_promoted | 0.000465 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5548    0.8186    0.6614       204
           D     0.6000    0.0252    0.0484       119
           A     0.5916    0.6494    0.6192       174

    accuracy                         0.5694       497
   macro avg     0.5821    0.4978    0.4430       497
weighted avg     0.5785    0.5694    0.4998       497

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
