# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-26 16:23
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 29 | 0.1287 | 56.83% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5683 | Target > 0.52 |
| **RPS** | 0.1287 | Più basso = meglio |
| **Recall Home Win** | 81.46% | |
| **Recall Draw** | 2.52% | Classe più difficile |
| **Recall Away Win** | 64.94% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1287 | 56.83% |
| **Bet365** | 0.1288 | 56.22% |
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
| Scommesse piazzate | 78 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1016.32 |
| Profitto | €+16.32 |
| ROI | +1.6% |
| Stake totale | 32.2% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.000000 | 16/29 |
| away_promoted | 0.000221 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5548    0.8146    0.6601       205
           D     0.6000    0.0252    0.0484       119
           A     0.5885    0.6494    0.6175       174

    accuracy                         0.5683       498
   macro avg     0.5811    0.4964    0.4420       498
weighted avg     0.5774    0.5683    0.4990       498

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
