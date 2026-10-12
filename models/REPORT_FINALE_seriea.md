# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-12 14:23
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 17 | 0.1361 | 52.10% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5210 | Target > 0.52 |
| **RPS** | 0.1361 | Più basso = meglio |
| **Recall Home Win** | 75.88% | |
| **Recall Draw** | 7.48% | Classe più difficile |
| **Recall Away Win** | 56.95% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1361 | 52.10% |
| **Bet365** | 0.1784 | 37.38% |
| **Delta** | +0.0423 ✅ | +0.1472 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 428 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €9313471.19 |
| Profitto | €+9312471.19 |
| ROI | +931247.1% |
| Stake totale | 1361583.3% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 17 feature |
|------|-------------|------|
| home_promoted | 0.016004 | 16/17 |
| away_promoted | 0.030456 | 17/17 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5223    0.7588    0.6187       170
           D     0.2353    0.0748    0.1135       107
           A     0.5850    0.5695    0.5772       151

    accuracy                         0.5210       428
   macro avg     0.4475    0.4677    0.4365       428
weighted avg     0.4727    0.5210    0.4777       428

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
