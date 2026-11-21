# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-21 00:42
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 29 | 0.1294 | 56.56% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5656 | Target > 0.52 |
| **RPS** | 0.1294 | Più basso = meglio |
| **Recall Home Win** | 81.41% | |
| **Recall Draw** | 2.56% | Classe più difficile |
| **Recall Away Win** | 64.53% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1294 | 56.56% |
| **Bet365** | 0.1296 | 55.94% |
| **Delta** | +0.0002 ✅ | +0.0061 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 102 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1037.83 |
| Profitto | €+37.83 |
| ROI | +3.8% |
| Stake totale | 23.4% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.000179 | 16/29 |
| away_promoted | 0.000221 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5492    0.8141    0.6559       199
           D     0.7500    0.0256    0.0496       117
           A     0.5873    0.6453    0.6150       172

    accuracy                         0.5656       488
   macro avg     0.6288    0.4950    0.4401       488
weighted avg     0.6108    0.5656    0.4961       488

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
