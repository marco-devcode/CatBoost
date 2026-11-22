# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-21 15:42
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 29 | 0.1295 | 56.35% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5635 | Target > 0.52 |
| **RPS** | 0.1295 | Più basso = meglio |
| **Recall Home Win** | 81.41% | |
| **Recall Draw** | 2.56% | Classe più difficile |
| **Recall Away Win** | 63.95% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1295 | 56.35% |
| **Bet365** | 0.1296 | 55.94% |
| **Delta** | +0.0001 ✅ | +0.0041 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 72 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1031.53 |
| Profitto | €+31.53 |
| ROI | +3.1% |
| Stake totale | 30.8% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.000000 | 16/29 |
| away_promoted | 0.000468 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5492    0.8141    0.6559       199
           D     0.5000    0.0256    0.0488       117
           A     0.5882    0.6395    0.6128       172

    accuracy                         0.5635       488
   macro avg     0.5458    0.4931    0.4392       488
weighted avg     0.5511    0.5635    0.4951       488

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
