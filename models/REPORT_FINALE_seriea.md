# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-16 14:42
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 32 | 0.1293 | 56.35% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5635 | Target > 0.52 |
| **RPS** | 0.1293 | Più basso = meglio |
| **Recall Home Win** | 77.89% | |
| **Recall Draw** | 11.97% | Classe più difficile |
| **Recall Away Win** | 61.63% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1293 | 56.35% |
| **Bet365** | 0.1296 | 55.94% |
| **Delta** | +0.0003 ✅ | +0.0041 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 488 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €725.69 |
| Profitto | €-274.31 |
| ROI | -27.4% |
| Stake totale | 704.1% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.003366 | 16/32 |
| away_promoted | 0.018668 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5805    0.7789    0.6652       199
           D     0.3182    0.1197    0.1739       117
           A     0.5989    0.6163    0.6074       172

    accuracy                         0.5635       488
   macro avg     0.4992    0.5049    0.4822       488
weighted avg     0.5241    0.5635    0.5271       488

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
