# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-09 15:04
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 32 | 0.1293 | 56.56% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5656 | Target > 0.52 |
| **RPS** | 0.1293 | Più basso = meglio |
| **Recall Home Win** | 75.88% | |
| **Recall Draw** | 15.38% | Classe più difficile |
| **Recall Away Win** | 62.21% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1293 | 56.56% |
| **Bet365** | 0.1296 | 55.94% |
| **Delta** | +0.0004 ✅ | +0.0061 ✅ |

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
| Bankroll finale | €782.95 |
| Profitto | €-217.05 |
| ROI | -21.7% |
| Stake totale | 729.8% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.002978 | 16/32 |
| away_promoted | 0.013291 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5853    0.7588    0.6608       199
           D     0.3600    0.1538    0.2156       117
           A     0.5944    0.6221    0.6080       172

    accuracy                         0.5656       488
   macro avg     0.5132    0.5116    0.4948       488
weighted avg     0.5345    0.5656    0.5354       488

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
