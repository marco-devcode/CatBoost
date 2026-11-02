# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-02 14:43
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 32 | 0.1293 | 56.30% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5630 | Target > 0.52 |
| **RPS** | 0.1293 | Più basso = meglio |
| **Recall Home Win** | 76.41% | |
| **Recall Draw** | 14.91% | Classe più difficile |
| **Recall Away Win** | 61.08% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1293 | 56.30% |
| **Bet365** | 0.1296 | 55.88% |
| **Delta** | +0.0003 ✅ | +0.0042 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 476 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €740.51 |
| Profitto | €-259.49 |
| ROI | -25.9% |
| Stake totale | 684.7% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.002806 | 16/32 |
| away_promoted | 0.011891 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5820    0.7641    0.6608       195
           D     0.3542    0.1491    0.2099       114
           A     0.5930    0.6108    0.6018       167

    accuracy                         0.5630       476
   macro avg     0.5097    0.5080    0.4908       476
weighted avg     0.5313    0.5630    0.5321       476

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
