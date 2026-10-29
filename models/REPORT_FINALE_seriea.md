# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-29 12:20
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 32 | 0.1304 | 56.01% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5601 | Target > 0.52 |
| **RPS** | 0.1304 | Più basso = meglio |
| **Recall Home Win** | 75.53% | |
| **Recall Draw** | 15.93% | Classe più difficile |
| **Recall Away Win** | 61.21% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1304 | 56.01% |
| **Bet365** | 0.1301 | 55.36% |
| **Delta** | -0.0003 ⚠️ | +0.0064 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 466 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €548.88 |
| Profitto | €-451.12 |
| ROI | -45.1% |
| Stake totale | 655.2% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.002327 | 16/32 |
| away_promoted | 0.013285 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5820    0.7553    0.6574       188
           D     0.3396    0.1593    0.2169       113
           A     0.5976    0.6121    0.6048       165

    accuracy                         0.5601       466
   macro avg     0.5064    0.5089    0.4930       466
weighted avg     0.5287    0.5601    0.5320       466

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
