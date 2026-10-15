# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-10-15 11:01
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 17 | 0.1320 | 50.80% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5080 | Target > 0.52 |
| **RPS** | 0.1320 | Più basso = meglio |
| **Recall Home Win** | 77.07% | |
| **Recall Draw** | 5.94% | Classe più difficile |
| **Recall Away Win** | 54.24% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1320 | 50.80% |
| **Bet365** | 0.1907 | 34.31% |
| **Delta** | +0.0587 ✅ | +0.1649 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 376 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €689035942.44 |
| Profitto | €+689034942.44 |
| ROI | +68903494.2% |
| Stake totale | 73010186.2% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 17 feature |
|------|-------------|------|
| home_promoted | 0.012671 | 16/17 |
| away_promoted | 0.031621 | 17/17 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.4979    0.7707    0.6050       157
           D     0.5455    0.0594    0.1071       101
           A     0.5246    0.5424    0.5333       118

    accuracy                         0.5080       376
   macro avg     0.5227    0.4575    0.4152       376
weighted avg     0.5191    0.5080    0.4488       376

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
