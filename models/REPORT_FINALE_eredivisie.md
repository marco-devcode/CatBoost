# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-10-29 12:25
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 32 | 0.1293 | 52.03% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5203 | Target > 0.52 |
| **RPS** | 0.1293 | Più basso = meglio |
| **Recall Home Win** | 79.14% | |
| **Recall Draw** | 0.00% | Classe più difficile |
| **Recall Away Win** | 60.80% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1293 | 52.03% |
| **Bet365** | 0.1305 | 52.54% |
| **Delta** | +0.0012 ✅ | -0.0051 ⚠️ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 394 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1504.20 |
| Profitto | €+504.20 |
| ROI | +50.4% |
| Stake totale | 336.5% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.003005 | 16/32 |
| away_promoted | 0.021134 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5244    0.7914    0.6308       163
           D     0.0000    0.0000    0.0000       106
           A     0.5135    0.6080    0.5568       125

    accuracy                         0.5203       394
   macro avg     0.3460    0.4665    0.3959       394
weighted avg     0.3799    0.5203    0.4376       394

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
