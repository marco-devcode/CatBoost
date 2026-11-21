# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-11-21 00:44
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 29 | 0.1309 | 51.94% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5194 | Target > 0.52 |
| **RPS** | 0.1309 | Più basso = meglio |
| **Recall Home Win** | 77.91% | |
| **Recall Draw** | 0.00% | Classe più difficile |
| **Recall Away Win** | 61.54% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1309 | 51.94% |
| **Bet365** | 0.1310 | 52.43% |
| **Delta** | +0.0001 ✅ | -0.0049 ⚠️ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 188 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1075.57 |
| Profitto | €+75.57 |
| ROI | +7.6% |
| Stake totale | 66.6% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.003049 | 16/29 |
| away_promoted | 0.001833 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5174    0.7791    0.6218       172
           D     0.0000    0.0000    0.0000       110
           A     0.5229    0.6154    0.5654       130

    accuracy                         0.5194       412
   macro avg     0.3468    0.4648    0.3957       412
weighted avg     0.3810    0.5194    0.4380       412

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
