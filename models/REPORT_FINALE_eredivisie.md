# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-11-26 16:26
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 29 | 0.1321 | 52.26% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5226 | Target > 0.52 |
| **RPS** | 0.1321 | Più basso = meglio |
| **Recall Home Win** | 78.74% | |
| **Recall Draw** | 0.00% | Classe più difficile |
| **Recall Away Win** | 60.58% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1321 | 52.26% |
| **Bet365** | 0.1323 | 52.26% |
| **Delta** | +0.0001 ✅ | +0.0000 ⚠️ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 121 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1088.19 |
| Profitto | €+88.19 |
| ROI | +8.8% |
| Stake totale | 62.9% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.002189 | 16/29 |
| away_promoted | 0.002711 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5093    0.7874    0.6185       174
           D     0.0000    0.0000    0.0000       110
           A     0.5461    0.6058    0.5744       137

    accuracy                         0.5226       421
   macro avg     0.3518    0.4644    0.3976       421
weighted avg     0.3882    0.5226    0.4425       421

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
