# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-10-22 11:14
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 32 | 0.1296 | 51.69% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5169 | Target > 0.52 |
| **RPS** | 0.1296 | Più basso = meglio |
| **Recall Home Win** | 77.02% | |
| **Recall Draw** | 0.00% | Classe più difficile |
| **Recall Away Win** | 61.98% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1296 | 51.69% |
| **Bet365** | 0.1306 | 52.73% |
| **Delta** | +0.0011 ✅ | -0.0104 ⚠️ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 385 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €1491.78 |
| Profitto | €+491.78 |
| ROI | +49.2% |
| Stake totale | 362.7% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.003063 | 16/32 |
| away_promoted | 0.022912 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5322    0.7702    0.6294       161
           D     0.0000    0.0000    0.0000       103
           A     0.4934    0.6198    0.5495       121

    accuracy                         0.5169       385
   macro avg     0.3419    0.4633    0.3930       385
weighted avg     0.3776    0.5169    0.4359       385

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
