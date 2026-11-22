# 🏆 Report Finale — CatBoost Eredivisie Match Prediction

**Data**: 2026-11-21 15:45
**Modello migliore**: Baseline (Pi-Rating + FE) — Eredivisie (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Eredivisie | 29 | 0.1308 | 52.67% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5267 | Target > 0.52 |
| **RPS** | 0.1308 | Più basso = meglio |
| **Recall Home Win** | 77.91% | |
| **Recall Draw** | 0.00% | Classe più difficile |
| **Recall Away Win** | 63.85% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1308 | 52.67% |
| **Bet365** | 0.1310 | 52.43% |
| **Delta** | +0.0002 ✅ | +0.0024 ✅ |

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
| Bankroll finale | €1042.78 |
| Profitto | €+42.78 |
| ROI | +4.3% |
| Stake totale | 63.4% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 29 feature |
|------|-------------|------|
| home_promoted | 0.002755 | 16/29 |
| away_promoted | 0.002296 | 17/29 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5234    0.7791    0.6262       172
           D     0.0000    0.0000    0.0000       110
           A     0.5321    0.6385    0.5804       130

    accuracy                         0.5267       412
   macro avg     0.3518    0.4725    0.4022       412
weighted avg     0.3864    0.5267    0.4446       412

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
