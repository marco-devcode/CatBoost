# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-11-05 14:01
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 32 | 0.1304 | 56.07% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5607 | Target > 0.52 |
| **RPS** | 0.1304 | Più basso = meglio |
| **Recall Home Win** | 75.90% | |
| **Recall Draw** | 14.04% | Classe più difficile |
| **Recall Away Win** | 61.54% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1304 | 56.07% |
| **Bet365** | 0.1301 | 55.86% |
| **Delta** | -0.0002 ⚠️ | +0.0021 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 478 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €625.56 |
| Profitto | €-374.44 |
| ROI | -37.4% |
| Stake totale | 646.5% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 32 feature |
|------|-------------|------|
| home_promoted | 0.002290 | 16/32 |
| away_promoted | 0.014415 | 17/32 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5804    0.7590    0.6578       195
           D     0.3333    0.1404    0.1975       114
           A     0.5943    0.6154    0.6047       169

    accuracy                         0.5607       478
   macro avg     0.5027    0.5049    0.4867       478
weighted avg     0.5264    0.5607    0.5292       478

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
