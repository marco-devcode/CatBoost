# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-10-15 10:56
**Modello migliore**: Baseline (Pi-Rating + FE) — Serie A (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) — Serie A | 17 | 0.1430 | 48.63% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.4863 | Target > 0.52 |
| **RPS** | 0.1430 | Più basso = meglio |
| **Recall Home Win** | 67.05% | |
| **Recall Draw** | 16.36% | Classe più difficile |
| **Recall Away Win** | 50.66% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1430 | 48.63% |
| **Bet365** | 0.1789 | 37.21% |
| **Delta** | +0.0359 ✅ | +0.1142 ✅ |

---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da 50% a
> 10%), con hard cap del
> 5% / 2%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | 438 |
| Bankroll iniziale | €1000 |
| Bankroll finale | €4022547.89 |
| Profitto | €+4021547.89 |
| ROI | +402154.8% |
| Stake totale | 597210.5% del bankroll |

![Kelly Simulation](kelly_simulation.png)

---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su 17 feature |
|------|-------------|------|
| home_promoted | 0.025793 | 16/17 |
| away_promoted | 0.041025 | 17/17 |

![SHAP Summary](shap_summary.png)

---

## 5. Classification Report

```
              precision    recall  f1-score   support

           H     0.5130    0.6705    0.5813       176
           D     0.2903    0.1636    0.2093       110
           A     0.5274    0.5066    0.5168       152

    accuracy                         0.4863       438
   macro avg     0.4436    0.4469    0.4358       438
weighted avg     0.4621    0.4863    0.4655       438

```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
