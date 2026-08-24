# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-08-24 06:55
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 16 | 0.1333 | 51.19% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5119 | Target > 0.52 |
| **RPS** | 0.1333 | Più basso = meglio |
| **Recall Home Win** | 73.86% | |
| **Recall Draw** | 13.19% | Classe più difficile |
| **Recall Away Win** | 55.11% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1333 | 51.19% |
| **Bet365** | 0.1721 | 36.64% |
| **Delta** | +0.0388 ✅ | +0.1455 ✅ |

---

## 4. Classification Report

```
              precision    recall  f1-score   support

           H     0.5353    0.7386    0.6207       616
           D     0.3395    0.1319    0.1900       417
           A     0.5280    0.5511    0.5393       479

    accuracy                         0.5119      1512
   macro avg     0.4676    0.4739    0.4500      1512
weighted avg     0.4790    0.5119    0.4761      1512

```

---

## 5. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
