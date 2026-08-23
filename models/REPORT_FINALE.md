# 🏆 Report Finale — CatBoost Serie A Match Prediction

**Data**: 2026-08-23 11:13
**Modello migliore**: Baseline (Pi-Rating + FE) (`v1_baseline`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
| Baseline (Pi-Rating + FE) | 16 | 0.1312 | 53.11% ◄ **BEST**|

---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | 0.5311 | Target > 0.52 |
| **RPS** | 0.1312 | Più basso = meglio |
| **Recall Home Win** | 77.76% | |
| **Recall Draw** | 13.19% | Classe più difficile |
| **Recall Away Win** | 56.16% | |

---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost (v1_baseline)** | 0.1312 | 53.11% |
| **Bet365** | 0.1721 | 36.64% |
| **Delta** | +0.0409 ✅ | +0.1647 ✅ |

---

## 4. Classification Report

```
              precision    recall  f1-score   support

           H     0.5481    0.7776    0.6430       616
           D     0.3929    0.1319    0.1975       417
           A     0.5402    0.5616    0.5507       479

    accuracy                         0.5311      1512
   macro avg     0.4937    0.4904    0.4637      1512
weighted avg     0.5028    0.5311    0.4909      1512

```

---

## 5. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
