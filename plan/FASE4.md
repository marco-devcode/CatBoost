# Fase 4 — Modello CatBoost: Piano Dettagliato

---

## 4.1 Training Baseline

Prima di tutto, assicurati che i dati siano pronti dal Phase 3 (pi-rating calcolati, split temporale fatto).

```python
# src/model.py
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

# Carica i dati preparati nella Fase 3
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',   # 3 classi: H / D / A
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100                   # stampa ogni 100 iterazioni
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50      # ferma se non migliora per 50 round
)
```

**Output atteso nel terminale:**
```
0:    learn: 0.6213    test: 0.6089    best: 0.6089
100:  learn: 0.5847    test: 0.5412    best: 0.5412
...
Bestmodel at iteration 380
```

---

## 4.2 Valutazione con RPS e Accuracy

```python
# src/evaluate.py
import numpy as np

def ranked_probability_score(y_true, y_proba):
    """
    RPS: più basso è meglio.
    Bookmaker benchmark: ~0.202
    Target CatBoost:     ~0.1925
    """
    n_classes = y_proba.shape[1]
    rps_list = []
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n_classes)[true])
        cum_pred = np.cumsum(prob)
        rps_list.append(np.mean((cum_pred - cum_true) ** 2))
    return np.mean(rps_list)

# Ottieni probabilità e predizioni
probs = model.predict_proba(X_test)
preds = model.predict(X_test).flatten()

# Calcola metriche
rps = ranked_probability_score(y_test.values, probs)
acc = (preds == y_test.values).mean()

print(f"Accuracy : {acc:.4f}   → target > 0.52")
print(f"RPS      : {rps:.4f}   → target < 0.202")
print(f"Batti bookmaker? {'✅ SÌ' if rps < 0.202 else '❌ NO'}")
```

Aggiungi anche un report di classificazione per capire dove sbaglia:

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print(classification_report(y_test, preds, target_names=['Home Win','Draw','Away Win']))

cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['H','D','A'], yticklabels=['H','D','A'])
plt.title('Confusion Matrix — CatBoost Serie A')
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
```

> **Nota critica:** i pareggi (Draw) sono la classe più difficile da predire. Un modello mediocre li ignora quasi sempre — se la recall su D è < 20%, è un segnale di underfit sulla classe mediana.

---

## 4.3 Hyperparameter Tuning con Cross-Validation Temporale

Usa `TimeSeriesSplit` di scikit-learn, **mai** shuffle casuale.

```python
# src/model.py — sezione tuning
from catboost import cv, Pool
from sklearn.model_selection import TimeSeriesSplit

train_pool = Pool(X_train, y_train)

# Griglia di parametri da testare
param_grid = [
    {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6},
    {'iterations': 1000, 'learning_rate': 0.03, 'depth': 8},
    {'iterations': 1000, 'learning_rate': 0.05, 'depth': 6},
]

best_score = 0
best_params = None

for params in param_grid:
    full_params = {
        **params,
        'loss_function': 'MultiClass',
        'eval_metric': 'Accuracy',
        'random_seed': 42
    }
    results = cv(
        pool=train_pool,
        params=full_params,
        fold_count=5,
        shuffle=False,      # CRITICO: mantiene ordine temporale
        stratified=False,
        verbose=False
    )
    score = results['test-Accuracy-mean'].max()
    print(f"Params: {params} → CV Accuracy: {score:.4f}")
    if score > best_score:
        best_score = score
        best_params = params

print(f"\nMigliori parametri: {best_params}")
print(f"CV Accuracy: {best_score:.4f}")
```

Poi riaddestra il modello finale con i migliori parametri **su tutto il training set:**

```python
final_model = CatBoostClassifier(
    **best_params,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False
)
final_model.fit(X_train, y_train)
```

---

## 4.4 SHAP — Interpretabilità del Modello

```python
# Richiede: pip install shap
import shap

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

feature_names = ['π_home_H', 'π_away_H', 'π_home_A', 'π_away_A']

# 1. Importanza globale — tutte le partite del test set
shap.summary_plot(
    shap_values[0],         # classe 0 = Home Win
    X_test,
    feature_names=feature_names,
    plot_type='bar',
    show=False
)
plt.savefig('results/shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. Spiegazione di una singola partita (es. indice 0 = prima del test)
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][0],
    X_test.iloc[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.savefig('results/shap_single_match.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. Beeswarm plot — distribuzione impatto di ogni feature
shap.summary_plot(shap_values[0], X_test, feature_names=feature_names, show=False)
plt.savefig('results/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
```

> **Cosa aspettarsi:** `π_home_H` (pi-rating della squadra di casa nel ruolo casalingo) risulta quasi sempre la feature con SHAP value più alto. Se non è così, c'è probabilmente un bug nel calcolo del pi-rating.

---

## 4.5 Salvataggio del Modello

```python
# Salva
final_model.save_model('models/catboost_serie_a_v1.cbm')
print("Modello salvato in models/catboost_serie_a_v1.cbm")

# Verifica ricaricamento
from catboost import CatBoostClassifier
model_check = CatBoostClassifier()
model_check.load_model('models/catboost_serie_a_v1.cbm')

# Test rapido
test_pred = model_check.predict_proba(X_test[:5])
print("Test ricaricamento OK:", test_pred.shape)  # → (5, 3)
```

---

## Checklist Fase 4

| Step | Obiettivo | Metrica target |
|------|-----------|---------------|
| Baseline training | Modello che gira | Accuracy > 50% |
| Valutazione RPS | Calibrazione probabilità | RPS < 0.210 |
| CV Tuning | Ottimizzazione params | CV Accuracy > 0.52 |
| SHAP analysis | π_home_H = feature #1 | Rank SHAP coerente |
| Salvataggio | File `.cbm` funzionante | Load senza errori |

---

---

# Prompt per AI Agent

```
Sei un esperto di machine learning applicato al calcio.
Il tuo compito è implementare e ottimizzare la Fase 4 
di un sistema di predizione CatBoost + Pi-Rating per la Serie A.

## CONTESTO
- Dataset: Serie A stagioni 2010–2024, ~5.000 partite
- Feature input: 4 colonne (pi_home_home, pi_away_home, 
  pi_home_away, pi_away_away) — snapshot pre-match
- Target: 0=Home Win, 1=Draw, 2=Away Win
- Split: training < agosto 2022, test >= agosto 2022
- File già presenti: data/X_train.csv, X_test.csv, 
  y_train.csv, y_test.csv

## COMPITI DA ESEGUIRE IN ORDINE

1. TRAINING BASELINE
   - Addestra CatBoostClassifier con: iterations=500, 
     lr=0.05, depth=6, loss=MultiClass, seed=42
   - Usa early_stopping_rounds=50 su eval_set di test
   - Logga accuracy e RPS al termine

2. VALUTAZIONE
   - Implementa ranked_probability_score() da zero 
     (non usare librerie esterne)
   - Stampa: Accuracy, RPS, classification_report 
     con classi ['Home','Draw','Away']
   - Salva confusion matrix come PNG in results/

3. HYPERPARAMETER TUNING
   - Usa catboost.cv() con fold_count=5, shuffle=False
   - Testa almeno 4 combinazioni di depth e learning_rate
   - Riaddestra il modello finale sui migliori parametri

4. SHAP ANALYSIS
   - Installa shap se non presente
   - Genera summary_plot (bar) e salvalo in results/
   - Stampa il ranking delle feature per importanza SHAP

5. SALVATAGGIO
   - Salva il modello in models/catboost_serie_a_v1.cbm
   - Verifica che il ricaricamento funzioni con un 
     predict_proba di test

## VINCOLI TECNICI
- Python 3.10 o 3.11
- NON usare random split: sempre TimeSeriesSplit o 
  catboost.cv() con shuffle=False
- NON guardare il test set durante il tuning
- Gestisci eventuali NaN nelle feature prima del fit

## OUTPUT ATTESI
- Accuracy finale: > 0.52
- RPS finale: < 0.202 (benchmark bookmaker)
- File: models/catboost_serie_a_v1.cbm
- Grafici: results/confusion_matrix.png, 
  results/shap_importance.png

## IN CASO DI ERRORI
- Se catboost non installa: pip install catboost --quiet
- Se shap fallisce su TreeExplainer: usa 
  shap.Explainer(model) come fallback
- Se RPS > 0.210: controlla che i pi-rating siano 
  stati calcolati PRIMA di ogni partita (no leakage)

Inizia dal punto 1 e procedi in sequenza.
Dopo ogni step stampa un riepilogo con i risultati 
ottenuti prima di passare al successivo.
```