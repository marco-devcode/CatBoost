# Fase 6 — Valutazione Finale (Giorni 13–14)

---

## 1. Preparazione del test set finale

Prima di tutto, ricostruisci il dataset di test con tutte le feature aggiunte nelle fasi precedenti.

```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# Carica dati e modello
df = pd.read_csv('data/serie_a_features.csv')
model = CatBoostClassifier()
model.load_model('models/catboost_serie_a_v1.cbm')

# Riapplica lo stesso cutoff temporale usato in training
cutoff = '2022-08-01'
df_test = df[df['Date'] >= cutoff].copy()

feature_cols = [
    'pi_home_home', 'pi_away_home',
    'pi_home_away', 'pi_away_away',
    'home_form_5', 'away_form_5',
    'home_rest_days', 'away_rest_days',
    'home_xg_rolling3', 'away_xg_rolling3'
]

X_test = df_test[feature_cols]
y_test = df_test['FTR'].map({'H': 0, 'D': 1, 'A': 2})
```

---

## 2. Calcolo RPS (metrica principale)

```python
def ranked_probability_score(y_true, y_proba):
    n_classes = y_proba.shape[1]
    rps_list = []
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n_classes)[true])
        cum_pred = np.cumsum(prob)
        rps_list.append(np.mean((cum_pred - cum_true) ** 2))
    return np.mean(rps_list)

probs = model.predict_proba(X_test)
y_pred = model.predict(X_test).flatten()

rps_model = ranked_probability_score(y_test.values, probs)
acc_model  = (y_pred == y_test.values).mean()

print(f"Accuracy : {acc_model:.4f}  → target > 0.54")
print(f"RPS      : {rps_model:.4f}  → target < 0.202")
```

---

## 3. Confronto diretto con il bookmaker

Le quote Bet365 sono già incluse nei CSV di football-data.co.uk (`B365H`, `B365D`, `B365A`).

```python
def odds_to_prob(row):
    """Rimuove l'overround e normalizza le quote in probabilità."""
    raw   = [1/row['B365H'], 1/row['B365D'], 1/row['B365A']]
    total = sum(raw)
    return [p / total for p in raw]

# Costruisci matrice di probabilità bookmaker
bm_probs = np.array(df_test.apply(odds_to_prob, axis=1).tolist())
rps_bm   = ranked_probability_score(y_test.values, bm_probs)
acc_bm   = (np.argmax(bm_probs, axis=1) == y_test.values).mean()

print(f"\n{'Modello':<25} {'RPS':>8}  {'Accuracy':>10}")
print("-" * 45)
print(f"{'Il tuo CatBoost':<25} {rps_model:>8.4f}  {acc_model:>9.2%}")
print(f"{'Bookmaker Bet365':<25} {rps_bm:>8.4f}  {acc_bm:>9.2%}")
print(f"\nDelta RPS : {rps_bm - rps_model:+.4f}")
print("✅ Batti il bookmaker in calibrazione!" if rps_model < rps_bm else "⚠️  Bookmaker ancora avanti.")
```

---

## 4. Report completo con classification report

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

label_names = ['Home Win', 'Draw', 'Away Win']

print(classification_report(
    y_test, y_pred,
    target_names=label_names,
    digits=4
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d',
    xticklabels=label_names,
    yticklabels=label_names,
    cmap='Blues'
)
plt.title('Confusion Matrix — CatBoost Serie A')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150)
plt.show()
```

> **Cosa cercare:** il Draw (pareggio) è storicamente la classe più difficile da predire. Se il modello la ignora (tutto classificato come H/A), c'è un problema di class imbalance da affrontare con `class_weights`.

---

## 5. Calibrazione delle probabilità

Un modello può essere accurato ma mal calibrato — le sue probabilità non riflettono le frequenze reali. Verifica e correggi.

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Curva di calibrazione per classe "Home Win"
prob_true, prob_pred = calibration_curve(
    (y_test == 0).astype(int),
    probs[:, 0],
    n_bins=10
)

plt.plot(prob_pred, prob_true, marker='o', label='CatBoost')
plt.plot([0,1],[0,1], 'k--', label='Perfetta calibrazione')
plt.xlabel('Probabilità predetta')
plt.ylabel('Frequenza reale')
plt.title('Calibration Curve — Home Win')
plt.legend()
plt.savefig('results/calibration_curve.png', dpi=150)
plt.show()

# Se la curva devia molto → applica Platt Scaling
from sklearn.linear_model import LogisticRegression
# (da fare su un validation set separato, non sul test!)
```

---

## 6. Analisi profitto simulato (Kelly betting)

```python
def kelly_fraction(p_model, odds, fraction=0.25):
    """Kelly ridotto al 25% per gestione del rischio."""
    b = odds - 1
    q = 1 - p_model
    k = (b * p_model - q) / b
    return max(0, k * fraction)

bankroll = 1000.0
history  = []

for i, (_, row) in enumerate(df_test.iterrows()):
    p_home = probs[i, 0]
    odds_h = row['B365H']
    k      = kelly_fraction(p_home, odds_h)

    if k > 0:
        stake    = bankroll * k
        win      = y_test.iloc[i] == 0          # 0 = Home Win
        bankroll += stake * (odds_h - 1) if win else -stake
        history.append(bankroll)

print(f"Bankroll finale: €{bankroll:.2f}")
plt.plot(history)
plt.title('Simulazione bankroll — Kelly 25%')
plt.xlabel('Scommesse piazzate')
plt.ylabel('Bankroll (€)')
plt.savefig('results/kelly_simulation.png', dpi=150)
plt.show()
```

> ⚠️ Questa è una simulazione didattica su dati storici. Il risultato su test set è sempre ottimistico rispetto al futuro reale.

---

## 7. Salvataggio finale e pacchetto di deployment

```python
import json

# Salva modello
model.save_model('models/catboost_serie_a_v1.cbm')

# Salva metriche in JSON per tracciabilità
metrics = {
    "rps_model"    : round(rps_model, 4),
    "rps_bookmaker": round(rps_bm, 4),
    "accuracy"     : round(float(acc_model), 4),
    "n_test"       : len(y_test),
    "cutoff"       : cutoff,
    "features"     : feature_cols
}
with open('results/metrics_v1.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Modello e metriche salvati.")


# Funzione di inferenza pronta all'uso
def predici_partita(home_team, away_team, pi_ratings, df_features):
    """
    Predice W/D/L con probabilità per una partita futura.
    Richiede pi_ratings aggiornati e le rolling features calcolate.
    """
    h_home, h_away = pi_ratings.get_rating(home_team)
    a_home, a_away = pi_ratings.get_rating(away_team)

    # Recupera le feature di forma più recenti
    home_row = df_features[df_features['HomeTeam'] == home_team].iloc[-1]
    away_row = df_features[df_features['AwayTeam'] == away_team].iloc[-1]

    X = [[
        h_home, h_away, a_home, a_away,
        home_row['home_form_5'],  away_row['away_form_5'],
        home_row['home_rest_days'], away_row['away_rest_days'],
        home_row['home_xg_rolling3'], away_row['away_xg_rolling3']
    ]]

    p = model.predict_proba(X)[0]
    return {
        'home_win' : round(p[0], 3),
        'draw'     : round(p[1], 3),
        'away_win' : round(p[2], 3)
    }

# Esempio
print(predici_partita('Juventus', 'Milan', pi, df))
# → {'home_win': 0.42, 'draw': 0.28, 'away_win': 0.30}
```

---

---

# Prompt per agente AI

```
Sei un assistente esperto in machine learning applicato al football prediction.
Hai accesso a un modello CatBoostClassifier già addestrato ('models/catboost_serie_a_v1.cbm')
e a un dataset Serie A aggiornato ('data/serie_a_features.csv').

## Obiettivo
Esegui la valutazione finale del modello e restituisci un report strutturato.

## Dati disponibili
- CSV principale: data/serie_a_features.csv
  Colonne rilevanti: Date, HomeTeam, AwayTeam, FTR, B365H, B365D, B365A,
  pi_home_home, pi_away_home, pi_home_away, pi_away_away,
  home_form_5, away_form_5, home_rest_days, away_rest_days,
  home_xg_rolling3, away_xg_rolling3
- Test set: partite con Date >= '2022-08-01'
- Encoding label: H=0, D=1, A=2

## Task da eseguire in sequenza

1. CARICA il modello e il dataset. Stampa il numero di partite nel test set.

2. CALCOLA le seguenti metriche sul test set:
   - Accuracy del modello
   - RPS del modello (usa la funzione ranked_probability_score definita sotto)
   - Accuracy del bookmaker (argmax delle probabilità normalizzate da B365H/D/A)
   - RPS del bookmaker
   - Delta RPS (bookmaker - modello): positivo = il modello batte il bookmaker

3. GENERA un classification_report con label ['Home Win', 'Draw', 'Away Win']
   e salva la confusion matrix in results/confusion_matrix.png

4. TRACCIA la calibration curve per la classe Home Win
   e salva in results/calibration_curve.png

5. ESEGUI la simulazione Kelly (fraction=0.25, bankroll iniziale=1000€)
   sulle scommesse Home Win e salva il grafico in results/kelly_simulation.png

6. SALVA un file JSON in results/metrics_v1.json con:
   rps_model, rps_bookmaker, accuracy, n_test, cutoff, features usate

7. RESTITUISCI un sommario testuale con:
   - Le 4 metriche principali (RPS e Accuracy, modello vs bookmaker)
   - Giudizio: il modello supera il target RPS < 0.202?
   - La classe peggio predetta dalla confusion matrix
   - Bankroll finale della simulazione Kelly
   - Un suggerimento concreto per migliorare il modello nella prossima iterazione

## Funzione RPS da usare
def ranked_probability_score(y_true, y_proba):
    n_classes = y_proba.shape[1]
    rps_list = []
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n_classes)[true])
        cum_pred = np.cumsum(prob)
        rps_list.append(np.mean((cum_pred - cum_true) ** 2))
    return np.mean(rps_list)

## Vincoli
- Non usare mai dati del test set per qualsiasi operazione di fitting o scaling.
- Tutti i grafici devono avere titolo, label sugli assi e salvataggio in alta qualità (dpi=150).
- Se mancano colonne xG nel CSV, escludi quelle feature e logga un warning.
- Il sommario finale deve essere stampato in un formato leggibile, non come dump di dizionario.
```