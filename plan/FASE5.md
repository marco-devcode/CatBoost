# Fase 5 — Feature Engineering (Settimana 2)

---

## Obiettivo

Partire dal modello baseline (~52-54% accuracy con solo 4 feature pi-rating) e portarlo a **56-58%** aggiungendo feature contestuali, di forma e statistiche avanzate come xG.

---

## 5.1 — Rolling Form Recente

**Cosa fa:** calcola i punti medi degli ultimi N match per ogni squadra, separando le prestazioni in casa e in trasferta.

```python
# src/features.py

import pandas as pd
import numpy as np

def rolling_points(df, team_col, goals_for_col, goals_against_col, window=5):
    """
    Calcola i punti medi negli ultimi N match per ogni squadra.
    IMPORTANTE: usa shift(1) per non includere la partita corrente (data leakage!)
    """
    df = df.copy()
    df['points'] = df.apply(
        lambda r: 3 if r[goals_for_col] > r[goals_against_col]
        else (1 if r[goals_for_col] == r[goals_against_col] else 0),
        axis=1
    )
    return df.groupby(team_col)['points'].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# Applica al dataframe principale
df['home_form_5'] = rolling_points(df, 'HomeTeam', 'FTHG', 'FTAG', window=5)
df['away_form_5'] = rolling_points(df, 'AwayTeam', 'FTAG', 'FTHG', window=5)

# Verifica: nessun NaN nelle ultime stagioni
print(df[['home_form_5', 'away_form_5']].isna().sum())
```

> **Impatto atteso:** +1-2 punti percentuali di accuracy da sola.

---

## 5.2 — Goal Scored / Conceded Rolling

Oltre ai punti, aggiungere media gol fatti e subiti cattura la qualità offensiva/difensiva in modo più granulare.

```python
def rolling_goals(df, team_col, goals_col, window=5, label='scored'):
    return df.groupby(team_col)[goals_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

df['home_goals_scored_5']    = rolling_goals(df, 'HomeTeam', 'FTHG', window=5)
df['home_goals_conceded_5']  = rolling_goals(df, 'HomeTeam', 'FTAG', window=5)
df['away_goals_scored_5']    = rolling_goals(df, 'AwayTeam', 'FTAG', window=5)
df['away_goals_conceded_5']  = rolling_goals(df, 'AwayTeam', 'FTHG', window=5)

# Differenziale gol — feature sintetica molto utile
df['home_goal_diff_5'] = df['home_goals_scored_5'] - df['home_goals_conceded_5']
df['away_goal_diff_5'] = df['away_goals_scored_5'] - df['away_goals_conceded_5']
```

---

## 5.3 — Fattore Campo e Giorni di Riposo

**Giorni di riposo:** squadre con meno riposo tendono a performare peggio, specialmente in trasferta.

```python
df = df.sort_values('Date').reset_index(drop=True)

# Calcola l'ultima data giocata per ogni squadra (home o away)
all_teams = pd.concat([
    df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'team'}),
    df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'team'})
]).sort_values('Date')

last_match_date = (
    all_teams.groupby('team')['Date']
    .apply(lambda x: x.shift(1))  # shift per escludere la partita corrente
    .reset_index(level=0, drop=True)
)

all_teams['last_match'] = last_match_date.values
all_teams['rest_days'] = (all_teams['Date'] - all_teams['last_match']).dt.days.fillna(7)

# Merge sul dataframe principale
rest_home = all_teams[all_teams.index.isin(df.index)].groupby('team')['rest_days'].first()

df['home_rest_days'] = df['HomeTeam'].map(
    all_teams.drop_duplicates('team', keep='last').set_index('team')['rest_days']
).fillna(7)

df['away_rest_days'] = df['AwayTeam'].map(
    all_teams.drop_duplicates('team', keep='last').set_index('team')['rest_days']
).fillna(7)

# Differenziale riposo — vantaggio/svantaggio relativo
df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
```

---

## 5.4 — Integrazione xG da Understat

**Expected Goals** è il predittore più potente disponibile gratuitamente. Richiede `soccerdata`.

```bash
# Terminale — installa se non già fatto
pip install soccerdata
```

```python
import soccerdata as sd

# Scarica dati xG da Understat per Serie A
# Nota: prima esecuzione crea cache locale in ~/.soccerdata/
scraper = sd.Understat(leagues='ITA-Serie A', seasons=range(2016, 2024))
schedule = scraper.read_schedule()

# Estrai xG per partita
xg_df = schedule[['home_team', 'away_team', 'date', 'home_xg', 'away_xg']].copy()
xg_df['date'] = pd.to_datetime(xg_df['date'])
xg_df = xg_df.dropna(subset=['home_xg', 'away_xg'])

# Merge con il dataframe principale
df = df.merge(
    xg_df.rename(columns={
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'date': 'Date'
    }),
    on=['HomeTeam', 'AwayTeam', 'Date'],
    how='left'
)

# Rolling xG ultimi 3 match (window=3 ottimale secondo letteratura Bundesliga)
df['home_xg_rolling3'] = df.groupby('HomeTeam')['home_xg'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['away_xg_rolling3'] = df.groupby('AwayTeam')['away_xg'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# xG differenziale
df['xg_diff'] = df['home_xg_rolling3'] - df['away_xg_rolling3']
```

> **Nota:** Se Understat non è disponibile per stagioni vecchie, usare le colonne `HS`, `AS` (tiri) e `HST`, `AST` (tiri in porta) già presenti nei CSV di football-data.co.uk come proxy.

```python
# Proxy xG da tiri in porta (disponibile in football-data.co.uk)
df['home_shots_on_target_ratio'] = df.groupby('HomeTeam')['HST'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
df['away_shots_on_target_ratio'] = df.groupby('AwayTeam')['AST'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
```

---

## 5.5 — Head-to-Head Storico

```python
def h2h_win_rate(df, window=5):
    """Win rate della squadra di casa negli ultimi N scontri diretti"""
    results = []
    for idx, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        past = df[
            (df.index < idx) &
            (
                ((df['HomeTeam'] == h) & (df['AwayTeam'] == a)) |
                ((df['HomeTeam'] == a) & (df['AwayTeam'] == h))
            )
        ].tail(window)
        
        if len(past) == 0:
            results.append(0.4)  # prior neutro
            continue
        
        home_wins = sum(
            (past['HomeTeam'] == h) & (past['FTR'] == 'H') |
            (past['AwayTeam'] == h) & (past['FTR'] == 'A')
        )
        results.append(home_wins / len(past))
    
    return results

df['h2h_home_winrate'] = h2h_win_rate(df, window=5)
```

> **Attenzione:** questa funzione è O(n²) — su 5000 partite richiede ~2-3 minuti. Ottimizzarla con un dizionario se si lavora su dataset più grandi.

---

## 5.6 — Assemblaggio Feature Matrix Finale

```python
# src/features.py — funzione principale

FEATURE_COLS = [
    # Pi-rating (baseline)
    'pi_home_home', 'pi_away_home', 'pi_home_away', 'pi_away_away',
    # Forma recente
    'home_form_5', 'away_form_5',
    'home_goal_diff_5', 'away_goal_diff_5',
    # Riposo
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    # xG (se disponibile)
    'home_xg_rolling3', 'away_xg_rolling3', 'xg_diff',
    # Head to head
    'h2h_home_winrate',
]

df_clean = df[FEATURE_COLS + ['Date', 'FTR']].dropna()

X = df_clean[FEATURE_COLS]
y = df_clean['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Split temporale
cutoff = '2022-08-01'
X_train = X[df_clean['Date'] < cutoff]
y_train = y[df_clean['Date'] < cutoff]
X_test  = X[df_clean['Date'] >= cutoff]
y_test  = y[df_clean['Date'] >= cutoff]

print(f"Feature totali: {len(FEATURE_COLS)}")
print(f"Train: {len(X_train)} partite | Test: {len(X_test)} partite")
```

---

## 5.7 — Verifica Importanza Feature con SHAP

```python
import shap
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function='MultiClass',
    random_seed=42,
    verbose=200
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# SHAP feature importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot — mostra le feature più impattanti per classe "Home Win"
shap.summary_plot(
    shap_values[0],
    X_test,
    feature_names=FEATURE_COLS,
    max_display=12
)
```

---

---

# Prompt per AI Agent

```
Sei un assistente esperto in machine learning applicato al football analytics.

## CONTESTO
Sto sviluppando un modello CatBoost per la predizione dei risultati di partite 
di calcio (Serie A). Il modello baseline usa 4 feature pi-rating e raggiunge 
~52-54% di accuracy con RPS ~0.210. L'obiettivo è portarlo a 56-58% accuracy 
e RPS < 0.200 (sotto il benchmark dei bookmaker).

Il dataset è strutturato così:
- Colonne originali: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A
- Colonne pi-rating già calcolate: pi_home_home, pi_away_home, pi_home_away, pi_away_away
- Dati disponibili: Serie A stagioni 2010-2024 (~5000 partite)
- Python 3.11, librerie: catboost, penaltyblog, soccerdata, pandas, numpy, shap

## REGOLE CRITICHE
1. MAI usare random split — solo split temporale (cutoff date)
2. Tutte le feature devono essere calcolate con shift(1) per evitare data leakage
3. I rating pi devono essere snapshot pre-match
4. Il test set sono le ultime 2 stagioni (dal 2022-08-01 in poi)

## TASK CORRENTE
[DESCRIVI QUI COSA VUOI FARE, esempi:]

→ "Genera il codice completo per calcolare le feature di forma rolling 
   (ultimi 5 match) separando prestazioni home e away per ogni squadra"

→ "Il merge con i dati xG di Understat produce molti NaN. 
   Analizza il problema e proponi una strategia di fallback 
   usando le colonne HST/AST già disponibili"

→ "Il modello con le nuove feature ha accuracy 55.1% ma RPS 0.208 — 
   peggiore del baseline. Spiega perché può succedere e come 
   diagnosticare il problema con SHAP"

→ "Crea una funzione che, dato un nome squadra e un oggetto PiRating aggiornato, 
   restituisca un dizionario con tutte le feature pronte per la predizione 
   della prossima partita"

## OUTPUT ATTESO
- Codice Python pronto all'uso, commentato
- Segnala esplicitamente qualsiasi rischio di data leakage
- Se proponi parametri (window size, soglie), giustificali con la letteratura 
  o con motivazioni empiriche
- Indica il comando pip se servono librerie aggiuntive
```