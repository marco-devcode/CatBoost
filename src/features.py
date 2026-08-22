# src/features.py
"""
Fase 5 — Feature Engineering per CatBoost Serie A.

Aggiunge feature contestuali al di sopra dei 4 Pi-Rating baseline:
  1. Rolling Form (punti ultimi 5 match) — home/away separati
  2. Rolling Goals (gol fatti/subiti ultimi 5 match)
  3. Goal Difference Rolling
  4. Rest Days (giorni di riposo)
  5. Head-to-Head storico
  6. Shots on Target proxy per xG

Tutte le feature usano shift(1) per evitare data leakage.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RATINGS_PATH = DATA_DIR / "serie_a_ratings.csv"
RAW_PATH = DATA_DIR / "serie_a_clean.csv"
OUT_PATH = DATA_DIR / "serie_a_features.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.1 Rolling Points Form (ultimi N match, home/away separati)
# ═══════════════════════════════════════════════════════════════════════════════
def add_rolling_form(df, window=5):
    """Calcola i punti medi negli ultimi N match per ogni squadra."""
    df = df.copy()

    # Punti per ogni partita (prospetta home)
    df['points_home'] = np.where(
        df['FTHG'] > df['FTAG'], 3,
        np.where(df['FTHG'] == df['FTAG'], 1, 0)
    )
    df['points_away'] = np.where(
        df['FTAG'] > df['FTHG'], 3,
        np.where(df['FTAG'] == df['FTHG'], 1, 0)
    )

    # Rolling per squadra di casa (performance in casa)
    form_home = (
        df.groupby('HomeTeam')['points_home']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Rolling per squadra ospite (performance in trasferta)
    form_away = (
        df.groupby('AwayTeam')['points_away']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df[f'home_form_{window}'] = form_home
    df[f'away_form_{window}'] = form_away

    df.drop(columns=['points_home', 'points_away'], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2 Rolling Goals Scored / Conceded
# ═══════════════════════════════════════════════════════════════════════════════
def add_rolling_goals(df, window=5):
    """Media gol fatti e subiti negli ultimi N match."""
    df = df.copy()

    df[f'home_goals_scored_{window}'] = (
        df.groupby('HomeTeam')['FTHG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df[f'home_goals_conceded_{window}'] = (
        df.groupby('HomeTeam')['FTAG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df[f'away_goals_scored_{window}'] = (
        df.groupby('AwayTeam')['FTAG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df[f'away_goals_conceded_{window}'] = (
        df.groupby('AwayTeam')['FTHG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df[f'home_goal_diff_{window}'] = (
        df[f'home_goals_scored_{window}'] - df[f'home_goals_conceded_{window}']
    )
    df[f'away_goal_diff_{window}'] = (
        df[f'away_goals_scored_{window}'] - df[f'away_goals_conceded_{window}']
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.3 Rest Days (giorni di riposo)
# ═══════════════════════════════════════════════════════════════════════════════
def add_rest_days(df):
    """Calcola i giorni di riposo per ogni squadra prima di ogni partita."""
    df = df.copy()

    home_dates = df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'team'})
    away_dates = df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'team'})
    all_dates = pd.concat([home_dates, away_dates]).sort_values('Date').reset_index(drop=True)

    all_dates['last_match'] = all_dates.groupby('team')['Date'].shift(1)
    all_dates['rest_days'] = (all_dates['Date'] - all_dates['last_match']).dt.days

    home_rest = (
        all_dates[all_dates['Date'].isin(df['Date'])]
        .groupby(['Date', 'team'])['rest_days']
        .first()
        .reset_index()
        .rename(columns={'team': 'HomeTeam', 'rest_days': 'home_rest_days'})
    )
    away_rest = (
        all_dates[all_dates['Date'].isin(df['Date'])]
        .groupby(['Date', 'team'])['rest_days']
        .first()
        .reset_index()
        .rename(columns={'team': 'AwayTeam', 'rest_days': 'away_rest_days'})
    )

    df = df.merge(home_rest, on=['Date', 'HomeTeam'], how='left')
    df = df.merge(away_rest, on=['Date', 'AwayTeam'], how='left')

    df['home_rest_days'] = df['home_rest_days'].fillna(7)
    df['away_rest_days'] = df['away_rest_days'].fillna(7)

    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.4 Head-to-Head Storico
# ═══════════════════════════════════════════════════════════════════════════════
def add_h2h(df, window=5):
    """Win rate della squadra di casa negli ultimi N scontri diretti."""
    df = df.copy()
    results = []
    h2h_cache = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        h, a = row['HomeTeam'], row['AwayTeam']
        key = (min(h, a), max(h, a))

        if key in h2h_cache:
            past_matches = h2h_cache[key]
            past_matches = [m for m in past_matches if m['idx'] < idx]
        else:
            past_matches = []

        if len(past_matches) == 0:
            results.append(0.4)  # prior neutro (leggero vantaggio casa)
        else:
            recent = past_matches[-window:]
            home_wins = sum(1 for m in recent if m['winner'] == 'home')
            results.append(home_wins / len(recent))

        if key not in h2h_cache:
            h2h_cache[key] = []

        if row['FTR'] == 'H':
            winner = 'home'
        elif row['FTR'] == 'A':
            winner = 'away'
        else:
            winner = 'draw'

        h2h_cache[key].append({'idx': idx, 'winner': winner})

    df['h2h_home_winrate'] = results
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.5 Shots on Target proxy (se disponibili nel CSV grezzo)
# ═══════════════════════════════════════════════════════════════════════════════
def add_shots_proxy(df, window=5):
    """Usa tiri in porta come proxy per xG (dati football-data.co.uk)."""
    df = df.copy()

    has_shots = 'HST' in df.columns and 'AST' in df.columns

    if has_shots:
        df[f'home_shots_target_{window}'] = (
            df.groupby('HomeTeam')['HST']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f'away_shots_target_{window}'] = (
            df.groupby('AwayTeam')['AST']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df['shots_diff'] = df[f'home_shots_target_{window}'] - df[f'away_shots_target_{window}']
    else:
        print("  [INFO] Colonne HST/AST non disponibili, skipping shots proxy")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Assemblaggio Feature Matrix
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    # Pi-rating (baseline)
    'pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A',
    # Forma recente
    'home_form_5', 'away_form_5',
    'home_goal_diff_5', 'away_goal_diff_5',
    # Riposo
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    # Head to head
    'h2h_home_winrate',
]

# Feature opzionali (se disponibili)
OPTIONAL_COLS = [
    'home_shots_target_5', 'away_shots_target_5', 'shots_diff',
    'home_goals_scored_5', 'home_goals_conceded_5',
    'away_goals_scored_5', 'away_goals_conceded_5',
]


def build_feature_matrix(df, cutoff='2022-08-01'):
    """Costruisce la matrice feature completa e fa il split temporale."""
    df_clean = df[FEATURE_COLS + ['Date', 'FTR']].copy()

    for col in OPTIONAL_COLS:
        if col in df.columns:
            df_clean[col] = df[col]

    df_clean = df_clean.dropna(subset=FEATURE_COLS + ['FTR'])

    df_train = df_clean[df_clean['Date'] < cutoff].copy()
    df_test  = df_clean[df_clean['Date'] >= cutoff].copy()

    feature_cols = [c for c in df_clean.columns if c not in ('Date', 'FTR')]

    X_train = df_train[feature_cols].reset_index(drop=True)
    y_train = df_train['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)
    X_test  = df_test[feature_cols].reset_index(drop=True)
    y_test  = df_test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)

    return X_train, X_test, y_train, y_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("FASE 5 — Feature Engineering (Serie A)")
    print("=" * 55)

    df = pd.read_csv(RATINGS_PATH, parse_dates=['Date'])
    print(f"Dataset: {len(df)} partite")

    print("\n5.1 — Rolling Form...")
    df = add_rolling_form(df, window=5)

    print("5.2 — Rolling Goals...")
    df = add_rolling_goals(df, window=5)

    print("5.3 — Rest Days...")
    df = add_rest_days(df)

    print("5.4 — Head-to-Head (può richiedere 1-2 min)...")
    df = add_h2h(df, window=5)

    print("5.5 — Shots on Target proxy...")
    df_raw = pd.read_csv(RAW_PATH)
    if 'HST' in df_raw.columns and 'AST' in df_raw.columns:
        df['HST'] = df_raw['HST'].values
        df['AST'] = df_raw['AST'].values
        df = add_shots_proxy(df, window=5)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nDataset con feature salvato in {OUT_PATH}")

    print(f"\n{'─'*50}")
    print("Feature generate:")
    all_feature_cols = [c for c in df.columns if c not in
                        ('Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'season',
                         'HST', 'AST', 'HS', 'AS')]
    for col in all_feature_cols:
        print(f"  {col:<25} mean={df[col].mean():.3f}, std={df[col].std():.3f}")

    X_train, X_test, y_train, y_test, feature_cols = build_feature_matrix(df)

    print(f"\n{'─'*50}")
    print(f"Feature matrix:")
    print(f"  Feature ({len(feature_cols)}): {feature_cols}")
    print(f"  Train: {X_train.shape}")
    print(f"  Test : {X_test.shape}")

    X_train.to_csv(DATA_DIR / 'X_train_v2.csv', index=False)
    X_test.to_csv(DATA_DIR / 'X_test_v2.csv', index=False)
    y_train.to_csv(DATA_DIR / 'y_train_v2.csv', index=False)
    y_test.to_csv(DATA_DIR / 'y_test_v2.csv', index=False)
    print(f"\nSplit v2 salvato in data/")

    print(f"\n✓ Fase 5 completata — pronto per re-training")
