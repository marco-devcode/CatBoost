# src/ratings.py
"""
Calcolo Pi-Rating per ogni squadra della Serie A.
Basato sul modello Dixon-Coles (penaltyblog).

Feature generate per ogni partita (calcolate PRIMA della partita — no leakage):
  pi_home_H : attack_home - defence_away + home_adv  (forza casa H vs A)
  pi_away_H : attack_away - defence_home - home_adv   (forza trasferta A vs H)
  pi_home_A : defence_home  (componente difensiva di H)
  pi_away_A : attack_away   (componente offensiva di A)
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from penaltyblog.models import DixonColesGoalModel

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "serie_a_clean.csv"
OUT_PATH = DATA_DIR / "serie_a_ratings.csv"
RATINGS_SNAPSHOT_PATH = DATA_DIR / "pi_ratings_snapshot.json"

# Ricalcola il modello alla fine di ogni stagione
RECOMPUTE_SEASONS = True


def get_season_endpoints(df):
    """Ritorna gli indici di fine di ogni stagione."""
    seasons = df['season'].unique()
    endpoints = {}
    for s in seasons:
        mask = df['season'] == s
        idx = df[mask].index.max()
        endpoints[s] = idx
    return endpoints


def compute_pi_ratings(df: pd.DataFrame):
    """
    Calcola i Pi-Rating in modo efficiente:
    - Il modello DixonColes viene fittato una volta per stagione
    - Ogni partita usa i rating della stagione precedente (no leakage)

    Per la prima stagione usiamo rating neutri.
    Per le stagioni successive, usiamo i rating stimati da TUTTE le stagioni precedenti.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    teams_all = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))

    # Rating di default (neutro)
    attack = {t: 1.0 for t in teams_all}
    defence = {t: -1.0 for t in teams_all}
    home_adv = 0.3

    # Matrici dei risultati
    pi_cols = ('pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A')
    result = np.zeros((n, 4))

    season_ends = get_season_endpoints(df)
    sorted_seasons = sorted(season_ends.keys())

    # Teniamo traccia di quali stagioni abbiamo già processato
    processed_seasons = set()

    for i in range(n):
        row = df.iloc[i]
        home = row['HomeTeam']
        away = row['AwayTeam']
        current_season = row['season']

        # Verifica se siamo alla fine di una stagione completata
        # Ricalcola i rating con tutti i dati fino a ora
        if current_season not in processed_seasons:
            if i >= 76:  # almeno 2 giornate
                past = df.iloc[:i]
                try:
                    model = DixonColesGoalModel(
                        goals_home=past['FTHG'].values,
                        goals_away=past['FTAG'].values,
                        teams_home=past['HomeTeam'].values,
                        teams_away=past['AwayTeam'].values,
                    )
                    model.fit(minimizer_options={'maxiter': 1000})
                    params = model.get_params()

                    for t in teams_all:
                        key_a = f"attack_{t}"
                        key_d = f"defence_{t}"
                        if key_a in params:
                            attack[t] = params[key_a]
                        if key_d in params:
                            defence[t] = params[key_d]

                    home_adv = params.get("home_advantage", home_adv)
                    processed_seasons.add(current_season)
                    if len(processed_seasons) % 3 == 0:
                        print(f"  Stagione {current_season}: rating aggiornati con {len(past)} partite")
                except Exception as e:
                    print(f"  [WARN] Stagione {current_season}: fit fallito: {e}")

        # Calcola Pi-Rating con i parametri correnti
        result[i, 0] = attack[home] - defence[away] + home_adv   # pi_home_H
        result[i, 1] = attack[away] - defence[home] - home_adv   # pi_away_H
        result[i, 2] = defence[home]                               # pi_home_A
        result[i, 3] = attack[away]                                # pi_away_A

    for j, col in enumerate(pi_cols):
        df[col] = result[:, j]

    # Dict dei rating finali
    final_ratings = {
        t: {"attack": round(attack[t], 4), "defence": round(defence[t], 4)}
        for t in teams_all
    }
    final_ratings["_global"] = {"home_advantage": round(home_adv, 4)}

    return df, final_ratings


def create_train_test_split(df: pd.DataFrame, cutoff: str = '2022-08-01'):
    """Split temporale e salvataggio CSV."""
    feature_cols = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A']

    df_train = df[df['Date'] < cutoff].copy()
    df_test  = df[df['Date'] >= cutoff].copy()

    X_train = df_train[feature_cols]
    y_train = df_train['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)
    X_test  = df_test[feature_cols]
    y_test  = df_test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_train.to_csv(DATA_DIR / 'X_train.csv', index=False)
    X_test.to_csv(DATA_DIR / 'X_test.csv', index=False)
    y_train.to_csv(DATA_DIR / 'y_train.csv', index=False)
    y_test.to_csv(DATA_DIR / 'y_test.csv', index=False)

    print(f"\nSplit temporale (cutoff={cutoff}):")
    print(f"  Train: {len(X_train)} partite")
    print(f"  Test : {len(X_test)} partite")
    print(f"  Feature: {feature_cols}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("=" * 55)
    print("FASE 3 — Calcolo Pi-Rating Serie A + Split Temporale")
    print("=" * 55)

    df = pd.read_csv(RAW_PATH, parse_dates=['Date'])
    print(f"Dataset: {len(df)} partite, {df['HomeTeam'].nunique()} squadre")
    print(f"Periodo: {df['Date'].min().date()} → {df['Date'].max().date()}")

    print(f"\nCalcolo Pi-Rating (ricalcolo per stagione)...")
    df_rated, final_ratings = compute_pi_ratings(df)

    df_rated.to_csv(OUT_PATH, index=False)
    print(f"\nDataset con Pi-Rating salvato in {OUT_PATH}")

    with open(RATINGS_SNAPSHOT_PATH, 'w') as f:
        json.dump(final_ratings, f, indent=2)
    print(f"Pi-Rating snapshot salvato in {RATINGS_SNAPSHOT_PATH}")

    print(f"\n{'─'*50}")
    print("Statistiche Pi-Rating:")
    for col in ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A']:
        print(f"  {col:<12}: mean={df_rated[col].mean():+.3f}, "
              f"std={df_rated[col].std():.3f}")

    print(f"\nTop 5 squadre per Attack Rating:")
    sorted_teams = sorted(
        [(t, r['attack']) for t, r in final_ratings.items() if t != '_global'],
        key=lambda x: -x[1]
    )
    for team, att in sorted_teams[:5]:
        d = final_ratings[team]['defence']
        print(f"  {team:<20} att={att:+.3f}  def={d:+.3f}")

    X_train, X_test, y_train, y_test = create_train_test_split(df_rated)

    print(f"\n✓ Fase 3 completata — pronto per Fase 4")
