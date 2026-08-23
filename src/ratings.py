# src/ratings.py
"""
Calcolo Pi-Rating (Dixon-Coles) per la Serie A, implementato a mano
(senza dipendere da penaltyblog, che falliva silenziosamente lasciando
i rating ai valori di default -> feature costanti).

Il modello Dixon-Coles stima, per ogni squadra, un parametro di attacco
(attack) e uno di difesa (defence), piu' un vantaggio casalingo (home_adv).
Le probabilita' dei gol seguono una Poisson:
    lambda_home = exp(attack_home - defence_away + home_adv)
    lambda_away = exp(attack_away - defence_home)
I parametri sono stimati via Maximum Likelihood (scipy.optimize).

Feature generate per ogni partita (calcolate PRIMA della partita -> no leakage):
    pi_home_H : attack_home - defence_away + home_adv
    pi_away_H : attack_away - defence_home - home_adv
    pi_home_A : defence_home
    pi_away_A : attack_away
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RAW_PATH = DATA_DIR / "serie_a_clean.csv"
OUT_PATH = DATA_DIR / "serie_a_ratings.csv"
RATINGS_SNAPSHOT_PATH = DATA_DIR / "pi_ratings_snapshot.json"


def fit_dixon_coles(home, away, hg, ag, teams):
    """Stima attack/defence/home_adv via MLE (Poisson con fattore Dixon-Coles tau)."""
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    # Parametri: attack[0..n-2], defence[0..n-2], home_adv  (attack ultima squadra = 0 per identificabilita')
    n_params = 2 * (n_teams - 1) + 1

    def unpack(p):
        attack = np.zeros(n_teams)
        defence = np.zeros(n_teams)
        for i in range(n_teams - 1):
            attack[i] = p[i]
            defence[i] = p[n_teams - 1 + i]
        # ultima squadra: attack=0, defence=0 (somma a zero)
        home_adv = p[-1]
        return attack, defence, home_adv

    def neg_log_likelihood(p):
        attack, defence, home_adv = unpack(p)
        hi = np.array([team_idx[h] for h in home])
        ai = np.array([team_idx[a] for a in away])
        lam_h = np.exp(attack[hi] - defence[ai] + home_adv)
        lam_a = np.exp(attack[ai] - defence[hi])
        # log-likelihood Poisson
        ll = -np.sum(lam_h + lam_a) + np.sum(hg * np.log(lam_h + 1e-10) + ag * np.log(lam_a + 1e-10))
        # vincolo somma attack = 0, somma defence = 0 (per identificabilita')
        ll += -1000 * (np.sum(attack)**2 + np.sum(defence)**2)
        return -ll

    x0 = np.zeros(n_params)
    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 500})
    attack, defence, home_adv = unpack(res.x)
    return {teams[i]: {"attack": float(attack[i]), "defence": float(defence[i])} for i in range(n_teams)}, float(home_adv)


def get_season_endpoints(df):
    seasons = df['season'].unique()
    endpoints = {}
    for s in seasons:
        mask = df['season'] == s
        idx = df[mask].index.max()
        endpoints[s] = idx
    return endpoints


def compute_pi_ratings(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    n = len(df)

    teams_all = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))

    attack = {t: 0.0 for t in teams_all}
    defence = {t: 0.0 for t in teams_all}
    home_adv = 0.3

    pi_cols = ('pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A')
    result = np.zeros((n, 4))

    season_ends = get_season_endpoints(df)
    processed_seasons = set()
    last_fit_n = 0

    for i in range(n):
        row = df.iloc[i]
        home = row['HomeTeam']
        away = row['AwayTeam']
        current_season = row['season']

        # Ricalcola i rating con tutti i dati fino a qui (almeno 76 partite = 2 giornate)
        if current_season not in processed_seasons and i - last_fit_n >= 76:
            past = df.iloc[:i]
            try:
                ratings, ha = fit_dixon_coles(
                    past['HomeTeam'].values, past['AwayTeam'].values,
                    past['FTHG'].values.astype(float), past['FTAG'].values.astype(float),
                    teams_all,
                )
                for t in teams_all:
                    attack[t] = ratings[t]['attack']
                    defence[t] = ratings[t]['defence']
                home_adv = ha
                processed_seasons.add(current_season)
                last_fit_n = i
            except Exception as e:
                print(f"  [WARN] fit Dixon-Coles @ {i} fallito: {e}")

        result[i, 0] = attack[home] - defence[away] + home_adv   # pi_home_H
        result[i, 1] = attack[away] - defence[home] - home_adv   # pi_away_H
        result[i, 2] = defence[home]                               # pi_home_A
        result[i, 3] = attack[away]                                # pi_away_A

    for j, col in enumerate(pi_cols):
        df[col] = result[:, j]

    final_ratings = {
        t: {"attack": round(attack[t], 4), "defence": round(defence[t], 4)}
        for t in teams_all
    }
    final_ratings["_global"] = {"home_advantage": round(home_adv, 4)}

    return df, final_ratings


def create_train_test_split(df: pd.DataFrame, cutoff: str = '2022-08-01'):
    feature_cols = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A']

    df_train = df[df['Date'] < cutoff].copy()
    df_test  = df[df['Date'] >= cutoff].copy()

    X_train = df_train[feature_cols].reset_index(drop=True)
    y_train = df_train['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)
    X_test  = df_test[feature_cols].reset_index(drop=True)
    y_test  = df_test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)

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
    print("FASE 3 — Calcolo Pi-Rating Serie A (Dixon-Coles, MLE)")
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
        print(f"  {col:<12}: mean={df_rated[col].mean():+.3f}, std={df_rated[col].std():.3f}")

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
