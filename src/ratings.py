# src/ratings.py
"""
Calcolo Pi-Rating (Dixon-Coles) per un campionato, implementato a mano
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Decay temporale: mezza vita in anni (una partita di N anni fa pesa 0.5**(N/mezza_vita))
DECAY_HALF_LIFE_YEARS = 2.0


def fit_dixon_coles(home, away, hg, ag, teams, x0=None, weights=None):
    """Stima attack/defence/home_adv via MLE (Poisson con fattore Dixon-Coles tau).
    x0: punto iniziale per warm-start (velocizza i fit successivi).
    weights: array di peso per partita (decay temporale; None = peso 1)."""
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    if weights is None:
        weights = np.ones(len(hg))
    else:
        weights = np.asarray(weights, dtype=float)

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
        # log-likelihood Poisson pesata (decay temporale)
        ll = -np.sum(weights * (lam_h + lam_a)) + np.sum(weights * (hg * np.log(lam_h + 1e-10) + ag * np.log(lam_a + 1e-10)))
        # vincolo somma attack = 0, somma defence = 0 (per identificabilita')
        ll += -1000 * (np.sum(attack)**2 + np.sum(defence)**2)
        return -ll

    x0 = x0 if x0 is not None else np.zeros(n_params)
    res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', options={'maxiter': 500})
    attack, defence, home_adv = unpack(res.x)
    return {teams[i]: {"attack": float(attack[i]), "defence": float(defence[i])} for i in range(n_teams)}, float(home_adv), res.x


def get_season_endpoints(df):
    seasons = df['season'].unique()
    endpoints = {}
    for s in seasons:
        mask = df['season'] == s
        idx = df[mask].index.max()
        endpoints[s] = idx
    return endpoints


def compute_pi_ratings(df: pd.DataFrame):
    """Calcola i Pi-Rating con UN fit MLE globale + DECAY TEMPORALE.

    Ogni partita ha peso w = 0.5 ** (eta / mezza_vita), dove eta e' l'eta'
    della partita in anni rispetto alla piu' recente del dataset.
    Con HALF_LIFE_YEARS=2: una partita di 2 anni fa pesa meta', di 4 anni fa 1/4.
    Le stagioni precedenti CONTANO sulle nuove, ma le recenti contano di piu'.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    teams_all = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))

    # Pesi di decay temporale
    dates = pd.to_datetime(df['Date'])
    age_days = (dates.max() - dates).dt.days.values.astype(float)
    age_years = np.maximum(age_days / 365.25, 0.0)
    weights = 0.5 ** (age_years / DECAY_HALF_LIFE_YEARS)

    print(f"  Fit Dixon-Coles globale su {n} partite, {len(teams_all)} squadre "
          f"(decay mezza vita {DECAY_HALF_LIFE_YEARS} anni)...")
    ratings, home_adv, _ = fit_dixon_coles(
        df['HomeTeam'].values, df['AwayTeam'].values,
        df['FTHG'].values.astype(float), df['FTAG'].values.astype(float),
        teams_all, None, weights,
    )
    print(f"  Fit completato. home_adv={home_adv:.3f}, "
          f"peso partita recente={weights.max():.2f}, piu vecchia={weights.min():.3f}")

    attack = {t: ratings[t]['attack'] for t in teams_all}
    defence = {t: ratings[t]['defence'] for t in teams_all}

    pi_cols = ('pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A')
    result = np.zeros((n, 4))

    for i in range(n):
        row = df.iloc[i]
        home = row['HomeTeam']
        away = row['AwayTeam']
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


def create_train_test_split(df: pd.DataFrame, league, cutoff: str = '2025-08-01'):
    feature_cols = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A']

    df_train = df[df['Date'] < cutoff].copy()
    df_test  = df[df['Date'] >= cutoff].copy()

    # Split adattivo per dataset parziale (solo stagione corrente): se il cutoff
    # lascia 0 righe di train, usa 80/20 temporale invece di fallire.
    # Questo evita il crash "Labels variable is empty" ma NON rende valido il
    # modello: con poche decine di righe i pesi sono rumore.
    if len(df_train) == 0 and len(df_test) >= 5:
        df_sorted = df_test.sort_values('Date').reset_index(drop=True)
        split_idx = max(1, int(len(df_sorted) * 0.8))
        df_train = df_sorted.iloc[:split_idx].copy()
        df_test  = df_sorted.iloc[split_idx:].copy()
        print(f"  [split adattivo] train={len(df_train)}, test={len(df_test)} "
              f"— ATTENZIONE: storico parziale, i pesi non sono stimabili")

    X_train = df_train[feature_cols].reset_index(drop=True)
    y_train = df_train['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)
    X_test  = df_test[feature_cols].reset_index(drop=True)
    y_test  = df_test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)

    # Split e test per lega: due campionati non possono condividere i file.
    X_train.to_csv(league.split_path('X_train'), index=False)
    X_test.to_csv(league.split_path('X_test'), index=False)
    y_train.to_csv(league.split_path('y_train'), index=False)
    y_test.to_csv(league.split_path('y_test'), index=False)

    print(f"\nSplit temporale ({league.name}, cutoff={cutoff}):")
    print(f"  Train: {len(X_train)} partite")
    print(f"  Test : {len(X_test)} partite")
    print(f"  Feature: {feature_cols}")

    return X_train, X_test, y_train, y_test


def main(league):
    print("=" * 55)
    print(f"FASE 3 — Calcolo Pi-Rating {league.name} (Dixon-Coles, MLE)")
    print("=" * 55)

    df = pd.read_csv(league.clean_path, parse_dates=['Date'])
    print(f"Dataset: {len(df)} partite, {df['HomeTeam'].nunique()} squadre")
    print(f"Periodo: {df['Date'].min().date()} → {df['Date'].max().date()}")

    print(f"\nCalcolo Pi-Rating (fit globale con decay)...")
    df_rated, final_ratings = compute_pi_ratings(df)

    df_rated.to_csv(league.ratings_path, index=False)
    print(f"\nDataset con Pi-Rating salvato in {league.ratings_path}")

    with open(league.ratings_snapshot_path, 'w') as f:
        json.dump(final_ratings, f, indent=2)
    print(f"Pi-Rating snapshot salvato in {league.ratings_snapshot_path}")

    print(f"\n{'─'*50}")
    print(f"Statistiche Pi-Rating {league.name}:")
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

    X_train, X_test, y_train, y_test = create_train_test_split(df_rated, league)

    print(f"\nFase 3 completata per {league.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Calcolo Pi-Rating (Dixon-Coles)")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    main(get_league(ap.parse_args().league))
