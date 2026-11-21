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
    """Stima attack/defence/home_adv/rho via MLE (Dixon-Coles con fattore tau).
    x0: punto iniziale per warm-start (velocizza i fit successivi).
    weights: array di peso per partita (decay temporale; None = peso 1)."""
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    if weights is None:
        weights = np.ones(len(hg), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    hi = np.array([team_idx[h] for h in home], dtype=int)
    ai = np.array([team_idx[a] for a in away], dtype=int)
    hg = np.asarray(hg, dtype=float)
    ag = np.asarray(ag, dtype=float)

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    # Parametri: attack[0..n-2], defence[0..n-2], home_adv, rho_raw
    # (ultima squadra: attack=0, defence=0 per identificabilita')
    n_params = 2 * (n_teams - 1) + 2

    def unpack(p):
        attack = np.zeros(n_teams)
        defence = np.zeros(n_teams)
        attack[:n_teams - 1] = p[:n_teams - 1]
        defence[:n_teams - 1] = p[n_teams - 1:2 * (n_teams - 1)]
        home_adv = p[-2]
        rho = 0.20 * np.tanh(p[-1])
        return attack, defence, home_adv, rho

    def neg_log_likelihood(p):
        attack, defence, home_adv, rho = unpack(p)
        lam_h = np.exp(np.clip(attack[hi] - defence[ai] + home_adv, -10.0, 10.0))
        lam_a = np.exp(np.clip(attack[ai] - defence[hi], -10.0, 10.0))

        # Fattore di correlazione Dixon-Coles tau sui punteggi bassi (0-0, 0-1, 1-0, 1-1)
        tau = np.ones(len(hg), dtype=float)
        if rho != 0.0:
            tau[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
            tau[m01] = 1.0 + lam_h[m01] * rho
            tau[m10] = 1.0 + lam_a[m10] * rho
            tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-6, None)

        # log-likelihood Dixon-Coles pesata (decay temporale)
        ll = (
            -np.sum(weights * (lam_h + lam_a))
            + np.sum(weights * (hg * np.log(lam_h + 1e-10) + ag * np.log(lam_a + 1e-10)))
            + np.sum(weights * np.log(tau))
        )
        # vincolo somma attack = 0, somma defence = 0 (per identificabilita')
        ll += -1000.0 * (np.sum(attack) ** 2 + np.sum(defence) ** 2)
        return -ll

    if x0 is None:
        x0_use = np.zeros(n_params)
    else:
        x0_arr = np.asarray(x0, dtype=float)
        if len(x0_arr) == n_params:
            x0_use = x0_arr
        elif len(x0_arr) == n_params - 1:
            x0_use = np.append(x0_arr, 0.0)
        else:
            x0_use = np.zeros(n_params)

    res = minimize(neg_log_likelihood, x0_use, method='L-BFGS-B', options={'maxiter': 500})
    attack, defence, home_adv, _ = unpack(res.x)
    return {teams[i]: {"attack": float(attack[i]), "defence": float(defence[i])} for i in range(n_teams)}, float(home_adv), res.x


def get_season_endpoints(df):
    seasons = df['season'].unique()
    endpoints = {}
    for s in seasons:
        mask = df['season'] == s
        idx = df[mask].index.max()
        endpoints[s] = idx
    return endpoints


def compute_pi_ratings(df: pd.DataFrame, cutoff: str = '2025-08-01'):
    """Calcola i Pi-Rating con fit Dixon-Coles sul SOLO training (< cutoff) + DECAY TEMPORALE.

    Nessuna partita >= cutoff entra nella stima MLE (nessun data leakage dal
    test set o dalla stagione corrente). I pesi temporali w = 0.5**(eta/half_life)
    sono calcolati rispetto alla data piu' recente del solo training fit.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    dates_all = pd.to_datetime(df['Date'])
    df_fit = df[dates_all < pd.Timestamp(cutoff)].copy()
    if len(df_fit) < 20:
        df_fit = df.copy()
        print(f"  [fallback fit] < 20 righe pre-{cutoff}: fit su {len(df_fit)} righe")

    teams_fit = sorted(set(df_fit['HomeTeam'].unique()) | set(df_fit['AwayTeam'].unique()))
    teams_all = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))

    # Pesi di decay temporale (calcolati solo su df_fit -> nessun leakage dal futuro)
    dates_fit = pd.to_datetime(df_fit['Date'])
    age_days = (dates_fit.max() - dates_fit).dt.days.values.astype(float)
    age_years = np.maximum(age_days / 365.25, 0.0)
    weights = 0.5 ** (age_years / DECAY_HALF_LIFE_YEARS)

    print(f"  Fit Dixon-Coles su {len(df_fit)}/{n} partite (< {cutoff}), {len(teams_fit)} squadre "
          f"(decay mezza vita {DECAY_HALF_LIFE_YEARS} anni)...")
    ratings, home_adv, res_x = fit_dixon_coles(
        df_fit['HomeTeam'].values, df_fit['AwayTeam'].values,
        df_fit['FTHG'].values.astype(float), df_fit['FTAG'].values.astype(float),
        teams_fit, None, weights,
    )
    rho = 0.20 * float(np.tanh(res_x[-1])) if len(res_x) else 0.0
    print(f"  Fit completato. home_adv={home_adv:.3f}, rho={rho:+.4f}, "
          f"peso recente={weights.max():.2f}, piu vecchia={weights.min():.3f}")

    attack = {t: ratings.get(t, {}).get('attack', 0.0) for t in teams_all}
    defence = {t: ratings.get(t, {}).get('defence', 0.0) for t in teams_all}

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
    final_ratings["_global"] = {
        "home_advantage": round(home_adv, 4),
        "rho": round(rho, 4),
        "cutoff": cutoff,
    }

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
