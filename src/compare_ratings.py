# src/compare_ratings.py
"""
Confronto: CatBoost CON Pi-Rating vs SENZA Pi-Rating.
Misura Accuracy e F1 (macro) su entrambi per rispondere alla domanda:
"i Pi-Rating aggiungono segnale predittivo?"

Feature set A (CON):  pi_home_H, pi_away_H, pi_home_A, pi_away_A
                      + home_form_5, away_form_5, home_goal_diff_5, away_goal_diff_5,
                        home_rest_days, away_rest_days, h2h_home_winrate
Feature set B (SENZA): solo le 7 non-Pi-Rating
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

sys.stdout.reconfigure(encoding='utf-8')

from ratings import compute_pi_ratings, DECAY_HALF_LIFE_YEARS
from features import (add_rolling_form, add_rolling_goals, add_rest_days,
                      add_h2h, add_shots_proxy, FEATURE_COLS)
from leagues import get_league

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Lega risolta da riga di comando o da LEAGUE nell'ambiente.
# (OPTIONAL_COLS non esiste piu' in features.py: l'import faceva fallire
# questo script — e il workflow compare.yml che lo esegue — su main.)
LEAGUE = get_league(None)
RAW_PATH = LEAGUE.clean_path

CUTOFF = '2025-08-01'


def load_features():
    df = pd.read_csv(RAW_PATH, parse_dates=['Date'])
    print(f"Dataset: {len(df)} partite")

    print("Calcolo Pi-Rating...")
    df, _ = compute_pi_ratings(df)

    print("Feature engineering...")
    df = add_rolling_form(df, 5)
    df = add_rolling_goals(df, 5)
    df = add_rest_days(df)
    df = add_h2h(df, 5)
    df_raw = pd.read_csv(RAW_PATH)
    if 'HST' in df_raw.columns:
        df['HST'] = df_raw['HST'].values
        df['AST'] = df_raw['AST'].values
        df = add_shots_proxy(df, 5)

    return df


def prepare(df, use_pi: bool):
    cols = (FEATURE_COLS if use_pi else
            [c for c in FEATURE_COLS if not c.startswith('pi_')])
    df_c = df[cols + ['Date', 'FTR']].dropna(subset=cols + ['FTR']).copy()
    train = df_c[df_c['Date'] < CUTOFF]
    test = df_c[df_c['Date'] >= CUTOFF]
    Xtr, ytr = train[cols], train['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    Xte, yte = test[cols], test['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    return Xtr, Xte, ytr, yte


def train_eval(Xtr, Xte, ytr, yte, label):
    model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                               loss_function='MultiClass', verbose=0)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)

    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average='macro')
    f1_per = f1_score(yte, preds, average=None, labels=[0, 1, 2])

    # Log-loss multiclass (piu basso = meglio)
    logloss = log_loss(yte, probs, labels=[0, 1, 2])
    # Brier score multiclass: mean over classes of one-vs-rest Brier (piu basso = meglio)
    y_onehot = np.eye(3)[yte.values]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    print(f"\n[{label}] feature={Xtr.shape[1]}  train={len(Xtr)} test={len(Xte)}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  F1 macro   : {f1:.4f}")
    print(f"  F1 H/D/A   : {[round(x,3) for x in f1_per]}")
    print(f"  Log-loss   : {logloss:.4f}   (piu basso = meglio)")
    print(f"  Brier MC   : {brier:.4f}   (piu basso = meglio)")
    return {"acc": acc, "f1": f1, "logloss": logloss, "brier": brier}


if __name__ == "__main__":
    print("=" * 60)
    print(f"CONFRONTO: CatBoost CON vs SENZA Pi-Rating ({LEAGUE.name})")
    print("=" * 60)

    df = load_features()

    print("\n--- Modello SENZA Pi-Rating ---")
    Xtr_b, Xte_b, ytr_b, yte_b = prepare(df, use_pi=False)
    r_b = train_eval(Xtr_b, Xte_b, ytr_b, yte_b, "SENZA Pi-Rating")

    print("\n--- Modello CON Pi-Rating ---")
    Xtr_a, Xte_a, ytr_a, yte_a = prepare(df, use_pi=True)
    r_a = train_eval(Xtr_a, Xte_a, ytr_a, yte_a, "CON Pi-Rating")

    print("\n" + "=" * 60)
    print("RISULTATO COMPARATIVO")
    print("=" * 60)
    print(f"{'Metrica':<15} {'SENZA PI':>10} {'CON PI':>10} {'Delta':>9}")
    print("-" * 50)
    for key, name in [("acc", "Accuracy"), ("f1", "F1-macro"),
                      ("logloss", "Log-loss"), ("brier", "Brier MC")]:
        v_b, v_a = r_b[key], r_a[key]
        better = "✅" if ((v_a < v_b) if key in ("logloss", "brier") else (v_a > v_b)) else ""
        print(f"{name:<15} {v_b:>10.4f} {v_a:>10.4f} {v_a-v_b:>+9.4f} {better}")
    print("-" * 50)
    wins = sum(1 for k in ("acc", "f1") if r_a[k] > r_b[k]) + \
           sum(1 for k in ("logloss", "brier") if r_a[k] < r_b[k])
    print(f"  I Pi-Rating vincono su {wins}/4 metriche "
          f"{'→ MIGLIORANO il modello ✅' if wins >= 3 else '→ beneficio parziale ⚠️' if wins == 2 else '→ NON migliorano ❌'}")

    # Salva risultati JSON
    RESULTS_DIR = ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "confronto": {
            "senza_pi_rating": r_b,
            "con_pi_rating": r_a,
            "decay_half_life_years": DECAY_HALF_LIFE_YEARS,
            "cutoff": CUTOFF,
            "n_test": int(len(yte_a)),
        },
    }
    with open(ROOT / "results" / "confronto_pi_rating.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Salvato: results/confronto_pi_rating.json")
