# src/compare_ratings.py
"""
Confronto: CatBoost CON Pi-Rating vs SENZA Pi-Rating.
Misura Accuracy e F1 (macro) su entrambi per rispondere alla domanda:
"i Pi-Rating aggiungono segnale predittivo?"

Feature set A (CON):  pi_home_H, pi_away_H, pi_home_A, pi_away_A
                      + home_form_5, away_form_5, home_goal_diff_5, away_goal_diff_5,
                        home_rest_days, away_rest_days, rest_advantage, h2h_home_winrate
Feature set B (SENZA): solo le 8 non-Pi-Rating
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "serie_a_clean.csv"

CUTOFF = '2022-08-01'


def load_features():
    from ratings import compute_pi_ratings
    from features import (add_rolling_form, add_rolling_goals, add_rest_days,
                          add_h2h, add_shots_proxy, FEATURE_COLS, OPTIONAL_COLS)

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
    acc = accuracy_score(yte, preds)
    f1 = f1_score(yte, preds, average='macro')
    f1_per = f1_score(yte, preds, average=None, labels=[0, 1, 2])
    print(f"\n[{label}] feature={Xtr.shape[1]}  train={len(Xtr)} test={len(Xte)}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 macro : {f1:.4f}")
    print(f"  F1 per-classe H/D/A: {[round(x,3) for x in f1_per]}")
    return acc, f1


if __name__ == "__main__":
    print("=" * 60)
    print("CONFRONTO: CatBoost CON vs SENZA Pi-Rating (Serie A)")
    print("=" * 60)

    df = load_features()

    print("\n--- Modello SENZA Pi-Rating ---")
    Xtr_b, Xte_b, ytr_b, yte_b = prepare(df, use_pi=False)
    acc_b, f1_b = train_eval(Xtr_b, Xte_b, ytr_b, yte_b, "SENZA Pi-Rating")

    print("\n--- Modello CON Pi-Rating ---")
    Xtr_a, Xte_a, ytr_a, yte_a = prepare(df, use_pi=True)
    acc_a, f1_a = train_eval(Xtr_a, Xte_a, ytr_a, yte_a, "CON Pi-Rating")

    print("\n" + "=" * 60)
    print("RISULTATO")
    print("=" * 60)
    print(f"  SENZA Pi-Rating : Accuracy {acc_b:.4f} | F1-macro {f1_b:.4f}")
    print(f"  CON    Pi-Rating : Accuracy {acc_a:.4f} | F1-macro {f1_a:.4f}")
    delta_acc = acc_a - acc_b
    delta_f1 = f1_a - f1_b
    print(f"  Delta           : Accuracy {delta_acc:+.4f} | F1-macro {delta_f1:+.4f}")
    print(f"  I Pi-Rating {'MIGLIORANO' if delta_f1 > 0 else 'NON migliorano'} il modello")
