#!/usr/bin/env python3
"""
Quanto degrada il modello quando le quote Pinnacle/Avg NON sono disponibili?

IL PROBLEMA
-----------
In TRAINING il modello ha 41 feature, di cui 12 di mercato:
  mkt_b365_*     (4)  sempre disponibili
  mkt_pinnacle_* (4)  disponibili solo dove il CSV storico le ha
  mkt_avg_*      (4)  disponibili solo dove il CSV storico le ha

In INFERENZA la situazione e' diversa per tipo di partita:
  - partite GIA GIOCATE: le quote arrivano dal CSV storico -> tutte e 12
  - partite FUTURE: arriva solo UN bookmaker da TheOddsAPI -> solo 4

Il modello e' quindi addestrato in un regime di missingness diverso da
quello delle future. CatBoost gestisce i NaN, ma la capacita' predittiva
cambia. Questo script misura QUANTO cambia.

METODO
------
1. Addestra v2_full (come in src/model.py) sul training.
2. Valuta sul test set in TRE condizioni:
   a) tutte le feature di mercato disponibili (la situazione del training);
   b) Pinnacle e Avg azzerate a NaN (la situazione delle partite future);
   c) solo B365, modello ALLENATO con solo B365 (v2_b365_only).
3. Confronta: se (b) e' vicino a (a), si puo' tenere v2_full anche per le
   future. Se (b) e' molto peggio, il modello va ri-allestrato.

Uso:  python src/eval_missing_market.py --league seriea
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from implied import accuracy, brier_mse, f1_scores, log_loss, ranked_probability_score  # noqa: E402
from leagues import get_league  # noqa: E402

LABELS = ["H", "D", "A"]


def metrics(y, probs):
    preds = [int(np.argmax(p)) for p in probs]
    f1 = f1_scores(y, preds)
    return {
        "n": len(y),
        "accuracy": accuracy(y, preds),
        "f1_macro": f1["macro"],
        "mse_brier": brier_mse(y, probs),
        "rps": ranked_probability_score(y, probs),
        "log_loss": log_loss(y, probs),
    }


def fmt(name, m):
    return (f"  {name:<46} acc={m['accuracy']:>7.2%}  F1mac={m['f1_macro']:.4f}  "
            f"MSE={m['mse_brier']:.5f}  RPS={m['rps']:.5f}  LL={m['log_loss']:.5f}")


def train(X_train, y_train, seed=42, iterations=1000, lr=0.03, depth=6):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth,
                           loss_function="MultiClass", eval_metric="Accuracy",
                           random_seed=seed, verbose=False)
    m.fit(X_train, list(y_train))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    args = ap.parse_args()
    league = get_league(args.league)

    X_train = pd.read_csv(league.split_path("X_train_v2"))
    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_train = pd.read_csv(league.split_path("y_train_v2")).squeeze().astype(int).tolist()
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).tolist()

    base = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A', 'home_form_5',
            'away_form_5', 'home_goal_diff_5', 'away_goal_diff_5', 'home_rest_days',
            'away_rest_days', 'h2h_home_winrate', 'home_goals_scored_5',
            'home_goals_conceded_5', 'away_goals_scored_5', 'away_goals_conceded_5',
            'home_promoted', 'away_promoted']
    perf = [c for c in X_train.columns if c.startswith(
        ('home_shots', 'away_shots', 'home_corners', 'away_corners',
         'home_cards', 'away_cards', 'elo_'))]
    mkt = [c for c in X_train.columns if c.startswith("mkt_")]
    b365 = [c for c in mkt if c.startswith("mkt_b365_")]
    cols_full = base + perf + mkt
    cols_b365 = base + perf + b365

    print("=" * 100)
    print(f"  DEGRADO PER MERCATO MANCANTE — {league.name}")
    print("=" * 100)

    m_full = train(X_train[cols_full], y_train)
    p_test = m_full.predict_proba(X_test[cols_full])

    print(f"\n  {'─'*100}\n  v2_full: allenato con tutte le 12 feature di mercato\n  {'─'*100}")
    print(fmt("(a) test con TUTTO il mercato (situazione training)", metrics(y_test, p_test.tolist())))

    # (b) Pinnacle e Avg azzerate a NaN: la situazione delle partite FUTURE
    X_test_nomkt = X_test[cols_full].copy()
    for c in mkt:
        if not c.startswith("mkt_b365_"):
            X_test_nomkt[c] = np.nan
    p_nomkt = m_full.predict_proba(X_test_nomkt)
    print(fmt("(b) test con Pinnacle/Avg = NaN (situazione future)", metrics(y_test, p_nomkt.tolist())))

    # (c) modello ALLENATO solo con B365
    m_b365 = train(X_train[cols_b365], y_train)
    p_b365 = m_b365.predict_proba(X_test[cols_b365])
    print(fmt("(c) modello ALLENATO solo con B365", metrics(y_test, p_b365.tolist())))

    m_a = metrics(y_test, p_test.tolist())
    m_b = metrics(y_test, p_nomkt.tolist())
    m_c = metrics(y_test, p_b365.tolist())
    print(f"\n  -> (b) vs (a): RPS {m_a['rps']:.5f} -> {m_b['rps']:.5f} "
          f"({m_b['rps'] - m_a['rps']:+.5f}) | acc {m_a['accuracy']:.2%} -> {m_b['accuracy']:.2%}")
    print(f"  -> (c) vs (a): RPS {m_a['rps']:.5f} -> {m_c['rps']:.5f} "
          f"({m_c['rps'] - m_a['rps']:+.5f}) | acc {m_a['accuracy']:.2%} -> {m_c['accuracy']:.2%}")
    print(f"\n  -> DECISIONE: se (b) e' accettabile, si tiene v2_full anche per le "
          f"partite future (le quote mancanti sono NaN, il modello le gestisce).")

    out = ROOT / "models" / f"missing_market_{league.key}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"full": m_a, "full_senza_pinnacle_avg": m_b, "solo_b365": m_c},
                  f, indent=2, default=float)
    print(f"  Report: {out}")


if __name__ == "__main__":
    main()
