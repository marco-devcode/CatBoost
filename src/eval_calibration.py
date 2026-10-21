#!/usr/bin/env python3
"""
LA CALIBRAZIONE AIUTA? Confronto pulito, out-of-sample, sul dataset corrente.

IL PROBLEMA
-----------
Due implementazioni della calibrazione Dirichlet davano risposte opposte:

- src/model.py: calibrazione adattata sulle previsioni OUT-OF-SAMPLE di un
  modello AUSILIARIO allenato sulla meta' del training.
- src/experiments.py (prima): calibrazione adattata sulle previsioni
  IN-SAMPLE del modello finale (bug: il modello ha gia' visto quelle
  etichette).

E i due esperimenti non concordavano sul segno (migliora o peggiora). Il
motivo e' che l'ausILIario allenato su meta' dei dati e' molto piu' debole
del modello finale: la sua distribuzione di confidenza e' diversa, quindi i
parametri di calibrazione non trasferiscono.

IL PROTOCOLO CORRETTO (questo script)
--------------------------------------
Confronto apples-to-apples sul TEST set, che nessuno dei modelli ha visto:

  (1) modello allenato sull'80% del training  -> previsioni GREZZE sul test
  (2) lo STESSO modello, calibrato sui restanti 20% del training (out-of-
      sample: quel modello non li ha visti) -> previsioni CALIBRATE sul test
  (3) modello allenato sul 100% del training   -> previsioni GREZZE sul test
  (4) lo STESSO modello, calibrato come in src/model.py (ausiliario su 50%)
      -> previsioni CALIBRATE sul test

Se (2) batte (1), la calibrazione e' utile e si applica al modello di
produzione. Se peggiora, va rimossa: aggiungere complessita' che peggiora il
risultato e' peggio che non calibrare.

Uso:  python src/eval_calibration.py --league seriea
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

from experiments import (  # noqa: E402
    apply_dirichlet_calibration,
    fit_dirichlet_calibration,
)
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
    return (f"  {name:<52} acc={m['accuracy']:>7.2%}  F1mac={m['f1_macro']:.4f}  "
            f"MSE={m['mse_brier']:.5f}  RPS={m['rps']:.5f}  LL={m['log_loss']:.5f}")


def train(X, y, seed=42, iterations=1000, lr=0.03, depth=6):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth,
                           loss_function="MultiClass", eval_metric="Accuracy",
                           random_seed=seed, verbose=False)
    m.fit(X, list(y))
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

    print("=" * 104)
    print(f"  LA CALIBRAZIONE AIUTA? — {league.name} — train={len(X_train)} test={len(X_test)}")
    print("=" * 104)

    # ── (1)/(2): modello sull'80%, calibrazione sul 20% escluso ────────────
    n = len(y_train)
    cut = int(n * 0.8)
    m80 = train(X_train.iloc[:cut], y_train[:cut])
    p80 = m80.predict_proba(X_test)
    m80_cal = metrics(y_test, p80.tolist())
    print(f"\n  {'─'*104}\n  Modello allenato sull'80% del training (n={cut})\n  {'─'*104}")
    print(fmt("(1) GREZZO", m80_cal))

    alpha, w = fit_dirichlet_calibration(y_train[cut:], m80.predict_proba(X_train.iloc[cut:]))
    p80c = apply_dirichlet_calibration(p80, alpha, w)
    m80c = metrics(y_test, p80c.tolist())
    print(fmt("(2) CALIBRATO sul 20% escluso (out-of-sample)", m80c))
    print(f"      alpha={np.round(alpha, 4).tolist()} w={w:.4f}")
    print(f"      -> RPS {m80_cal['rps']:.5f} -> {m80c['rps']:.5f} "
          f"({m80c['rps'] - m80_cal['rps']:+.5f}) | "
          f"LL {m80_cal['log_loss']:.5f} -> {m80c['log_loss']:.5f} "
          f"({m80c['log_loss'] - m80_cal['log_loss']:+.5f})")

    # ── (3)/(4): modello sul 100%, calibrazione come in src/model.py ───────
    m100 = train(X_train, y_train)
    p100 = m100.predict_proba(X_test)
    m100_raw = metrics(y_test, p100.tolist())
    print(f"\n  {'─'*104}\n  Modello allenato sul 100% del training (n={n}) — quello di produzione\n  {'─'*104}")
    print(fmt("(3) GREZZO", m100_raw))

    half = n // 2
    aux = train(X_train.iloc[half:], y_train[half:])
    alpha2, w2 = fit_dirichlet_calibration(y_train[:half], aux.predict_proba(X_train.iloc[:half]))
    p100c = apply_dirichlet_calibration(p100, alpha2, w2)
    m100c = metrics(y_test, p100c.tolist())
    print(fmt("(4) CALIBRATO via ausiliario su 50% (come src/model.py)", m100c))
    print(f"      alpha={np.round(alpha2, 4).tolist()} w={w2:.4f}")
    print(f"      -> RPS {m100_raw['rps']:.5f} -> {m100c['rps']:.5f} "
          f"({m100c['rps'] - m100_raw['rps']:+.5f}) | "
          f"LL {m100_raw['log_loss']:.5f} -> {m100c['log_loss']:.5f} "
          f"({m100c['log_loss'] - m100_raw['log_loss']:+.5f})")

    print(f"\n  {'─'*104}\n  VERDETTO\n  {'─'*104}")
    verdict = []
    d2 = m80c["rps"] - m80_cal["rps"]
    verdict.append(("calibrazione su holdout 20% (protocollo pulito)", d2))
    d4 = m100c["rps"] - m100_raw["rps"]
    verdict.append(("calibrazione via ausiliario 50% (src/model.py)", d4))
    for name, d in verdict:
        mark = "MIGLIORA" if d < -1e-4 else ("peggiora" if d > 1e-4 else "neutra")
        print(f"  {name:<52} RPS {d:+.5f}  -> {mark}")
    print("\n  Se entrambe peggiorano, la calibrazione va RIMOSSA: e' complessita'")
    print("  che non guadagna niente. Se una migliora, si adotta quella.")

    out = ROOT / "models" / f"calibration_check_{league.key}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"m80_grezzo": m80_cal, "m80_calibrato": m80c,
                   "m100_grezzo": m100_raw, "m100_calibrato": m100c,
                   "alpha_holdout": alpha.tolist(), "w_holdout": w,
                   "alpha_aux50": alpha2.tolist(), "w_aux50": w2},
                  f, indent=2, default=float)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
