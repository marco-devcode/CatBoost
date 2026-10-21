#!/usr/bin/env python3
"""
Sperimenti sul modello: cosa batte davvero il bookmaker?

Obiettivo: misurare, su TRAIN e TEST, l'effetto di (a) togliere le feature di
mercato, (b) aggiungere calibrazione, (c) usare la chiusura come riferimento.

PERCHE' SERVE (la domanda vera)
---------------------------------
Il modello v2 ha le feature di mercato (quote de-viggate) e cosi' raggiunge
RPS ~0.1302, pari a bet365 (~0.1301): sta COPIANDO il mercato. La domanda
e' se il segnale di CALCIO (Pi-Rating + forma + tiri in porta + Elo) batte
il mercato o no. Per rispondere serve un modello SENZA feature di mercato,
valutato con lo stesso protocollo.

VARIANTI MISURATE
-----------------
  v1_base       : le 17 feature base del repo (pre-modifiche)
  v2_full       : base + prestazione + Elo + mercato
  v2_no_market  : base + prestazione + Elo (SENZA mercato)
  v2_full_cal   : v2_full + calibrazione Dirichlet (su validation)
  v2_nomkt_cal  : v2_no_market + calibrazione Dirichlet
  v1_base_cal   : v1_base + calibrazione Dirichlet

CALIBRAZIONE
-------------
Dirichlet calibration (Kull et al., "Beyond Temperature Scaling: Obtaining
Well-Calibrated Multiclass Probabilities with Dirichlet Calibration",
NeurIPS 2019): si adatta un vettore alpha per classe con
    p(y=k | x) = alpha_k / sum_j alpha_j
e i parametri si ottimizzano minimizzando la NLL sul set di validazione.
Su 3 classi e' 3 parametri, quindi serve un set di validazione: qui e' la
PRIMA metà' del training (temporale, quindi nessun leakage dal test).

Uso:  python src/experiments.py --league seriea
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

from implied import (  # noqa: E402
    DE_VIG_METHODS,
    accuracy,
    brier_mse,
    f1_scores,
    log_loss,
    ranked_probability_score,
)

LABELS = ["H", "D", "A"]


# ─────────────────────────────────────────────────────────────────────────────
# Calibrazione Dirichlet
# ─────────────────────────────────────────────────────────────────────────────
def fit_dirichlet_calibration(y_true, probs, l2: float = 1e-3, iters: int = 400,
                               lr: float = 0.05):
    """Adatta alpha_k su un set di validazione (Kull et al. 2019).

    Restituisce (alpha, w) con p_k = softmax(alpha_k + w * log p_k).
    Il termine w e' quello che permette al modello di correggere la
    sovrastima di confidenza (la temperatura implicita di Dirichlet).
    """
    from scipy.optimize import minimize
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)
    n_classes = probs.shape[1]
    log_p = np.log(np.clip(probs, 1e-12, None))
    onehot = np.eye(n_classes)[y_true]

    def nll(theta):
        alpha = theta[:n_classes]
        w = theta[n_classes]
        z = alpha[None, :] + w * log_p
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        loss = -np.mean(np.sum(onehot * np.log(np.clip(p, 1e-12, None)), axis=1))
        loss += l2 * float(np.sum(alpha ** 2))
        return loss

    theta0 = np.concatenate([np.zeros(n_classes), [0.0]])
    res = minimize(nll, theta0, method="L-BFGS-B",
                   options={"maxiter": iters, "ftol": 1e-12})
    return res.x[:n_classes], float(res.x[n_classes])


def apply_dirichlet_calibration(probs, alpha, w):
    probs = np.asarray(probs, dtype=float)
    log_p = np.log(np.clip(probs, 1e-12, None))
    z = np.asarray(alpha)[None, :] + w * log_p
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione
# ─────────────────────────────────────────────────────────────────────────────
def metrics_block(y_true, probs) -> dict:
    preds = [int(np.argmax(p)) for p in probs]
    f1 = f1_scores(y_true, preds)
    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, preds),
        "f1_macro": f1["macro"],
        "f1_weighted": f1["weighted"],
        "f1_per_class": {LABELS[k]: v["f1"] for k, v in f1["per_class"].items()},
        "mse_brier": brier_mse(y_true, probs),
        "rps": ranked_probability_score(y_true, probs),
        "log_loss": log_loss(y_true, probs),
    }


def fmt(name: str, m: dict) -> str:
    return (f"  {name:<26} n={m['n']:>5}  acc={m['accuracy']:>7.2%}  "
            f"F1mac={m['f1_macro']:.4f}  F1w={m['f1_weighted']:.4f}  "
            f"MSE={m['mse_brier']:.5f}  RPS={m['rps']:.5f}  LL={m['log_loss']:.5f}")


def train_catboost(X_train, y_train, X_test, y_test, seed=42, iterations=1000,
                    learning_rate=0.03, depth=6):
    """Allena CatBoost SENZA usare il test set.

    BUG CORRETTO: prima qui si passava `eval_set=(X_test, y_test)` con
    early stopping. L'early stopping sul test set e' LEAKAGE: il numero di
    iterazioni veniva scelto guardando le prestazioni sul test, quindi i
    numeri riportati erano ottimisticamente falsati (il modello "sapeva"
    quanto fermarsi grazie al test).

    Ora l'early stopping usa una fetta TOTALE del training (ultimi 15%),
    temporale, e il test set non e' mai guardato durante l'allenamento.
    """
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        loss_function="MultiClass", eval_metric="Accuracy",
        random_seed=seed, verbose=False,
    )
    n = len(y_train)
    n_es = max(int(n * 0.15), 20)
    if n - n_es >= 50:
        model.fit(
            X_train.iloc[:n - n_es], list(y_train[:n - n_es]),
            eval_set=(X_train.iloc[n - n_es:], list(y_train[n - n_es:])),
            early_stopping_rounds=50,
        )
    else:
        model.fit(X_train, list(y_train))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from leagues import get_league
    league = get_league(args.league)

    X_train = pd.read_csv(league.split_path("X_train_v2"))
    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_train = pd.read_csv(league.split_path("y_train_v2")).squeeze().astype(int).tolist()
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).tolist()

    base_cols = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A',
                 'home_form_5', 'away_form_5', 'home_goal_diff_5', 'away_goal_diff_5',
                 'home_rest_days', 'away_rest_days', 'h2h_home_winrate',
                 'home_goals_scored_5', 'home_goals_conceded_5',
                 'away_goals_scored_5', 'away_goals_conceded_5',
                 'home_promoted', 'away_promoted']
    perf_cols = [c for c in X_train.columns if c.startswith(('home_shots', 'away_shots',
                                                              'home_corners', 'away_corners',
                                                              'home_cards', 'away_cards',
                                                              'elo_'))]
    mkt_cols = [c for c in X_train.columns if c.startswith("mkt_")]

    # "v2_b365_only": il modello che puo' andare in PRODUZIONE.
    #
    # In inferenza le quote disponibili dipendono dalla fonte: per le partite
    # future c'e' solo TheOddsAPI (un bookmaker = B365). Pinnacle e la media di
    # mercato esistono solo per le partite gia' giocate (dal CSV storico).
    # Un modello allenato con 12 feature di mercato e' quindi addestrato in
    # un regime di missingness DIVERSO da quello dell'inferenza.
    # Questa variante usa solo le 4 feature B365, che sono sempre disponibili:
    # e' l'unica che garantisce allineamento perfetto training/inferenza.
    b365_cols = [c for c in mkt_cols if c.startswith("mkt_b365_")]
    variants = {
        "v1_base": base_cols,
        "v2_full": base_cols + perf_cols + mkt_cols,
        "v2_no_market": base_cols + perf_cols,
        "v2_b365_only": base_cols + perf_cols + b365_cols,
    }
    if not perf_cols:
        print("ATTENZIONE: nessuna feature di prestazione trovata nel dataset.")
    if not mkt_cols:
        print("ATTENZIONE: nessuna feature di mercato trovata nel dataset.")

    print("=" * 100)
    print(f"  ESPERIMENTI — {league.name}  (train={len(X_train)}, test={len(X_test)})")
    print("=" * 100)

    report = {"league": league.key, "variants": {}}
    models = {}

    for name, cols in variants.items():
        cols = [c for c in cols if c in X_train.columns]
        model = train_catboost(X_train[cols], y_train, X_test[cols], y_test)
        models[name] = (model, cols)
        p_tr = model.predict_proba(X_train[cols])
        p_te = model.predict_proba(X_test[cols])
        m_tr, m_te = metrics_block(y_train, p_tr), metrics_block(y_test, p_te)
        print(f"\n{'-'*100}\n  {name}  ({len(cols)} feature)\n{'-'*100}")
        print(fmt(f"{name} TRAIN", m_tr))
        print(fmt(f"{name} TEST", m_te))
        report["variants"][name] = {"n_features": len(cols), "train": m_tr, "test": m_te}

    # ── Calibrazione Dirichlet: validation = prima meta' del training ──────
    print(f"\n{'='*100}\n  CALIBRAZIONE DIRICHLET (validation = prima meta' del training, temporale)\n{'='*100}")
    n_val = len(y_train) // 2
    y_val, y_fit = y_train[:n_val], y_train[n_val:]

    cal_report = {}
    for name, (model, cols) in models.items():
        p_val = model.predict_proba(X_train[cols][:n_val])
        alpha, w = fit_dirichlet_calibration(y_val, p_val)
        p_te_cal = apply_dirichlet_calibration(model.predict_proba(X_test[cols]), alpha, w)
        p_tr_cal = apply_dirichlet_calibration(model.predict_proba(X_train[cols]), alpha, w)
        m_te = metrics_block(y_test, p_te_cal)
        m_te_raw = report["variants"][name]["test"]
        m_tr = metrics_block(y_train, p_tr_cal)
        print(f"\n  {name}_cal   alpha={np.round(alpha, 3).tolist()} w={w:.3f}")
        print(fmt(f"{name}_cal TRAIN", m_tr))
        print(fmt(f"{name}_cal TEST", m_te))
        print(f"      -> RPS test {m_te_raw['rps']:.5f} -> {m_te['rps']:.5f} "
              f"({m_te['rps'] - m_te_raw['rps']:+.5f}) | "
              f"LL {m_te_raw['log_loss']:.5f} -> {m_te['log_loss']:.5f} "
              f"({m_te['log_loss'] - m_te_raw['log_loss']:+.5f})")
        cal_report[name] = {"alpha": alpha.tolist(), "w": w, "train": m_tr, "test": m_te}
        report["variants"][f"{name}_cal"] = {"train": m_tr, "test": m_te,
                                            "alpha": alpha.tolist(), "w": w}

    report["calibration"] = cal_report
    out = Path(args.out) if args.out else (ROOT / "models" / f"experiments_{league.key}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
