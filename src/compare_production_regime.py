#!/usr/bin/env python3
"""
CONFRONTO NEL REGIME REALE DI INFERENZA.

Il test set contiene 448 partite: 25/26 (giocate, con tutte le quote dal
CSV storico) e 26/27 (la stagione corrente). In PRODUZIONE, pero', le
partite FUTURE hanno quote da un solo bookmaker (TheOddsAPI): le colonne
Pinnacle e consensus mancano, e il modello le riceve come mancanti.

Questo script confronta i modelli candidati in QUEL regime, perche' e' l'unico
confronto che dice quale modello funziona meglio quando gira davvero:

  A) 41 feature, tutte disponibili               (training: sempre cosi')
  B) 41 feature, Pinnacle/Avg mancanti          (produzione sulle future)
  C) 11 feature selezionate                      (non usa Pinnacle/Avg: mai
                                                  mancanti, nessun disallineamento)
  D) 11 feature, Pinnacle/Avg comunque assenti   (identico a C per costruzione)

Con i parametri scelti sul TRAINING (mode params), non quelli migliori sul
test set.

Uso:  python src/compare_production_regime.py --league seriea
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
from select_features import (  # noqa: E402
    logloss_per_match,
    paired_bootstrap,
    paired_wilcoxon,
    rps_per_match,
    train_model,
)

LABELS = ["H", "D", "A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    args = ap.parse_args()
    from leagues import get_league
    league = get_league(args.league)

    X_train = pd.read_csv(league.split_path("X_train_v2"))
    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_train = pd.read_csv(league.split_path("y_train_v2")).squeeze().astype(int).tolist()
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).tolist()
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    ALL = list(X_train.columns)

    with open(ROOT / "models" / f"selected_features_{league.key}.json", encoding="utf-8") as f:
        selected = json.load(f)["selected"]
    with open(ROOT / "models" / f"selected_params_{league.key}.json", encoding="utf-8") as f:
        best = json.load(f)["best"]
    params = {"iterations": best["iterations"], "lr": best["lr"], "depth": best["depth"]}

    print("=" * 100)
    print(f"  REGIME DI INFERENZA — {league.name} — parametri scelti sul training: {params}")
    print(f"  set semplificato: {len(selected)} feature")
    print("=" * 100)

    # righe del 26/27 (la stagione corrente): le 'future' in produzione
    from leagues import get_league as _gl
    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    from features import FEATURE_COLS
    df = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam"]].copy()
    df = df.dropna(subset=FEATURE_COLS + ["FTR"])
    df = df[df["Date"] >= pd.Timestamp("2025-08-01")].reset_index(drop=True)
    # le righe del test set sono le ultime len(y_test) di df (stesso filtro)
    mask_2627 = (df["Date"] >= pd.Timestamp("2026-06-01")).to_numpy()
    n_2627 = int(mask_2627.sum())

    variants = {}
    m_all = train_model(X_train[ALL], y_train, params["iterations"], params["lr"], params["depth"])
    variants["A_41_tutte"] = (m_all, ALL, X_test[ALL].copy())

    X_prod = X_test[ALL].copy()
    for c in ALL:
        if c.startswith("mkt_pinnacle_") or c.startswith("mkt_avg_"):
            X_prod[c] = np.nan
    variants["B_41_senza_pinnacle_avg"] = (m_all, ALL, X_prod)

    m_sel = train_model(X_train[selected], y_train, params["iterations"], params["lr"], params["depth"])
    variants["C_11_selezionate"] = (m_sel, selected, X_test[selected].copy())

    # D: 32 feature = 41 senza le 8 di Pinnacle/consensus e senza i 3
    # overround. Sono proprio le feature che in PRODUZIONE sulle partite
    # future non esistono (un solo bookmaker da TheOddsAPI): toglierle
    # allinea training e inferenza e semplifica.
    drop = [c for c in ALL if c.startswith("mkt_pinnacle_") or c.startswith("mkt_avg_")
            or c.endswith("_overround")]
    no_ext = [c for c in ALL if c not in drop]
    m_noext = train_model(X_train[no_ext], y_train, params["iterations"], params["lr"], params["depth"])
    variants["D_32_senza_mercato_esteso"] = (m_noext, no_ext, X_test[no_ext].copy())

    print(f"\n  n partite nel test set: {len(y_test)} | stagione 26/27: {n_2627}\n")
    res = {}
    for tag, (m, cols, X) in variants.items():
        p = m.predict_proba(X)
        rps_pm = rps_per_match(y_test, p)
        ll_pm = logloss_per_match(y_test, p)
        preds = [int(np.argmax(x)) for x in p]
        f1 = f1_scores(list(y_test), preds)
        res[tag] = {
            "n_features": len(cols),
            "accuracy": float(accuracy(list(y_test), preds)),
            "f1_macro": float(f1["macro"]),
            "mse": float(brier_mse(list(y_test), p.tolist())),
            "rps": float(ranked_probability_score(list(y_test), p.tolist())),
            "log_loss": float(log_loss(list(y_test), p.tolist())),
            "rps_2627": float(rps_pm[mask_2627].mean()) if n_2627 else None,
            "accuracy_2627": float(np.mean([preds[i] == y_test[i] for i in range(len(y_test))
                                            if mask_2627[i]])) if n_2627 else None,
            "rps_per_match": rps_pm,
        }
        r = res[tag]
        print(f"  {tag:<28} {r['n_features']:>3} feat | acc {r['accuracy']:.2%} | "
              f"F1 {r['f1_macro']:.4f} | RPS {r['rps']:.5f} | "
              f"acc 26/27 {r['accuracy_2627']:.2%} | RPS 26/27 {r['rps_2627']:.5f}")

    print(f"\n  {'─'*100}\n  TEST PAIRED (RPS per partita)\n  {'─'*100}")
    base = "A_41_tutte"
    for tag in ("B_41_senza_pinnacle_avg", "C_11_selezionate",
                "D_32_senza_mercato_esteso"):
        bs = paired_bootstrap(res[base]["rps_per_match"], res[tag]["rps_per_match"])
        wc = paired_wilcoxon(res[base]["rps_per_match"], res[tag]["rps_per_match"])
        print(f"\n  {tag} vs {base}:")
        print(f"    diff media RPS : {bs['diff']:+.6f}   CI 95% [{bs['ci_low']:+.6f}, {bs['ci_high']:+.6f}]")
        print(f"    p bootstrap    : {bs['p_bootstrap']:.4f}   Wilcoxon p: {wc['p']:.4f}")

    # il confronto che conta davvero per la produzione: C (11 feature, sempre
    # disponibili) contro B (41 feature, regime di produzione)
    bs = paired_bootstrap(res["B_41_senza_pinnacle_avg"]["rps_per_match"],
                          res["C_11_selezionate"]["rps_per_match"])
    wc = paired_wilcoxon(res["B_41_senza_pinnacle_avg"]["rps_per_match"],
                         res["C_11_selezionate"]["rps_per_match"])
    print(f"\n  CONFRONTO DI PRODUZIONE  C (11 feature) vs B (41 feature senza Pinnacle/Avg):")
    print(f"    diff media RPS : {bs['diff']:+.6f}   CI 95% [{bs['ci_low']:+.6f}, {bs['ci_high']:+.6f}]")
    print(f"    p bootstrap    : {bs['p_bootstrap']:.4f}   Wilcoxon p: {wc['p']:.4f}")
    verdict = "C non e' peggiore" if bs["p_bootstrap"] >= 0.05 else "C e' peggiore (significativo)"
    print(f"    → {verdict}")
    res["test_C_vs_B"] = {"bootstrap": bs, "wilcoxon": wc}

    out = ROOT / "models" / f"production_regime_{league.key}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"params": params, "selected": selected,
                   "results": {k: {kk: vv for kk, vv in v.items()
                                   if not kk.endswith("_per_match")}
                               for k, v in res.items()}}, f, indent=2, default=float)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
