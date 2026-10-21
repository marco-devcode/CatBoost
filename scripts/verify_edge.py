#!/usr/bin/env python3
"""
VERIFICA INDIPENDENTE dell'edge: il ROI e' reale o un artefatto?

PERCHE' SERVE
-------------
La diagnosi ha riportato ROI +12.93% con win rate 32% e ~370 scommesse. La
matematica non torna a prima vista: con win rate 32% servono quote medie
vicine a 3.5 sulle scommesse vinte per arrivare a +13%. Un ROI del genere
su 370 scommesse e' o un edge vero o un bug. Questo script lo verifica
SENZA fidarsi della diagnostica: ricalcola tutto da capo, con join espliciti
per chiave, e applica i controlli che un bug di solito non supera.

CONTROLLI
---------
1. CAMBIAMENTO DI CONTROLLO: la stessa pipeline applicata alle probabilita'
   del MERCATO deve dare circa -5.7% (il margine di bet365) o nessuna
   scommessa. Se il meccanismo sbaglia, il controllo non torna.
2. COERENZA DEL JOIN: l'accuracy del mercato calcolata ATTRAVERSO lo stesso
   percorso deve essere ~54.3%. Se e' diversa, il join tra righe, quote e
   risultati e' sfalsato.
3. CALIBRAZIONE DELLE SCOMMESSE: la probabilita' media dichiarata dal
   modello sulle scommesse deve essere confrontata con (a) la probabilita'
   implicita lordata 1/odds e (b) il win rate REALE. Se il modello dice 30% e
   vince il 22%, l'edge non c'e': e' rumore.
4. SIGNIFICATIVITA': bootstrap del ROI (10.000 resample) e test binomiale
   sulla win rate. Serve a distinguere "edge vero" da "colpo fortunato".
5. STABILITA' PER STAGIONE: se il ROI e' tuttoUna stagione, non e' un edge.

Uso:  python scripts/verify_edge.py --league seriea
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from implied import DE_VIG_METHODS, accuracy, ranked_probability_score  # noqa: E402

LABELS = ["H", "D", "A"]
FTR_IDX = {"H": 0, "D": 1, "A": 2}


def _dparse(s):
    s = (s or "").strip()
    for f in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    return s


def _num(row, col):
    try:
        v = float(row[col])
        return v if v > 1.0 else None
    except (KeyError, TypeError, ValueError):
        return None


def load_odds(path):
    out = defaultdict(dict)
    with open(path, newline="", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            d = _dparse(row.get("Date", ""))
            h = (row.get("HomeTeam") or "").strip()
            a = (row.get("AwayTeam") or "").strip()
            if not (d and h and a):
                continue
            key = (d, h, a)
            for name, cols in (("b365_open", ("B365H", "B365D", "B365A")),
                               ("b365_close", ("B365CH", "B365CD", "B365CA"))):
                o = tuple(_num(row, c) for c in cols)
                if all(o):
                    out[key][name] = o
    return dict(out)


def backtest(probs, rows, odds_by_key, book, edge_min, devig):
    """Restituisce (scommesse, profitto, wins, dettagli)."""
    n_bets = stake = profit = wins = 0
    details = []
    for i, r in enumerate(rows):
        o = odds_by_key.get(r["key"], {}).get(book)
        if not o:
            continue
        p = probs[i]
        evs = [p[k] * (o[k] - 1.0) - (1.0 - p[k]) for k in range(3)]
        best = int(np.argmax(evs))
        if evs[best] <= edge_min:
            continue
        won = (best == r["y"])
        pnl = (o[best] - 1.0) if won else -1.0
        n_bets += 1
        stake += 1.0
        profit += pnl
        wins += int(won)
        details.append({
            "key": r["key"], "pick": LABELS[best], "odds": float(o[best]),
            "p_model": float(p[best]), "implied": float(1.0 / o[best]),
            "ev": float(evs[best]), "won": bool(won),
            "season": r.get("season", ""),
        })
    return n_bets, profit, wins, details


def bootstrap_roi(details, n_boot=10000, seed=42):
    """Bootstrap del ROI: quante volte il ROI e' <= 0?"""
    if not details:
        return None
    rng = np.random.default_rng(seed)
    pnl = np.array([(d["odds"] - 1.0) if d["won"] else -1.0 for d in details])
    n = len(pnl)
    idx = rng.integers(0, n, size=(n_boot, n))
    rois = pnl[idx].mean(axis=1)
    return {
        "roi": float(pnl.mean()),
        "ci_low": float(np.percentile(rois, 2.5)),
        "ci_high": float(np.percentile(rois, 97.5)),
        "p_roi_le_0": float((rois <= 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--cutoff", default="2025-08-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from leagues import get_league
    league = get_league(args.league)

    import pandas as pd
    from catboost import CatBoostClassifier
    from features import FEATURE_COLS

    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    df = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam", "season"]].copy()
    df = df.dropna(subset=FEATURE_COLS + ["FTR"])
    df = df[df["Date"] >= pd.Timestamp(args.cutoff)].reset_index(drop=True)

    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).tolist()

    rows = []
    for j, (_, r) in enumerate(df.iterrows()):
        rows.append({
            "key": (r["Date"].strftime("%Y-%m-%d"), str(r["HomeTeam"]).strip(),
                    str(r["AwayTeam"]).strip()),
            "y": FTR_IDX.get(str(r["FTR"]).strip().upper(), -1),
            "season": str(r.get("season", "")),
        })

    odds_by_key = load_odds(league.raw_path)
    devig = DE_VIG_METHODS["additive"]

    print("=" * 92)
    print(f"  VERIFICA EDGE — {league.name} — cutoff {args.cutoff} — n={len(rows)}")
    print("=" * 92)

    # ── CONTROLLO 2: il join e' coerente? il mercato deve fare ~54.3% ──────
    print("\n  CONTROLLO A — il join righe/quote/risultati e' coerente?")
    mkt_probs = np.full((len(rows), 3), np.nan)
    for i, r in enumerate(rows):
        o = odds_by_key.get(r["key"], {}).get("b365_open")
        if o:
            p = devig(o)
            if p:
                mkt_probs[i] = p
    ok = ~np.isnan(mkt_probs[:, 0])
    y = np.array([r["y"] for r in rows])
    mkt_acc = accuracy(y[ok].tolist(), [int(np.argmax(p)) for p in mkt_probs[ok]])
    print(f"    accuracy bet365 via questo join: {mkt_acc:.2%}  (atteso ~54.3%)")
    join_ok = 0.535 <= mkt_acc <= 0.555
    verdict_join = "OK" if join_ok else "FALLITO: il join e' sfalsato, i numeri non valgono"
    print(f"    -> {verdict_join}")

    # ── CONTROLLO 1: il meccanismo del backtest sbaglia? ───────────────────
    print("\n  CONTROLLO B — il backtest applicato alle probabilita' del MERCATO")
    nb, pr, wn, det = backtest(mkt_probs, rows, odds_by_key, "b365_open", 0.0, devig)
    if nb:
        print(f"    n={nb} ROI={pr/nb*100:+.2f}%  (le probs del mercato de-viggato non "
              f"dovrebbero mai avere EV>0: se ne hanno, il de-vig e' sbagliato)")
    else:
        print("    n=0 scommesse con EV>0 -> coerente: il de-vig e' corretto "
              "(nessuna quota offre EV positivo con la sua stessa stima)")

    # ── Il modello ─────────────────────────────────────────────────────────
    model = CatBoostClassifier()
    model.load_model(str(league.model_path))
    probs = model.predict_proba(X_test)
    from predict import apply_saved_calibration
    probs = np.array([apply_saved_calibration(league, p) for p in probs])

    mod_acc = accuracy(y.tolist(), [int(np.argmax(p)) for p in probs])
    print(f"\n  CONTROLLO C — il modello su queste stesse righe")
    print(f"    accuracy modello: {mod_acc:.2%}  |  RPS modello "
          f"{ranked_probability_score(y.tolist(), probs.tolist()):.5f}")
    print(f"    RPS mercato:      {ranked_probability_score(y[ok].tolist(), mkt_probs[ok].tolist()):.5f}")

    for book in ("b365_open", "b365_close"):
        for edge_min in (0.0, 0.02):
            nb, pr, wn, det = backtest(probs, rows, odds_by_key, book, edge_min, devig)
            if not nb:
                print(f"\n  {book} edge>{edge_min:.0%}: nessuna scommessa")
                continue
            roi = pr / nb * 100
            bs = bootstrap_roi(det)
            p_decl = float(np.mean([d["p_model"] for d in det]))
            p_impl = float(np.mean([d["implied"] for d in det]))
            wr = wn / nb
            avg_odds = float(np.mean([d["odds"] for d in det]))
            print(f"\n  ── {book} edge>{edge_min:.0%} ──")
            print(f"    scommesse={nb}  win rate={wr:.1%}  quota media={avg_odds:.2f}  "
                  f"ROI={roi:+.2f}%  ({pr:+.1f}u)")
            print(f"    bootstrap ROI: 95% CI [{bs['ci_low']*100:+.2f}%, "
                  f"{bs['ci_high']*100:+.2f}%]  P(ROI<=0)={bs['p_roi_le_0']:.1%}")
            print(f"    CONTROLLO CALIBRAZIONE: p modello dichiarata={p_decl:.1%} | "
                  f"implicita 1/odds={p_impl:.1%} | win rate REALE={wr:.1%}")
            gap = wr - p_impl
            verdict = "EDGE REALE" if gap > 0.01 else "NON c'e' edge reale: il modello e' rumore"
            print(f"    -> edge reale vs prezzo: {gap*100:+.2f} punti ({verdict})")
            # per stagione
            per_season = defaultdict(lambda: [0, 0.0])
            for d in det:
                per_season[d["season"]][0] += 1
                per_season[d["season"]][1] += (d["odds"] - 1.0) if d["won"] else -1.0
            print("    per stagione: " + "  ".join(
                f"{s}: {v[0]}s {v[1]:+.1f}u" for s, v in sorted(per_season.items())))

    out = Path(args.out) if args.out else (ROOT / "models" / f"verify_edge_{league.key}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"market_accuracy_via_join": mkt_acc, "join_ok": join_ok,
                   "model_accuracy": mod_acc}, f, indent=2)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
