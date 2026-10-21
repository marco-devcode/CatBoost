#!/usr/bin/env python3
"""
Sceglie il metodo di de-vig EMPIRICAMENTE, non per citazione.

Domanda: se il bookmaker ha un margine, quale stima delle probabilita' "vere"
e' piu' CALIBRATA rispetto ai risultati reali? Risposta: il metodo con il
log-loss (e RPS) piu' basso sulle partite giocate.

Metodo:
  - per ogni partita con quote B365 (apertura) e risultato, calcolo p_bm con
    ogni metodo di de-vig;
  - misuro log-loss, RPS, accuracy e Brier/MSE del metodo usato come
    PREDITTORE del risultato;
  - il metodo migliore e' quello che minimizza il log-loss.

Questo e' il test decisivo: se il margine fosse distribuito in modo uniforme
(metodo moltiplicativo corretto), il moltiplicativo avrebbe il log-loss
minimo. Se e' disomogeneo, un altro metodo vince.

Uso: python scripts/select_devig.py [--min-season 1011]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import csv  # noqa: E402
import datetime  # noqa: E402
from collections import defaultdict  # noqa: E402

from implied import (  # noqa: E402
    DE_VIG_METHODS,
    brier_mse,
    log_loss,
    ranked_probability_score,
)

FTR_IDX = {"H": 0, "D": 1, "A": 2}


def _dparse(s):
    s = s.strip()
    for f in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    return s


def load_rows(path):
    """(date, home, away, ftr, odds_apertura, odds_chiusura, season)"""
    out = []
    with open(path, newline="", encoding="latin-1") as f:
        r = csv.DictReader(f)
        for row in r:
            date = _dparse(row.get("Date", ""))
            home = (row.get("HomeTeam") or "").strip()
            away = (row.get("AwayTeam") or "").strip()
            ftr = (row.get("FTR") or "").strip().upper()
            if not date or not home or not away or ftr not in FTR_IDX:
                continue
            def num(col):
                try:
                    v = float(row[col])
                    return v if v > 1.0 else None
                except (KeyError, TypeError, ValueError):
                    return None
            out.append({
                "date": date, "home": home, "away": away, "y": FTR_IDX[ftr],
                "open": (num("B365H"), num("B365D"), num("B365A")),
                "close": (num("B365CH"), num("B365CD"), num("B365CA")),
                "pinnacle_open": (num("PSH"), num("PSD"), num("PSA")),
                "avg_open": (num("AvgH"), num("AvgD"), num("AvgA")),
                "season": (row.get("season") or row.get("Season") or "").strip(),
            })
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "serie_a_raw.csv"))
    args = ap.parse_args()

    rows = load_rows(args.raw)
    print(f"Righe con risultato: {len(rows)}")

    for label, key in (("APERTURA B365", "open"), ("CHIUSURA B365", "close")):
        usable = [r for r in rows if all(r[key])]
        print(f"\n{'=' * 74}")
        print(f"  {label} — partite utilizzabili: {len(usable)}")
        print(f"{'=' * 74}")
        if len(usable) < 50:
            print("  (poche partite: salto)")
            continue
        y = [r["y"] for r in usable]
        print(f"  {'metodo':<26} {'log-loss':>10} {'RPS':>10} {'acc':>8} {'MSE':>10}")
        print(f"  {'-' * 68}")
        results = {}
        for name, fn in DE_VIG_METHODS.items():
            probs = []
            for r in usable:
                p = fn(r[key])
                if p is None:
                    probs = None
                    break
                probs.append(p)
            if probs is None:
                print(f"  {name:<26}   (quote non valide)")
                continue
            ll = log_loss(y, probs)
            rps = ranked_probability_score(y, probs)
            acc = sum(1 for t, p in zip(y, probs) if max(range(3), key=lambda k: p[k]) == t) / len(y)
            mse = brier_mse(y, probs)
            results[name] = (ll, rps, acc, mse)
            print(f"  {name:<26} {ll:>10.5f} {rps:>10.5f} {acc:>8.2%} {mse:>10.5f}")
        if results:
            best = min(results, key=lambda k: results[k][0])
            print(f"\n  -> Metodo con log-loss minimo (migliore calibrazione): {best}")
            bl = results[best][0]
            bm = results.get("multiplicative")
            if bm:
                print(f"     multiplicative log-loss {bm[0]:.5f} -> {best} {bl:.5f} "
                      f"(delta {bm[0] - bl:+.5f})")

    # Margine medio per bookmaker (sharp vs soft)
    print(f"\n{'=' * 74}")
    print("  OVERROUND MEDIO per insieme di quote (stesso campionato, stessa partita)")
    print(f"{'=' * 74}")
    have = defaultdict(list)
    for r in rows:
        for label, key in (("B365 apertura", "open"), ("B365 chiusura", "close"),
                           ("Pinnacle apertura", "pinnacle_open"), ("Media mercato", "avg_open")):
            o = r[key]
            if all(o):
                have[label].append(sum(1 / x for x in o) - 1.0)
    for label, vals in have.items():
        if vals:
            vals_sorted = sorted(vals)
            med = vals_sorted[len(vals_sorted) // 2]
            print(f"  {label:<20} n={len(vals):>5}  medio={sum(vals)/len(vals):.4f}  mediana={med:.4f}")


if __name__ == "__main__":
    main()
