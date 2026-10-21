#!/usr/bin/env python3
"""
Harness di valutazione ALLINEATO — modello vs bookmaker, train e test.

PERCHE' ESISTE (bug trovato e misurato)
---------------------------------------
Il vecchio `src/evaluate.py` allineava le quote del bookmaker alle righe del
modello PER POSIZIONE:

    df_full["B365H"] = pd.to_numeric(raw["B365H"], errors="coerce").values[:len(df_full)]

Ma `data/serie_a_raw.csv` (output di download_data.py, che concatena le
stagioni scaricate) contiene righe VUOTE (5 righe: indici 1140, 1141, 1142,
1903, 2284 nel file attuale). `clean_data.py` le scarta, quindi il dataset
pulito ha 6140 righe mentre il raw ne ha 6145: dalla riga 1140 in poi ogni
riga del modello riceve le quote di una partita DIVERSA.

Misurato su data/serie_a_raw.csv: 4997 righe su 6140 hanno quote
posizionalmente sbagliate (81% del dataset). Percio' il confronto storico
"bookmaker RPS 0.1789 / accuracy 37.2%" non descrive bet365: descrive un
bookmaker con le quote mescolate a caso. Con l'allineamento corretto
(chiave data+home+away) bet365 reale ha RPS ~0.127 e accuracy ~54.6%.

Questo harness allinea per CHIAVE e riporta tutte le metriche richieste:
accuracy, F1 (macro/weighted/per-classe), MSE (Brier), RPS, log-loss, per
 TRAIN e TEST, piu' il confronto con il bookmaker de-viggato.

Uso:
    python src/evaluate_market.py --league seriea
    python src/evaluate_market.py --league seriea --model models/catboost_seriea_v1.cbm
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

from implied import (  # noqa: E402
    DE_VIG_METHODS,
    accuracy,
    brier_mse,
    f1_scores,
    log_loss,
    ranked_probability_score,
)

FTR_IDX = {"H": 0, "D": 1, "A": 2}
LABELS = ["H", "D", "A"]


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento quote dal RAW (allineamento per CHIAVE, non per posizione)
# ─────────────────────────────────────────────────────────────────────────────
def _dparse(s: str) -> str:
    s = (s or "").strip()
    for f in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, f).date().isoformat()
        except ValueError:
            pass
    return s


def _num(row: dict, col: str):
    try:
        v = float(row[col])
        return v if v > 1.0 else None
    except (KeyError, TypeError, ValueError):
        return None


def load_odds_by_key(raw_path: Path) -> dict:
    """{(date, home, away): {book: (h,d,a)}} letto dal CSV grezzo."""
    out: dict = defaultdict(dict)
    with open(raw_path, newline="", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            date = _dparse(row.get("Date", ""))
            home = (row.get("HomeTeam") or "").strip()
            away = (row.get("AwayTeam") or "").strip()
            if not date or not home or not away:
                continue
            key = (date, home, away)
            books = {
                "b365_open": (_num(row, "B365H"), _num(row, "B365D"), _num(row, "B365A")),
                "b365_close": (_num(row, "B365CH"), _num(row, "B365CD"), _num(row, "B365CA")),
                "pinnacle_open": (_num(row, "PSH"), _num(row, "PSD"), _num(row, "PSA")),
                "pinnacle_close": (_num(row, "PSCH"), _num(row, "PSCD"), _num(row, "PSCA")),
                "avg_open": (_num(row, "AvgH"), _num(row, "AvgD"), _num(row, "AvgA")),
            }
            for name, odds in books.items():
                if all(odds):
                    out[key][name] = odds
    return dict(out)


def load_truth_by_key(features_path: Path) -> dict:
    """{(date, home, away): (y_idx, season)} dal dataset con feature."""
    out = {}
    with open(features_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = (row.get("Date") or "")[:10]
            home = (row.get("HomeTeam") or "").strip()
            away = (row.get("AwayTeam") or "").strip()
            ftr = (row.get("FTR") or "").strip().upper()
            if not date or not home or not away or ftr not in FTR_IDX:
                continue
            out[(date, home, away)] = (FTR_IDX[ftr], (row.get("season") or "").strip())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Metriche
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
        "precision_per_class": {LABELS[k]: v["precision"] for k, v in f1["per_class"].items()},
        "recall_per_class": {LABELS[k]: v["recall"] for k, v in f1["per_class"].items()},
        "mse_brier": brier_mse(y_true, probs),
        "rps": ranked_probability_score(y_true, probs),
        "log_loss": log_loss(y_true, probs),
    }


def fmt_block(name: str, m: dict) -> str:
    return (f"  {name:<34} n={m['n']:>5}  acc={m['accuracy']:>7.2%}  "
            f"F1mac={m['f1_macro']:.4f}  F1w={m['f1_weighted']:.4f}  "
            f"MSE={m['mse_brier']:.5f}  RPS={m['rps']:.5f}  LL={m['log_loss']:.5f}")


def main():
    ap = argparse.ArgumentParser(description="Valutazione allineata modello vs bookmaker")
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--model", default=None, help="path .cbm (default: league.model_path)")
    ap.add_argument("--cutoff", default="2025-08-01", help="split temporale")
    ap.add_argument("--out", default=None, help="path JSON di output")
    args = ap.parse_args()

    from leagues import get_league
    league = get_league(args.league)

    print("=" * 92)
    print(f"  VALUTAZIONE ALLINEATA (chiave data+home+away) — {league.name}")
    print("=" * 92)

    model_path = Path(args.model) if args.model else league.model_path
    if not model_path.exists():
        print(f"ERRORE: modello non trovato: {model_path}")
        print("Il .cbm e' gitignored: generarlo con `python src/model.py --league "
              f"{league.key}`.")
        raise SystemExit(1)

    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    print(f"Modello: {model_path}")

    # ── Dati: X/y dal split, quote dal RAW per chiave ──────────────────────
    X_train = _read_csv(league.split_path("X_train_v2"))
    X_test = _read_csv(league.split_path("X_test_v2"))
    y_train = _read_csv(league.split_path("y_train_v2"))
    y_test = _read_csv(league.split_path("y_test_v2"))

    truth = load_truth_by_key(league.features_path)
    odds_by_key = load_odds_by_key(league.raw_path)

    # Le righe del dataset con feature NON hanno le quote: il truth e' per
    # chiave, quindi per ogni partita del test set cerco (data,home,away).
    # predict_fixtures conosce le quote per chiave; qui ricostruisco la
    # corrispondenza usando le stesse chiavi del calendario.
    print(f"\n  Split: train={len(X_train)} test={len(X_test)}  (cutoff {args.cutoff})")
    print(f"  Truth per chiave: {len(truth)} partite | quote per chiave: {len(odds_by_key)} partite")

    report: dict = {
        "league": league.key,
        "model": str(model_path),
        "cutoff": args.cutoff,
    }

    # ── Probabilita' del modello su train e test ──────────────────────────
    probs_train = model.predict_proba(X_train)
    probs_test = model.predict_proba(X_test)

    # Calibrazione Dirichlet: le probabilita' che il sito pubblica sono
    # quelle CALIBRATE, quindi anche la valutazione deve usare quelle.
    from predict import apply_saved_calibration
    probs_train = np.array([apply_saved_calibration(league, p) for p in probs_train])
    probs_test = np.array([apply_saved_calibration(league, p) for p in probs_test])

    m_train = metrics_block(y_train.astype(int).tolist(), probs_train)
    m_test = metrics_block(y_test.astype(int).tolist(), probs_test)

    print(f"\n{'-'*92}")
    print("  MODELLO")
    print(f"{'-'*92}")
    print(fmt_block("MODELL0 — TRAIN", m_train).replace("MODELL0", "MODELLO"))
    print(fmt_block("MODELLO — TEST", m_test))
    report["model_train"] = m_train
    report["model_test"] = m_test

    # ── Benchmark bookmaker: allineato per chiave sul TEST set ────────────
    # Ricostruisco le chiavi delle partite di test dal dataset con feature:
    # ordine temporale identico a quello usato da build_feature_matrix.
    test_keys = _test_keys(league, args.cutoff)
    print(f"\n{'-'*92}")
    print("  BOOKMAKER (allineato per chiave, TEST set)")
    print(f"{'-'*92}")

    books = ["b365_open", "b365_close", "pinnacle_open", "pinnacle_close", "avg_open"]
    devig_best = "multiplicative"
    bench: dict = {}

    for book in books:
        for method_name, devig in DE_VIG_METHODS.items():
            rows = []
            for i, key in enumerate(test_keys):
                o = odds_by_key.get(key, {}).get(book)
                if not o:
                    continue
                p = devig(o)
                if p is None:
                    continue
                rows.append((i, p))
            if len(rows) < 50:
                continue
            idx = [i for i, _ in rows]
            probs = [p for _, p in rows]
            y = [y_test.astype(int).tolist()[i] for i in idx]
            m = metrics_block(y, probs)
            m["coverage"] = len(rows)
            bench[f"{book}|{method_name}"] = m
            print(fmt_block(f"{book} [{method_name}]", m))

    # Migliore de-vig per bet365 apertura: quello con log-loss minimo.
    cands = {k: v for k, v in bench.items() if k.startswith("b365_open|")}
    if cands:
        best = min(cands, key=lambda k: cands[k]["log_loss"])
        devig_best = best.split("|")[1]
        print(f"\n  -> de-vig migliore per B365 apertura: {devig_best} "
              f"(log-loss {cands[best]['log_loss']:.5f})")
        if "b365_open|multiplicative" in cands:
            mult = cands["b365_open|multiplicative"]
            print(f"     moltiplicativo (storico): log-loss {mult['log_loss']:.5f} "
                  f"RPS {mult['rps']:.5f} acc {mult['accuracy']:.2%}")
            print(f"     delta log-loss = {mult['log_loss'] - cands[best]['log_loss']:+.5f}")
        report["best_devig"] = devig_best
    report["bookmaker"] = bench

    # ── Il modello batte il bookmaker? ─────────────────────────────────────
    print(f"\n{'-'*92}")
    print("  CONFRONTO (TEST set, stessa finestra)")
    print(f"{'-'*92}")
    if cands:
        ref = cands[best]
        print(f"  Modello   RPS {m_test['rps']:.5f}  acc {m_test['accuracy']:.2%}  "
              f"MSE {m_test['mse_brier']:.5f}  LL {m_test['log_loss']:.5f}")
        print(f"  B365      RPS {ref['rps']:.5f}  acc {ref['accuracy']:.2%}  "
              f"MSE {ref['mse_brier']:.5f}  LL {ref['log_loss']:.5f}  "
              f"[{devig_best}, n={ref['coverage']}]")
        print(f"\n  -> sul RPS il modello {'BATTE' if m_test['rps'] < ref['rps'] else 'NON batte'} "
              f"il bookmaker de-viggato (delta {ref['rps'] - m_test['rps']:+.5f})")
        print(f"  -> sull'accuracy il modello {'BATTE' if m_test['accuracy'] > ref['accuracy'] else 'NON batte'} "
              f"(delta {m_test['accuracy'] - ref['accuracy']:+.2%})")
        report["beats_bookmaker_rps"] = bool(m_test["rps"] < ref["rps"])
        report["beats_bookmaker_accuracy"] = bool(m_test["accuracy"] > ref["accuracy"])

    out = Path(args.out) if args.out else (ROOT / "models" / f"eval_market_{league.key}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report JSON: {out}")


def _read_csv(path: Path):
    """Legge un CSV; se ha una sola colonna restituisce una Series.

    I file y_*_v2.csv contengono UNA colonna (l'etichetta 0/1/2) senza
    intestazione: pandas la legge come DataFrame con una colonna di nome
    "0". Qui viene collassata in Series perche' il resto dell'harness lavora
    con liste.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return df[df.columns[0]].squeeze()
    return df


def _test_keys(league, cutoff: str) -> list:
    """Chiavi (date,home,away) delle partite di TEST, nello stesso ordine di X_test_v2.

    Il dataset con le feature (features_path) contiene gia' Date, HomeTeam,
    AwayTeam e FTR: e' la stessa tabella da cui build_feature_matrix ricava
    X_train_v2/X_test_v2. Si replica qui il suo filtro (dropna sulle feature
    BASE, split temporale al cutoff) per ottenere le STESSHE righe, nello
    STESSO ordine: e' la condizione per poter confrontare riga per riga le
    probabilita' del modello con le quote del bookmaker.

    Nota: il confronto NON puo' essere posizionale sul CSV grezzo. Il raw
    concatenato da download_data.py contiene righe vuote che clean_data.py
    scarta, quindi dal pezzo in cui si verifica la riga le quote sono
    sfalsate (misurato: 4997 righe su 6140 con quote sbagliate). Qui si
    abbina per CHIAVE (date, home, away).
    """
    import pandas as pd
    from features import FEATURE_COLS

    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    df_clean = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam"]].copy()
    df_clean = df_clean.dropna(subset=FEATURE_COLS + ["FTR"])
    test = df_clean[df_clean["Date"] >= pd.Timestamp(cutoff)]
    return [
        (r["Date"].strftime("%Y-%m-%d"), str(r["HomeTeam"]).strip(), str(r["AwayTeam"]).strip())
        for _, r in test.iterrows()
    ]


if __name__ == "__main__":
    main()
