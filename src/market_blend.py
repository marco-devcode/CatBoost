#!/usr/bin/env python3
"""
SHRINKAGE SUL MERCATO: la combinazione modello + consensus di mercato.

PERCHE' (il dato che apre la strada)
--------------------------------------
Misurato sul test set, de-vig corretto:

    bet365 apertura   RPS 0.13024
    bet365 chiusura   RPS 0.12983
    media di mercato  RPS 0.12990   <- il CONSENSUS e' migliore di bet365
    Pinnacle chiusura RPS 0.12901   (n=196, coperertura minore)

Il consensus di mercato e' una previsione MIGLIORE di bet365: bet365 e' un
bookmaker "soft" (overround 5.67% contro il 2.61% di Pinnacle), quindi la sua
probabilita' de-viggata e' piu' rumorosa.

Il modello da solo (senza prior di mercato) fa RPS 0.13166: peggio del
consensus. Ma se il modello ha un SEGNALE PROPRIO (piccolo e reale), allora
la combinazione dei due batte entrambi. E' il classico "wisdom of crowds":
il mercato e' il miglior predittore in assoluto, e un segnale privato piccolo
ma indipendente, combinato con il mercato, batte il mercato.

    p_final = w * p_model + (1 - w) * p_mercato

Il peso w e' l'unico parametro libero, e va FITTATO su dati che il modello
non ha visto (validation out-of-sample), non sul test set.

PERCHE' UNA SHRINKAGE LINEARE E NON IL MODELO STESSO
------------------------------------------------------
Il modello gia' riceve le quote come FEATURE (mkt_*): puo' usarle, ma gli
split di un albero non ottimizzano una media lineare di probabilita'. La
shrinkage applica la combinazione DIRETTAMENTE sulle probabilita' finali,
quindi e' esattamente la forma "modello + prior di mercato" che la letteratura
usa, ed e' ottimizzabile in modo diretto sul log-loss.

Uso:  python src/market_blend.py --league seriea
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

from implied import DE_VIG_METHODS, accuracy, brier_mse, f1_scores, log_loss, ranked_probability_score  # noqa: E402

LABELS = ["H", "D", "A"]
FTR_IDX = {"H": 0, "D": 1, "A": 2}


def _dparse(s: str) -> str:
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


def load_odds(raw_path: Path) -> dict:
    out: dict = defaultdict(dict)
    with open(raw_path, newline="", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            date = _dparse(row.get("Date", ""))
            home = (row.get("HomeTeam") or "").strip()
            away = (row.get("AwayTeam") or "").strip()
            if not date or not home or not away:
                continue
            key = (date, home, away)
            for name, cols in (("b365_open", ("B365H", "B365D", "B365A")),
                               ("b365_close", ("B365CH", "B365CD", "B365CA")),
                               ("pinnacle_open", ("PSH", "PSD", "PSA")),
                               ("pinnacle_close", ("PSCH", "PSCD", "PSCA")),
                               ("avg_open", ("AvgH", "AvgD", "AvgA"))):
                o = tuple(_num(row, c) for c in cols)
                if all(o):
                    out[key][name] = o
    return dict(out)


def metrics(y, probs) -> dict:
    preds = [int(np.argmax(p)) for p in probs]
    f1 = f1_scores(y, preds)
    return {
        "n": len(y),
        "accuracy": accuracy(y, preds),
        "f1_macro": f1["macro"],
        "f1_weighted": f1["weighted"],
        "mse_brier": brier_mse(y, probs),
        "rps": ranked_probability_score(y, probs),
        "log_loss": log_loss(y, probs),
    }


def fmt(name, m) -> str:
    return (f"  {name:<40} n={m['n']:>4}  acc={m['accuracy']:>7.2%}  "
            f"F1mac={m['f1_macro']:.4f}  MSE={m['mse_brier']:.5f}  "
            f"RPS={m['rps']:.5f}  LL={m['log_loss']:.5f}")


def blend(p_model, p_market, w):
    """p_final = w * p_model + (1 - w) * p_market."""
    return w * np.asarray(p_model) + (1.0 - w) * np.asarray(p_market)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--cutoff", default="2025-08-01")
    ap.add_argument("--market", default="avg_open",
                    choices=["avg_open", "b365_open", "b365_close", "pinnacle_close"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from leagues import get_league
    league = get_league(args.league)

    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(league.model_path))

    import pandas as pd
    from features import FEATURE_COLS

    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    df = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam", "season"]].copy()
    df = df.dropna(subset=FEATURE_COLS + ["FTR"])
    df = df[df["Date"] >= pd.Timestamp(args.cutoff)].reset_index(drop=True)

    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).tolist()
    probs = model.predict_proba(X_test)
    from predict import apply_saved_calibration
    probs = np.array([apply_saved_calibration(league, p) for p in probs])

    odds_by_key = load_odds(league.raw_path)
    devig = DE_VIG_METHODS["additive"]

    # Allineamento per chiave: le righe del test set hanno le stesse chiavi
    # delle righe di df (stesso ordine, stessa tabella).
    keys = [(r["Date"].strftime("%Y-%m-%d"), str(r["HomeTeam"]).strip(),
             str(r["AwayTeam"]).strip()) for _, r in df.iterrows()]

    p_market = np.full((len(keys), 3), np.nan)
    has_market = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        o = odds_by_key.get(k, {}).get(args.market)
        if not o:
            continue
        p = devig(o)
        if p is None:
            continue
        p_market[i] = p
        has_market[i] = True

    print("=" * 100)
    print(f"  SHRINKAGE SUL MERCATO — {league.name} — prior: {args.market}")
    print("=" * 100)
    print(f"  righe di test: {len(keys)} | con prior di mercato: {int(has_market.sum())}")

    idx = np.where(has_market)[0]
    if len(idx) < 50:
        print("  Troppe poche righe con prior: skip.")
        return

    y = np.array([y_test[i] for i in idx])
    pm = probs[idx]
    pk = p_market[idx]

    print(f"\n  {'─'*100}\n  Riferimenti (sulle stesse {len(idx)} righe)\n  {'─'*100}")
    print(fmt("modello (calibrato)", metrics(y.tolist(), pm.tolist())))
    print(fmt(f"mercato [{args.market}]", metrics(y.tolist(), pk.tolist())))

    # ── Fit del peso w su VALIDAZIONE out-of-sample ──────────────────────
    # La validazione e' la seconda meta' del TRAINING: il modello e' gia'
    # addestrato anche su quelle righe, quindi non sono out-of-sample per il
    # MODELLO. Per evitare il leakage si usa la tecnica del K-FOLD: si
    # allena un modello ausiliario su una meta' dei dati e si valuta la
    # shrinkage sull'altra meta'. Qui, per non riaddestrare, si usa un
    # criterio稳健 sul LOG-LOSS: il peso ottimo e' quello che minimizza il
    # log-loss, e si sceglie fra i valori misurati sul test SOLO come
    # VERIFICA (il valore scelto va riportato con la sua validazione).
    #
    # Approccio onesto e semplice: si sceglie w che minimizza il log-loss
    # sulle righe di VALIDAZIONE del training che il modello AUSILIARIO non
    # ha visto (stessa tecnica cross-fitted della calibrazione).
    print(f"\n  {'─'*100}\n  Scansione del peso w (w = peso del modello)\n  {'─'*100}")
    best = None
    for w in np.arange(0.0, 1.01, 0.05):
        m = metrics(y.tolist(), blend(pm, pk, float(w)).tolist())
        marker = ""
        if best is None or m["log_loss"] < best[1]["log_loss"]:
            best = (float(w), m)
            marker = "  <- migliore (test)"
        if abs(w * 20 - round(w * 20)) < 1e-9 and round(w * 100) % 10 == 0:
            print(fmt(f"w={w:.2f}", m) + marker)
    w_best, m_best = best
    print(f"\n  -> w ottimale sul TEST set: {w_best:.2f}  "
          f"(RPS {m_best['rps']:.5f}, acc {m_best['accuracy']:.2%})")
    print("     NOTA: questo w e' ottimizzato sul test set, quindi e' un TETTO")
    print("     superiore, non una stima onesta. Il valore da usare va")
    print("     scelto su validazione out-of-sample (vedi sotto).")

    # ── Verifica con peso scelto su validazione cross-fitted ─────────────
    print(f"\n  {'─'*100}\n  Peso w scelto su VALIDAZIONE cross-fitted (onesto)\n  {'─'*100}")
    w_val = cross_fitted_weight(league, args.market, devig, odds_by_key, args.cutoff)
    if w_val is not None:
        m_val = metrics(y.tolist(), blend(pm, pk, w_val).tolist())
        print(fmt(f"w={w_val:.3f} (da validazione)", m_val))
        m_mkt = metrics(y.tolist(), pk.tolist())
        m_mod = metrics(y.tolist(), pm.tolist())
        print(f"\n  -> RPS:    modello {m_mod['rps']:.5f} | mercato {m_mkt['rps']:.5f} | "
              f"blend(w={w_val:.2f}) -> {m_val['rps']:.5f}")
        print(f"  -> log-loss: modello {m_mod['log_loss']:.5f} | mercato "
              f"{m_mkt['log_loss']:.5f} | blend -> {m_val['log_loss']:.5f}")
        print(f"  -> accuracy: modello {m_mod['accuracy']:.2%} | mercato "
              f"{m_mkt['accuracy']:.2%} | blend -> {m_val['accuracy']:.2%}")
        if m_val["rps"] < m_mkt["rps"]:
            print(f"  -> IL BLEND BATTE il mercato (RPS {m_mkt['rps'] - m_val['rps']:+.5f})")
        else:
            print(f"  -> il blend NON batte il mercato (delta {m_val['rps'] - m_mkt['rps']:+.5f})")

        # ── Stabilita' del peso w: e' il vero test diAffidabilita' ────────
        # Se w* cambia molto fra validazione e test, il peso ottimale non
        # e' un parametro stabile e la shrinkage con peso fisso NON e' una
        # strategia affidabile: e' overfitting su un set piccolo.
        print("\n  Scansione w (test set) — per capire la STABILITA' del peso:")
        w_test_best = None
        for w in np.arange(0.0, 1.01, 0.05):
            m = metrics(y.tolist(), blend(pm, pk, float(w)).tolist())
            if w_test_best is None or m["log_loss"] < w_test_best[1]["log_loss"]:
                w_test_best = (float(w), m["log_loss"])
        print(f"     w* validazione = {w_val:.2f} | w* test = {w_test_best[0]:.2f}"
              f"  -> scarto {abs(w_val - w_test_best[0]):.2f}")
        if abs(w_val - w_test_best[0]) > 0.2:
            print("     -> peso INSTABILE: la shrinkage con peso fisso non e' "
                  "affidabile (overfitting del peso).")

        out = Path(args.out) if args.out else (ROOT / "models" / f"market_blend_{league.key}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"market": args.market, "w": float(w_val),
                       "test": m_val, "market_only": m_mkt, "model_only": m_mod,
                       "w_optimal_on_test": w_best}, f, indent=2, default=float)
        print(f"\n  Report: {out}")


def cross_fitted_weight(league, market, devig, odds_by_key, cutoff):
    """Sceglie w su validazione OUT-OF-SAMPLE: modello ausiliario allenato
    sulla seconda meta' del training, shrinkage valutata sulla prima meta'."""
    import pandas as pd
    from catboost import CatBoostClassifier
    from features import FEATURE_COLS

    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    df = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam", "season"]].copy()
    df = df.dropna(subset=FEATURE_COLS + ["FTR"])
    df = df[df["Date"] < pd.Timestamp(cutoff)].reset_index(drop=True)

    X = pd.read_csv(league.split_path("X_train_v2"))
    y = pd.read_csv(league.split_path("y_train_v2")).squeeze().astype(int).tolist()
    n_val = len(y) // 2

    # modello ausiliario: allenato solo sulla seconda meta'
    aux = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                             loss_function="MultiClass", eval_metric="Accuracy",
                             random_seed=42, verbose=False)
    aux.fit(X.iloc[n_val:], list(y[n_val:]))
    p_aux = aux.predict_proba(X.iloc[:n_val])

    keys = [(r["Date"].strftime("%Y-%m-%d"), str(r["HomeTeam"]).strip(),
             str(r["AwayTeam"]).strip()) for _, r in df.iterrows()]
    yv, pv, kv = [], [], []
    for i in range(n_val):
        o = odds_by_key.get(keys[i], {}).get(market)
        if not o:
            continue
        p = devig(o)
        if p is None:
            continue
        yv.append(y[i]); pv.append(p_aux[i]); kv.append(p)
    if len(yv) < 100:
        return None
    yv = np.array(yv); pv = np.array(pv); kv = np.array(kv)

    best = None
    for w in np.arange(0.0, 1.01, 0.02):
        ll = log_loss(yv.tolist(), blend(pv, kv, float(w)).tolist())
        if best is None or ll < best[1]:
            best = (float(w), ll)
    print(f"     validazione cross-fitted: n={len(yv)}  w*= {best[0]:.2f}  log-loss {best[1]:.5f}")
    return best[0]


if __name__ == "__main__":
    main()
