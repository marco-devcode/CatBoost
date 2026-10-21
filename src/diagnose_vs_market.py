#!/usr/bin/env python3
"""
Diagnosi: il modello ha edge SUL MERCATO, o lo copia soltanto?

Tre domande, tre misure.

1. DISACCORDO: quando il modello e il bookmaker dicono esiti diversi, chi
   ha ragione? Se il modello sbaglia piu' spesso del mercato, allora il
   modello non ha edge: sta solo imitando il mercato. Misura: sulle partite
   in disaccordo, quante volte ha ragione il modello (su 1, bet365 ha
   ragione 0 o viceversa).

2. VALORE (EV): un modello e' utile per scommettere se produce un valore
   atteso POSITIVO alle quote OFFERTE. Il bookmaker ha un margine (~5.7% per
   bet365, ~2.6% per Pinnacle): quindi serve p_model > 1/odds, non
   p_model > p_bookmaker. Misura: ROI a stake fisso e ROI Kelly, e la
   ripartizione per fascia di quota (dove il favourite-longshot bias
   dovrebbe concentrarsi il margine).

3. CLOSING LINE VALUE: se si scommette alle quote di APERTURA e si
   confronta con la CHIUSURA (il riferimento sharp), il CLV dice quanto
   della nostra edge dipende dall'informazione (chi si muove bene sul
   mercato haCLV alto) e quanto e' rumore. Misura: differenza media fra
   probabilita' implicite di apertura e di chiusura sulle partite su cui
   abbiamo puntato.

Uso:
    python src/diagnose_vs_market.py --league seriea
    python src/diagnose_vs_market.py --league seriea --season 2627
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

from implied import DE_VIG_METHODS, log_loss, ranked_probability_score  # noqa: E402

FTR_IDX = {"H": 0, "D": 1, "A": 2}
LABELS = ["H", "D", "A"]
DEVIG = "additive"


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
            books = {
                "b365_open": (_num(row, "B365H"), _num(row, "B365D"), _num(row, "B365A")),
                "b365_close": (_num(row, "B365CH"), _num(row, "B365CD"), _num(row, "B365CA")),
                "pinnacle_open": (_num(row, "PSH"), _num(row, "PSD"), _num(row, "PSA")),
                "pinnacle_close": (_num(row, "PSCH"), _num(row, "PSCD"), _num(row, "PSCA")),
            }
            for name, o in books.items():
                if all(o):
                    out[key][name] = o
    return dict(out)


def test_rows(league, cutoff: str, season: str | None = None):
    """Righe di test con le loro chiavi, nello stesso ordine di X_test_v2."""
    import pandas as pd
    from features import FEATURE_COLS

    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    df = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam", "season"]].copy()
    df = df.dropna(subset=FEATURE_COLS + ["FTR"])
    df = df[df["Date"] >= pd.Timestamp(cutoff)]
    if season:
        df = df[df["season"].astype(str) == str(season)]
    return [
        {
            "key": (r["Date"].strftime("%Y-%m-%d"), str(r["HomeTeam"]).strip(), str(r["AwayTeam"]).strip()),
            "y": FTR_IDX[str(r["FTR"]).strip().upper()],
            "season": str(r["season"]),
        }
        for _, r in df.iterrows()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--cutoff", default="2025-08-01")
    ap.add_argument("--season", default=None, help="filtra una stagione (es. 2627)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from leagues import get_league
    league = get_league(args.league)

    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(league.model_path))
    feature_names = list(model.feature_names_)

    import pandas as pd
    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze()
    probs = model.predict_proba(X_test)
    # Probabilita' CALIBRATE: sono quelle che il sito pubblica.
    from predict import apply_saved_calibration
    probs = np.array([apply_saved_calibration(league, p) for p in probs])

    rows = test_rows(league, args.cutoff, args.season)
    if not rows:
        print("Nessuna riga di test.")
        return

    # Le probabilita' del modello sono nel ORDINE di X_test_v2; le righe
    # filtrate per stagione vanno ri-allineate: si prende l'indice di ogni
    # riga nel dataset completo.
    import pandas as pd
    from features import FEATURE_COLS
    feats = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    df = feats[FEATURE_COLS + ["Date", "FTR", "HomeTeam", "AwayTeam", "season"]].copy()
    df = df.dropna(subset=FEATURE_COLS + ["FTR"])
    df = df[df["Date"] >= pd.Timestamp(args.cutoff)].reset_index(drop=True)
    mask = (df["season"].astype(str) == str(args.season)).to_numpy() if args.season else np.ones(len(df), bool)
    idx = np.where(mask)[0]
    rows = [rows[i] for i in range(len(rows)) if i < len(idx)]

    odds_by_key = load_odds(league.raw_path)
    devig = DE_VIG_METHODS[DEVIG]

    print("=" * 92)
    print(f"  DIAGNOSI vs MERCATO — {league.name}"
          + (f" — stagione {args.season}" if args.season else " (tutte le righe di test)"))
    print("=" * 92)
    print(f"  Modello: {len(idx)} partite | de-vig: {DEVIG}")

    # ── 1. DISACCORDO ─────────────────────────────────────────────────────
    print(f"\n{'─'*92}\n  1. DISACCORDO modello vs bookmaker\n{'─'*92}")
    for book in ("b365_open", "b365_close", "pinnacle_close"):
        agree = 0
        model_right = 0
        bm_right = 0
        neither = 0
        n = 0
        # Contatori SOLO sulle partite in disaccordo (la domanda e':
        # "quando i due divergono, chi ha ragione?").
        dis_model = 0
        dis_bm = 0
        dis_neither = 0
        for i, r in zip(idx, rows):
            o = odds_by_key.get(r["key"], {}).get(book)
            if not o:
                continue
            bp = devig(o)
            if bp is None:
                continue
            n += 1
            pm_pred = int(np.argmax(probs[i]))
            bm_pred = int(np.argmax(bp))
            if pm_pred == bm_pred:
                agree += 1
                if pm_pred == r["y"]:
                    model_right += 1
            else:
                if pm_pred == r["y"]:
                    model_right += 1
                    dis_model += 1
                elif bm_pred == r["y"]:
                    bm_right += 1
                    dis_bm += 1
                else:
                    neither += 1
                    dis_neither += 1
        if n:
            print(f"  {book:<16} n={n:>4}  accordo={agree/n:>6.1%}  "
                  f"modello corretto={model_right/n:>6.1%}  bm corretto={bm_right/n:>6.1%}  "
                  f"nessuno={neither/n:>5.1%}")
            # Sulle partite in disaccordo, il modello ha edge?
            dis = n - agree
            if dis:
                print(f"      -> sulle {dis} partite in DISACCORDO: modello corretto "
                      f"{dis_model} ({dis_model/dis:.1%}) vs bookmaker {dis_bm} "
                      f"({dis_bm/dis:.1%}), nessuno dei due {dis_neither}")

    # ── 2. VALORE (EV) alle quote offerte ─────────────────────────────────
    print(f"\n{'─'*92}\n  2. VALORE: EV e ROI alle quote OFFERTE (bet365 apertura)\n{'─'*92}")
    for book in ("b365_open", "b365_close"):
        for edge_min in (0.0, 0.02, 0.05):
            n_bets = 0
            wins = 0
            stake = 0.0
            profit = 0.0
            by_bucket = defaultdict(lambda: [0, 0.0])
            for i, r in zip(idx, rows):
                o = odds_by_key.get(r["key"], {}).get(book)
                if not o:
                    continue
                p_mkt = devig(o)
                if p_mkt is None:
                    continue
                pm = probs[i]
                # EV per esito: p*(odds-1) - (1-p)
                evs = [pm[k] * (o[k] - 1.0) - (1.0 - pm[k]) for k in range(3)]
                best = int(np.argmax(evs))
                if evs[best] <= edge_min:
                    continue
                # stake fisso 1 unita'
                n_bets += 1
                stake += 1.0
                won = (best == r["y"])
                pnl = (o[best] - 1.0) if won else -1.0
                profit += pnl
                wins += int(won)
                bucket = ("1.0-1.8" if o[best] < 1.8 else
                          "1.8-2.5" if o[best] < 2.5 else
                          "2.5-4.0" if o[best] < 4.0 else "4.0+")
                by_bucket[bucket][0] += 1
                by_bucket[bucket][1] += pnl
            if n_bets:
                roi = profit / stake * 100
                print(f"  {book:<12} edge>{edge_min:.0%}  scommesse={n_bets:>4}  "
                      f"win={wins/n_bets:>6.1%}  ROI={roi:>+7.2f}%  "
                      f"profitto={profit:>+8.2f}u")
                if edge_min == 0.0:
                    print(f"      per fascia di quota: " + "  ".join(
                        f"{b}: {v[0]}s {v[1]:+.1f}u" for b, v in sorted(by_bucket.items())))

    # ── 2b. EDGE STRUTTURALE: bookmaker "soft" vs "sharp" ────────────────
    # bet365 ha overround medio 5.67%, Pinnacle 2.61% (misurato su 6135
    # partite). Pinnacle e' il "sharp": il suo margine e' vicino al costo
    # operativo, quindi la sua probabilita' de-viggata e' una stima vicina
    # alla verita'. bet365 e' "soft": applica un margine maggiore.
    #
    # CONSEGUENZA OPERATIVA: la stessa partita costa di piu' su bet365 che su
    # Pinnacle. Se si conosce la stima sharp (Pinnacle) e si scommette
    # all'offerto di bet365, si ha un edge POSITIVO ogni volta che la
    # probabilita' sharp supera l'implicita lordata di bet365.
    #
    # Questo e' l'edge documentato del "line shopping" e del
    # favourite-longshot bias: il margine non e' uniforme, e il bookmaker
    # soft lo carica di piu' sui longshot.
    print(f"\n{'─'*92}\n  2b. EDGE STRUTTURALE soft (bet365) vs sharp (Pinnacle)\n{'─'*92}")
    sharp_books = [("pinnacle_open", "b365_open"), ("pinnacle_close", "b365_close")]
    for sharp_book, soft_book in sharp_books:
        n_bets = 0
        stake = 0.0
        profit = 0.0
        wins = 0
        by_bucket = defaultdict(lambda: [0, 0.0])
        for i, r in zip(idx, rows):
            ko = odds_by_key.get(r["key"], {}).get(sharp_book)
            ks = odds_by_key.get(r["key"], {}).get(soft_book)
            if not ko or not ks:
                continue
            p_sharp = devig(ko)
            if p_sharp is None:
                continue
            # implied lordata di bet365: 1/odds (include il margine)
            imp_soft = [1.0 / ks[k] for k in range(3)]
            # edge = p_sharp - implied_bet365 (la soglia di value betting)
            edges = [p_sharp[k] - imp_soft[k] for k in range(3)]
            best = int(np.argmax(edges))
            if edges[best] <= 0.0:
                continue
            n_bets += 1
            stake += 1.0
            won = (best == r["y"])
            pnl = (ks[best] - 1.0) if won else -1.0
            profit += pnl
            wins += int(won)
            bucket = ("1.0-1.8" if ks[best] < 1.8 else
                      "1.8-2.5" if ks[best] < 2.5 else
                      "2.5-4.0" if ks[best] < 4.0 else "4.0+")
            by_bucket[bucket][0] += 1
            by_bucket[bucket][1] += pnl
        if n_bets:
            print(f"  sharp={sharp_book:<15} vs soft={soft_book:<12} "
                  f"scommesse={n_bets:>4}  win={wins/n_bets:>6.1%}  "
                  f"ROI={profit/stake*100:>+7.2f}%  profitto={profit:>+8.2f}u")
            print("      per fascia di quota: " + "  ".join(
                f"{b}: {v[0]}s {v[1]:+.1f}u" for b, v in sorted(by_bucket.items())))
        else:
            print(f"  sharp={sharp_book:<15} vs soft={soft_book:<12} nessuna scommessa con edge")

    # ── 3. CLOSING LINE VALUE ─────────────────────────────────────────────
    print(f"\n{'─'*92}\n  3. CLOSING LINE VALUE (apertura -> chiusura)\n{'─'*92}")
    d_close_minus_open = []
    for i, r in zip(idx, rows):
        ko = odds_by_key.get(r["key"], {}).get("b365_open")
        kc = odds_by_key.get(r["key"], {}).get("b365_close")
        if not ko or not kc:
            continue
        po, pc = devig(ko), devig(kc)
        if po is None or pc is None:
            continue
        # quanto e' cambiata la probabilita' implicita di chiusura
        d_close_minus_open.append(max(abs(pc[k] - po[k]) for k in range(3)))
    if d_close_minus_open:
        arr = np.array(d_close_minus_open)
        print(f"  n={len(arr)}  spostamento medio |Δp| apertura->chiusura = {arr.mean():.4f} "
              f"(mediana {np.median(arr):.4f}, max {arr.max():.4f})")
        print("  -> il mercato muove le probabilita' implicite in media di "
              f"{arr.mean():.3f}: un modello che prevede il movimento di chiusura "
              "ha piu' probabilita' di battere la CHIUSURA (non l'apertura).")

    # ── 4. CLOSING LINE VALUE: il modello prevede il movimento del mercato? ─
    # Test decisivo per "battere il bookmaker" in modo OPERATIVO:
    # se il modello contiene informazione che il mercato di APERTURA non ha,
    # allora le sue previsioni si muovono nella stessa direzione della
    # chiusura PIU' di quanto faccia il mercato di apertura. Questo e' il CLV
    # (closing line value): il parametro con cui i professionisti valutano
    # una strategia, perche' il closing line e' il benchmark da battere.
    #
    # Metrica: su ogni partita, quanto si muove la probabilita' implicita di
    # chiusura rispetto all'apertura (per l'esito che il modello indica come
    # piu' probabile)? Se il modello e' un buon segnale, il suo esito
    # preferito dovrebbe essere quello che il mercato RAZZA (la quota
    # dell'esito scende = il mercato si e' mosso nella sua direzione).
    print(f"\n{'─'*92}\n  4. CLOSING LINE VALUE: il modello prevede il movimento?\n{'─'*92}")
    n_clv = 0
    n_follow = 0
    n_follow_mkt = 0
    for i, r in zip(idx, rows):
        ko = odds_by_key.get(r["key"], {}).get("b365_open")
        kc = odds_by_key.get(r["key"], {}).get("b365_close")
        if not ko or not kc:
            continue
        po, pc = devig(ko), devig(kc)
        if po is None or pc is None:
            continue
        n_clv += 1
        # esito preferito dal modello (sulla versione calibrata)
        k_model = int(np.argmax(probs[i]))
        # esito preferito dal mercato di apertura
        k_mkt = int(np.argmax(po))
        # "il mercato segue il modello": la quota di quell'esito SCENDE fra
        # apertura e chiusura (il mercato si e' mosso verso di lui)
        d_model = po[k_model] - pc[k_model]
        d_mkt = po[k_mkt] - pc[k_mkt]
        if d_model > 0:
            n_follow += 1
        if d_mkt > 0:
            n_follow_mkt += 1
    if n_clv:
        print(f"  n={n_clv}")
        print(f"  esito preferito dal MODELLO seguito dal mercato (quota in calo): "
              f"{n_follow}/{n_clv} = {n_follow/n_clv:.1%}")
        print(f"  esito preferito dal MERCATO seguito dal mercato:              "
              f"{n_follow_mkt}/{n_clv} = {n_follow_mkt/n_clv:.1%}")
        if n_follow > n_follow_mkt:
            print("  -> il modello anticipa il movimento del mercato meglio del "
                  "mercato stesso: ha informazione propria.")
        else:
            print("  -> il modello NON anticipa il movimento del mercato piu' "
                  "di quanto faccia il mercato: non c'e' informazione propria.")

    out = Path(args.out) if args.out else (ROOT / "models" / f"diagnose_market_{league.key}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_test": int(len(idx)), "season": args.season, "devig": DEVIG}, f, indent=2)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
