#!/usr/bin/env python3
"""
SEZIONE XGBOOST: modello alternativo, feature nuove, confronto col mercato.

Il percorso, in ordine:
  1. CATALOGO: cosa c'e' nel repo e cosa non si stava usando.
  2. COSTRUZIONE DEL DATASET: si RIPRODUCE la procedura esatta della
     pipeline di produzione (src/features.py: build_feature_matrix), non
     un taglio posizionale. E' il punto in cui il primo tentativo era
     sbagliato: tagliando le ultime N righe si perde l'allineamento
     fra righe ed etichette, e i risultati diventano rumore.
  3. xG DA UNDERSTAT: si unisce per CHIAVE (coppia di squadre + data con
     tolleranza di 2 giorni, perche' Understat sposta alcune date di un
     giorno) e da lì si ricostruiscono gli aggregati per squadra con
     shift(1): ogni feature guarda solo le partite PRECEDENTI.
  4. MERCATO DA ALTRI BOOKMAKER: quote di APERTURA de-viggate, line
     over/under 2.5, Asian Handicap. Le quote di chiusura NON si usano
     (hindsight), e ne' HTHG/HTAG (gol del primo tempo: noti solo a fine
     partita).
  5. MODELLI: XGBoost con tuning su CV TEMPORALE, CatBoost di produzione
     con gli stessi parametri, e cinque set di feature a confronto.
  6. IMPORTANZA E SIGNIFICATIVITA': gain di XGBoost + drop-column test
     (quanto peggiora la RPS togliendo la feature) con test paired.
  7. REFIT e CONFRONTO FINALE contro bet365 e contro CatBoost, con gli
     stessi test del resto del progetto: bootstrap paired, Wilcoxon,
     Diebold-Mariano.

Uso:  python src/xgb_experiment.py --league seriea --tune 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
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
from select_features import (  # noqa: E402
    dm_test,
    paired_bootstrap,
    paired_wilcoxon,
    rps_per_match,
)

DEVIG = DE_VIG_METHODS["additive"]
TOLERANCE_DAYS = 2
CUTOFF = "2025-08-01"

# Quote di APERTURA di altri bookmaker (mai di chiusura: hindsight)
OTHER_BOOKS = {
    "betfair": ("BWH", "BWD", "BWA"),
    "williamhill": ("WHH", "WHD", "WHA"),
    "interwetten": ("IWH", "IWD", "IWA"),
    "pinnacle": ("PSH", "PSD", "PSA"),
    "blackbet": ("BbAvH", "BbAvD", "BbAvA"),
}
# Il football-data fornisce solo la quota OVER 2.5 (non il lato UNDER), quindi
# la probabilita' non si puo' de-viggare: si usa la LORDA (il bookmaker
# sovrastima l'Over, quindi e' una stima conservativa del totale atteso).
OU_LINES = {"b365_ou25": ("B365>2.5", None)}
AH_LINES = {"b365_ahh": ("B365AHH", "B365AHA"), "avg_ahh": ("AvgAHH", "AvgAHA")}

# ── DEPLOYABILITÀ ────────────────────────────────────────────────────────
# Una feature e' "deployabile" solo se le quote che la generano sono
# disponibili per una partita future PRIMA del fischio finale. In produzione
# l'API delle quote restituisce UN bookmaker (bet365): quindi
#   - mkt_b365_*            -> deployabile
#   - mkt_betfair/williamhill/interwetten/blackbet/pinnacle -> richiedono un
#     abbonamento multi-book (line shopping): disponibili, ma non gratis
#   - mkt_avg_* (consensus) -> richiede piu' bookmaker: non deployabile
#   - *_overround (tranne b365) -> non deployabile
# L'xG di Understat e' deployabile: gli aggregati rolling usano solo le
# partite PRECEDENTI, che sono note da ore.
XGB_NJOBS = int(os.environ.get("XGB_NJOBS", "1"))
XG_PREFIXES = ("xG_", "xGA_", "npxG_", "npxGA_", "ppda_", "ppda_allowed_",
               "deep_", "deep_allowed_", "xpts_", "xpts_against_")
NON_DEPLOYABLE_PREFIX = ("mkt_avg_", "mkt_pinnacle_overround", "mkt_avg_overround")
BOOK_PREFIX = ("mkt_betfair_", "mkt_williamhill_", "mkt_interwetten_",
               "mkt_blackbet_", "mkt_pinnacle_")
B365_EXT = ("mkt_b365_ou25_p_over", "mkt_b365_ahh_line", "mkt_b365_ahh_cover")

# colonne da NON usare mai: sono note solo DOPO la partita
FORBIDDEN = ("FTHG", "FTAG", "FTR", "HST", "AST", "HS", "AS", "HC", "AC", "HY",
             "AY", "HTHG", "HTAG", "HTR", "B365CH", "B365CD", "B365CA", "PSCH",
             "PSCD", "PSCA", "VCH", "VCD", "VCA", "MaxCH", "MaxCD", "MaxCA",
             "AvgCH", "AvgCD", "AvgCA", "B365CH.1")


def nrm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def _num(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def devig3(h, d, a):
    try:
        h, d, a = float(h), float(d), float(a)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(x) and x > 1.0 for x in (h, d, a)):
        return None
    return DEVIG((h, d, a))


def build_market_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Quote di apertura di altri bookmaker, line over/under, Asian Handicap."""
    out = pd.DataFrame(index=raw.index)
    for label, cols in OTHER_BOOKS.items():
        if not all(c in raw.columns for c in cols):
            continue
        arr = np.column_stack([raw[c].map(_num).to_numpy(dtype=float) for c in cols])
        ok = np.all(np.isfinite(arr), axis=1) & np.all(arr > 1.0, axis=1)
        for j, side in enumerate(("H", "D", "A")):
            p = np.full(len(raw), np.nan)
            for i in np.where(ok)[0]:
                pr = devig3(*arr[i])
                if pr is not None:
                    p[i] = pr[j]
            out[f"mkt_{label}_p{side}"] = p
    for label, (c_over, c_under) in OU_LINES.items():
        if c_over not in raw.columns:
            continue
        o = raw[c_over].map(_num)
        if c_under is None:
            out[f"mkt_{label}_p_over"] = np.where(o > 1.0, 1.0 / o, np.nan)
            continue
        if c_under not in raw.columns:
            continue
        u = raw[c_under].map(_num)
        both = o.notna() & u.notna() & (o > 1.0) & (u > 1.0)
        out[f"mkt_{label}_p_over"] = np.where(
            both, (1.0 / o) / ((1.0 / o) + (1.0 / u)), np.nan)
    for label, cols in AH_LINES.items():
        if not all(c in raw.columns for c in cols):
            continue
        home = raw[cols[0]].map(_num)
        away = raw[cols[1]].map(_num)
        both = home.notna() & away.notna() & (home > 1.0) & (away > 1.0)
        out[f"mkt_{label}_line"] = home - away
        out[f"mkt_{label}_cover"] = np.where(
            both, (1.0 / home) / ((1.0 / home) + (1.0 / away)), np.nan)
    return out


def build_xg_features(df: pd.DataFrame, xg_by_row: dict) -> pd.DataFrame:
    """Aggregati xG per squadra, calcolati sul dataset e non su date approssimate.

    `xg_by_row` mappa l'indice del dataset alla riga Understat corrispondente
    (o None). Da lì, per ogni squadra, si prende la serie delle sue partite
    ORDINATE PER DATA e si calcolano media su 5 partite e media di stagione
    con shift(1): la riga corrente non entra mai nei propri aggregati.
    """
    recs = []
    for i, r in df.iterrows():
        u = xg_by_row.get(i)
        if u is None:
            continue
        recs.append({"i": i, "date": r["Date"], "home": nrm(r["HomeTeam"]),
                     "away": nrm(r["AwayTeam"]), "xG_h": u["xG_home"],
                     "xG_a": u["xG_away"], "npxG_h": u.get("npxG_home", np.nan),
                     "npxG_a": u.get("npxG_away", np.nan),
                     "ppda_h": u.get("ppda_home", np.nan),
                     "ppda_a": u.get("ppda_away", np.nan),
                     "deep_h": u.get("deep_home", np.nan),
                     "deep_a": u.get("deep_away", np.nan),
                     "xpts_h": u.get("xpts_home", np.nan),
                     "xpts_a": u.get("xpts_away", np.nan)})
    per_team = []
    for rec in recs:
        per_team.append({"team": rec["home"], "date": rec["date"], "i": rec["i"],
                         "xG": rec["xG_h"], "xGA": rec["xG_a"], "npxG": rec["npxG_h"],
                         "npxGA": rec["npxG_a"], "ppda": rec["ppda_h"],
                         "ppda_allowed": rec["ppda_a"], "deep": rec["deep_h"],
                         "deep_allowed": rec["deep_a"], "side": "home",
                         "xpts": rec.get("xpts_h", np.nan),
                         "xpts_against": rec.get("xpts_a", np.nan)})
        per_team.append({"team": rec["away"], "date": rec["date"], "i": rec["i"],
                         "xG": rec["xG_a"], "xGA": rec["xG_h"], "npxG": rec["npxG_a"],
                         "npxGA": rec["npxG_h"], "ppda": rec["ppda_a"],
                         "ppda_allowed": rec["ppda_h"], "deep": rec["deep_a"],
                         "deep_allowed": rec["deep_h"], "side": "away",
                         "xpts": rec.get("xpts_a", np.nan),
                         "xpts_against": rec.get("xpts_h", np.nan)})
    pt = pd.DataFrame(per_team).sort_values(["team", "date"])
    g = pt.groupby("team", sort=False)
    stat_cols = ["xG", "xGA", "npxG", "npxGA", "ppda", "ppda_allowed", "deep",
                 "deep_allowed", "xpts", "xpts_against"]
    for c in stat_cols:
        pt[f"{c}_5"] = g[c].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        pt[f"{c}_season"] = g[c].transform(lambda x: x.shift(1).expanding(min_periods=2).mean())

    home_idx = pt[pt["side"] == "home"].set_index("i")
    away_idx = pt[pt["side"] == "away"].set_index("i")
    out = pd.DataFrame(index=df.index)
    for c in stat_cols:
        for suf in ("_5", "_season"):
            col = f"{c}{suf}"
            out[f"{col}_home"] = home_idx[col].reindex(df.index)
            out[f"{col}_away"] = away_idx[col].reindex(df.index)
    return out


def _tune(sampler, cv_rps, make_model, cols, mask, args, xgb_params=None):
    """Ricerca randomizzata degli iperparametri su CV temporale."""
    best = None
    t0 = time.time()
    for k, p in enumerate(sampler, 1):
        try:
            s = cv_rps(lambda: make_model(p), cols, mask)
        except Exception as e:  # noqa: BLE001
            print(f"    cfg {k}: fallita ({str(e)[:80]})")
            continue
        if best is None or s < best[0]:
            best = (s, p)
            print(f"    cfg {k:>2}: RPS OOF {s:.5f}  <- migliore")
        if k % 8 == 0:
            print(f"    ...{k}/{len(sampler)} ({time.time()-t0:.0f}s)")
    if best is not None:
        print(f"\n  → migliore XGBoost: RPS OOF {best[0]:.5f}")
        print(f"    {best[1]}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--tune", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force-tune", action="store_true",
                    help="rifare il tuning anche se esiste il checkpoint")
    args = ap.parse_args()
    from leagues import get_league
    league = get_league(args.league)

    print("=" * 104)
    print(f"  SEZIONE XGBOOST — {league.name}")
    print("=" * 104)

    # ── 1. DATASET: si RIPRODUCE la procedura di produzione ────────────
    from features import FEATURE_COLS, build_feature_list
    feats_all = pd.read_csv(league.features_path, parse_dates=["Date"], low_memory=False)
    raw = pd.read_csv(league.raw_path, low_memory=False)

    feature_list = build_feature_list(feats_all.columns)
    # HomeTeam/AwayTeam servono per il join per chiave con Understat e con le
    # quote degli altri bookmaker: vanno tenuti, ma NON sono feature.
    df = feats_all[feature_list + ["Date", "FTR", "HomeTeam", "AwayTeam"]].copy()
    base_required = FEATURE_COLS + ["FTR"]
    df = df.dropna(subset=base_required)
    df_train = df[df["Date"] < CUTOFF].copy()
    df_test = df[df["Date"] >= CUTOFF].copy()
    print(f"  [split temporale] cutoff={CUTOFF} -> train={len(df_train)}, test={len(df_test)}")
    print(f"  (la pipeline di produzione usa gli stessi file: "
          f"X_train_v2 {pd.read_csv(league.split_path('X_train_v2')).shape})")

    # Le etichette si prendono dai file UFFICIALI (y_train_v2 / y_test_v2):
    # il .map() sul FTR puo' produrre NaN se il CSV ha valori inattesi, e un
    # NaN nell'etichetta fa fallire XGBoost con
    # "Check failed: sum_weight >= kRtEps".
    y = pd.read_csv(league.split_path("y_train_v2")).squeeze().astype(int).to_numpy()
    yte = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).to_numpy()
    assert len(y) == len(df_train) and len(yte) == len(df_test), \
        "etichette e righe non allineate"
    assert not np.isnan(y).any() and not np.isnan(yte).any(), "etichette NaN"
    # verifica di coerenza: l'etichetta ufficiale deve corrispondere al FTR
    ftr_map = df_train["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    assert np.array_equal(ftr_map.astype(int), y), "FTR e y_train_v2 discordi"
    ftr_map_te = df_test["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    assert np.array_equal(ftr_map_te.astype(int), yte), "FTR e y_test_v2 discordi"

    # ── 2. xG DA UNDERSTAT (join per chiave con tolleranza) ────────────
    us_path = ROOT / "data" / f"understat_{league.key}.csv"
    xg_by_row = {}
    if us_path.exists():
        us = pd.read_csv(us_path)
        us["date"] = pd.to_datetime(us["date"], errors="coerce")
        by_pair = {}
        for i, r in df.iterrows():
            by_pair.setdefault((nrm(r["HomeTeam"]), nrm(r["AwayTeam"])), []).append(
                (r["Date"], i))
        hits = 0
        for _, u in us.iterrows():
            key = (nrm(u["home"]), nrm(u["away"]))
            cands = by_pair.get(key)
            if not cands:
                continue
            best = min(cands, key=lambda t: abs((t[0] - u["date"]).days))
            if abs((best[0] - u["date"]).days) <= TOLERANCE_DAYS:
                xg_by_row[best[1]] = u.to_dict()
                hits += 1
        print(f"  understat: {len(us)} partite scaricate | join per chiave: "
              f"{len(xg_by_row)}/{len(df)} righe ({len(xg_by_row)/len(df)*100:.1f}%)")
        xg = build_xg_features(df, xg_by_row)
        for c in xg.columns:
            df[c] = xg[c].values
        print(f"  feature xG: {xg.shape[1]}")
    else:
        print("  ATTENZIONE: dati understat assenti, si continua senza xG")

    # ── 3. MERCATO DA ALTRI BOOKMAKER (join per chiave) ────────────────
    # Nel raw le date sono in formato gg/mm/aa: vanno normalizzate in ISO,
    # altrimenti il join per chiave con il dataset (che usa ISO) non
    # abbocca e TUTTE le feature degli altri bookmaker arrivano NaN.
    raw_dates = pd.to_datetime(raw["Date"], format="mixed", dayfirst=True)
    raw["_key"] = (raw_dates.dt.strftime("%Y-%m-%d") + "|" +
                   raw["HomeTeam"].astype(str).str.strip() + "|" +
                   raw["AwayTeam"].astype(str).str.strip())
    mkt = build_market_features(raw)
    df["_key"] = (df["Date"].dt.strftime("%Y-%m-%d") + "|" +
                  df["HomeTeam"].astype(str).str.strip() + "|" +
                  df["AwayTeam"].astype(str).str.strip())
    keymap = dict(zip(raw["_key"], mkt.index))
    for c in mkt.columns:
        df[c] = [mkt.at[keymap[k], c] if k in keymap else np.nan for k in df["_key"]]
    print(f"  feature di mercato da altri bookmaker: {len(mkt.columns)}")

    # ── 4. ASSEMBLAGGIO E PULIZIA ──────────────────────────────────────
    # Le feature di PRODUZIONE sono esattamente quelle del modello semplificato
    # (src/model.py: simplified_feature_set = 41 - 8 Pinnacle/consensus
    # - 3 overround = 32). Usare le 41 o le 38 renderebbe il confronto
    # "catboost di produzione" una fake: non sarebbe il modello in uso.
    from model import simplified_feature_set
    prod_cols = [c for c in simplified_feature_set(feature_list) if c not in FORBIDDEN]
    extra_cols = [c for c in df.columns
                  if c not in prod_cols and c not in FORBIDDEN and c != "_key"
                  and c not in ("Date", "FTR", "HomeTeam", "AwayTeam")]
    all_cols = prod_cols + extra_cols
    Xall = df[all_cols].copy()
    n_tr = len(df_train)
    Xtr = Xall.iloc[:n_tr].reset_index(drop=True)
    Xte2 = Xall.iloc[n_tr:].reset_index(drop=True)
    keep = [c for c in Xtr.columns
            if pd.api.types.is_numeric_dtype(Xtr[c]) and Xtr[c].notna().sum() > 50]
    Xtr, Xte2 = Xtr[keep], Xte2[keep]
    prod_keep = [c for c in keep if c in prod_cols]
    extra_keep = [c for c in keep if c not in prod_cols]
    print(f"  feature di produzione: {len(prod_keep)} | nuove: {len(extra_keep)} "
          f"| totali {len(keep)}")
    cover = {c: float(Xtr[c].notna().mean()) for c in extra_keep}
    low = sorted([(v, c) for c, v in cover.items() if v < 0.9])[:8]
    if low:
        print("  copertura <90% nel training: " +
              ", ".join(f"{c} {v*100:.0f}%" for v, c in low))

    # ── 5. MODELLI E TUNING ────────────────────────────────────────────
    from catboost import CatBoostClassifier
    from sklearn.model_selection import ParameterSampler
    from xgboost import XGBClassifier

    PROD_CB = {"iterations": 200, "learning_rate": 0.05, "depth": 4}

    def cb_model():
        return CatBoostClassifier(loss_function="MultiClass", eval_metric="Accuracy",
                                  random_seed=42, verbose=False, **PROD_CB)

    def xgb_model(p):
        return XGBClassifier(objective="multi:softprob", num_class=3,
                             eval_metric="mlogloss", random_state=42, n_jobs=XGB_NJOBS,
                             tree_method="hist", **p)

    xgb_space = {
        "n_estimators": [200, 400, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 6],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "reg_lambda": [0.5, 1.0, 5.0],
        "reg_alpha": [0.0, 0.5, 2.0],
        "gamma": [0.0, 0.5],
    }
    sampler = list(ParameterSampler(xgb_space, n_iter=args.tune, random_state=42))

    n = len(Xtr)
    edges = [int(n * i / 4) for i in range(5)]

    def cv_rps(fit_fn, cols_list, mask=None):
        """RPS out-of-fold con CV temporale (mai split casuale).

        Se `mask` e' dato, la CV temporale avviene SUL SOTTOSINSIEME delle
        righe mascherate: e' l'unico modo per confrontare equamente un
        modello con l'xG (disponibile solo dal 2014-15) e uno senza. Senza
        questo, il primo fold del training (2010-2013) non avrebbe nessuna
        riga con xG: addestramento vuoto e XGBoost morre con
        "Check failed: sum_weight >= kRtEps".
        """
        idx = np.where(mask)[0] if mask is not None else np.arange(n)
        e2 = [int(len(idx) * i / 4) for i in range(5)]
        oof = np.full(n, np.nan)
        for i in range(3):
            tr = idx[0:e2[i + 1]]
            va = idx[e2[i + 1]:e2[i + 2]]
            if len(tr) == 0:
                continue
            m = fit_fn()
            m.fit(Xtr.iloc[tr][cols_list], y[tr])
            oof[va] = rps_per_match(y[va], m.predict_proba(Xtr.iloc[va][cols_list]))
        return float(np.nanmean(oof))

    def deployable(c):
        """Una feature e' deployabile solo se le sue quote sono disponibili
        PRIMA della partita anche in produzione (un solo bookmaker)."""
        if c.startswith(NON_DEPLOYABLE_PREFIX):
            return False
        if c.endswith("_overround") and not c.startswith("mkt_b365_"):
            return False
        return True

    # Il tuning gira sul set MASSIMO DEPLOYABILE (D): gli iperparametri
    # vengono scelti una volta e poi applicati a tutti i set, cosi' il
    # confronto fra set e' equo (nessun set avvantaggiato da una griglia
    # tarata su di lui).
    def is_xg(c):
        """Le colonne xG hanno prefissi propri. NON usare endswith('_home'):
        anche 'elo_home' finisce cosi' e renderebbe la maschera sempre vera."""
        return c.startswith(XG_PREFIXES)

    xg_cols_all = [c for c in keep if is_xg(c)]
    has_xg_all = (Xtr[xg_cols_all].notna().any(axis=1).to_numpy() if xg_cols_all
                  else np.ones(n, dtype=bool))
    b365_all = [c for c in keep if c in B365_EXT]
    books_all = [c for c in keep if c.startswith(BOOK_PREFIX)]
    deploy_all = [c for c in keep if deployable(c)]
    tune_cols = (prod_keep + [c for c in deploy_all if c not in prod_keep])
    tune_mask = has_xg_all

    ckpt_path = ROOT / "models" / f"xgb_tuning_{league.key}.json"
    best = None
    if ckpt_path.exists() and not args.force_tune:
        ck = json.loads(ckpt_path.read_text())
        if ck.get("n_features") == len(tune_cols):
            best = (ck["rps_oof"], ck["xgb_params"])
            print(f"\n  parametri ripresi dal checkpoint {ckpt_path.name}: "
                  f"RPS OOF {best[0]:.5f}")
            print(f"    {best[1]}")
    if best is None:
        best = _tune(sampler, cv_rps, lambda: xgb_model, tune_cols, tune_mask, args, xgb_params=None)
    if best is None:
        print("  NESSUNA CONFIGURAZIONE VALIDA")
        return
    xgb_params = best[1]
    ckpt_path.write_text(json.dumps(
        {"rps_oof": best[0], "xgb_params": xgb_params,
         "n_features": len(tune_cols)}, indent=2))
    print(f"    (tuning: {args.tune} configurazioni su {len(tune_cols)} feature)")

    xg_cols = xg_cols_all
    has_xg = has_xg_all
    print(f"\n  righe di training con xG disponibile: {has_xg.sum()}/{n} "
          f"({has_xg.mean()*100:.1f}%)")

    xg_set = [c for c in extra_keep if is_xg(c)]
    b365_set = [c for c in extra_keep if c in B365_EXT]
    books_set = [c for c in extra_keep if c.startswith(BOOK_PREFIX)]
    other_set = [c for c in extra_keep
                 if c not in xg_set and c not in b365_set and c not in books_set
                 and deployable(c)]
    print(f"  gruppi: xG {len(xg_set)} | bet365-esteso {len(b365_set)} | "
          f"altri book {len(books_set)} | altro {len(other_set)}")
    if other_set:
        print(f"    (scartate non deployabili: "
              f"{[c for c in extra_keep if not deployable(c)][:8]})")

    # Tutti i set sono addestrati e validati sulle STESSE righe (quelle con
    # xG disponibile): cosi' l'unica differenza fra un set e l'altro sono le
    # feature. A_tutte righe mostra invece cosa succede addestrando anche
    # sulle stagioni pre-2014 (dove l'xG non esiste).
    sets = {
        "A produzione": (prod_keep, has_xg),
        "B produzione+xG": (prod_keep + xg_set, has_xg),
        "C produzione+bet365": (prod_keep + b365_set, has_xg),
        "D max_deployabile": (prod_keep + xg_set + b365_set + other_set, has_xg),
        "E D+line_shopping": (prod_keep + xg_set + b365_set + other_set
                              + books_set, has_xg),
        "A_tutte_righe": (prod_keep, None),
    }
    print(f"\n  CONFRONTO DEI SET DI FEATURE (CV temporale, XGBoost):")
    set_scores, set_cols = {}, {}
    for name, (cs, mask) in sets.items():
        s = cv_rps(lambda: xgb_model(xgb_params), cs, mask)
        set_scores[name] = {"rps_oof": s, "n_features": len(cs),
                            "train_rows": int(mask.sum()) if mask is not None else n}
        set_cols[name] = (cs, mask)
        print(f"    {name:<24} {len(cs):>3} feature  RPS OOF {s:.5f}")
    cb_score = cv_rps(cb_model, prod_keep, has_xg)
    print(f"    {'CatBoost produzione':<24} {len(prod_keep):>3} feature  "
          f"RPS OOF {cb_score:.5f}")

    # ── 6. IMPORTANZA E SIGNIFICATIVITÀ ───────────────────────────────
    print(f"\n  IMPORTANZA (XGBoost, gain):")
    m_imp = xgb_model(xgb_params)
    m_imp.fit(Xtr[keep], y)
    gain = m_imp.get_booster().get_score(importance_type="gain")
    imp = sorted(gain.items(), key=lambda x: -x[1])
    tot = sum(v for _, v in imp)
    for k, v in imp[:18]:
        print(f"    {k:<26} {v/tot*100:>5.2f}%")

    print(f"\n  SIGNIFICATIVITÀ DROP-COLUMN (test set, rimozione di una feature):")
    base_model = xgb_model(xgb_params)
    base_model.fit(Xtr[keep], y)
    rps_base = rps_per_match(yte, base_model.predict_proba(Xte2[keep]))
    drop_results = {}
    for c in [k for k, _ in imp[:8]]:
        cs = [x for x in keep if x != c]
        m2 = xgb_model(xgb_params)
        m2.fit(Xtr[cs], y)
        rps2 = rps_per_match(yte, m2.predict_proba(Xte2[cs]))
        bs = paired_bootstrap(rps_base, rps2)
        drop_results[c] = bs
        print(f"    senza {c:<24} RPS {rps_base.mean():.5f} -> {rps2.mean():.5f} "
              f"({bs['diff']:+.5f}) p={bs['p_bootstrap']:.3f}")

    # ── 7. REFIT E CONFRONTO FINALE ───────────────────────────────────
    print(f"\n  REFIT e CONFRONTO FINALE (test set, {len(yte)} partite):")
    final = {}

    def fit_predict(model, cs, mask=None):
        if mask is not None:
            idx = np.where(mask)[0]
            model.fit(Xtr.iloc[idx][cs], y[idx])
        else:
            model.fit(Xtr[cs], y)
        return model.predict_proba(Xte2[cs])

    def record(name, p, n_feat):
        pred = [int(np.argmax(x)) for x in p]
        final[name] = {
            "n_features": n_feat,
            "accuracy": float(accuracy(yte, pred)),
            "f1_macro": float(f1_scores(yte, pred)["macro"]),
            "mse": float(brier_mse(yte, p.tolist())),
            "log_loss": float(log_loss(yte, p.tolist())),
            "rps": float(ranked_probability_score(yte, p.tolist())),
        }

    record("catboost_produzione", fit_predict(cb_model(), prod_keep, has_xg),
           len(prod_keep))
    record("catboost_tutte_righe", fit_predict(cb_model(), prod_keep, None),
           len(prod_keep))
    for name, (cs, mask) in sets.items():
        record(f"xgboost_{name[0]}", fit_predict(xgb_model(xgb_params), cs, mask),
               len(cs))

    # benchmark bookmaker allineato per chiave
    from evaluate_market import load_odds_by_key
    odds_by_key = load_odds_by_key(Path(league.raw_path))
    y_b, b365 = [], []
    # NB: dopo il dropna l'indice di df NON e' contiguo, quindi df.iloc[n_tr:]
    # non e' il test set: si usa df_test, che e' lo stesso ordine di Xte2.
    for k, (_, r) in enumerate(df_test.iterrows()):
        key = (r["Date"].strftime("%Y-%m-%d"), str(r["HomeTeam"]).strip(),
               str(r["AwayTeam"]).strip())
        o = odds_by_key.get(key, {}).get("b365_open")
        if not o:
            continue
        p = devig3(*o) if isinstance(o, (list, tuple, np.ndarray)) else None
        if p is None:
            continue
        y_b.append(yte[k])
        b365.append(p)
    y_b = np.array(y_b)
    b365 = np.array(b365)
    final["bet365"] = {
        "n_features": 0,
        "accuracy": float(accuracy(y_b, [int(np.argmax(x)) for x in b365])),
        "f1_macro": float(f1_scores(y_b, [int(np.argmax(x)) for x in b365])["macro"]),
        "rps": float(ranked_probability_score(y_b, b365.tolist())),
    }
    print(f"  bet365 allineato su {len(y_b)} partite")

    print(f"\n  {'modello':<26} {'feat':>5} {'accuracy':>9} {'F1mac':>7} "
          f"{'MSE':>8} {'RPS':>9} {'logloss':>9}")
    for name, v in final.items():
        print(f"  {name:<26} {v['n_features']:>5} {v['accuracy']:>8.2%} "
              f"{v.get('f1_macro', float('nan')):>7.4f} {v.get('mse', float('nan')):>8.5f} "
              f"{v['rps']:>9.5f} {v.get('log_loss', float('nan')):>9.5f}")

    print(f"\n  {'─'*96}\n  TEST PAIRED contro bet365 (RPS per partita, appaiate)\n  {'─'*96}")
    rps_b = rps_per_match(y_b, b365)
    tests = {}
    for name, v in final.items():
        if name == "bet365":
            continue
        if name in ("catboost_produzione", "catboost_tutte_righe"):
            p = fit_predict(cb_model(), prod_keep,
                            has_xg if name == "catboost_produzione" else None)
        else:
            key = [k for k in sets if name.endswith(k[0])][0]
            cs, mask = set_cols[key]
            p = fit_predict(xgb_model(xgb_params), cs, mask)
        rps_m = rps_per_match(yte, p)[:len(y_b)]
        bs = paired_bootstrap(rps_b, rps_m)
        wc = paired_wilcoxon(rps_b, rps_m)
        dm = dm_test(rps_b, rps_m)
        tests[name] = {"bootstrap": bs, "wilcoxon": wc, "dm": dm}
        verdict = "BATTUTO" if bs["diff"] < 0 and bs["p_bootstrap"] < 0.05 else "non battuto"
        print(f"\n  {name} vs bet365  [{verdict}]")
        print(f"    diff RPS {bs['diff']:+.6f}  CI 95% [{bs['ci_low']:+.6f}, "
              f"{bs['ci_high']:+.6f}]")
        print(f"    p bootstrap {bs['p_bootstrap']:.4f} | Wilcoxon {wc['p']:.4f} | DM p={dm['p']}")

    out = Path(args.out) if args.out else (ROOT / "models" / f"xgb_{league.key}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"xgb_params": xgb_params,
                   "cv_rps": {"xgb_best": best[0], "catboost_produzione": cb_score},
                   "set_scores_cv": set_scores,
                   "n_features_used": len(keep),
                   "coverage_train": cover,
                   "importance": {k: float(v / tot) for k, v in imp},
                   "drop_column": drop_results,
                   "final": final,
                   "tests_vs_bet365": tests}, f, indent=2, default=float)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
