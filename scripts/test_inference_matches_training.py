#!/usr/bin/env python3
"""Verifica che l'inferenza riceva le STESSE feature su cui il modello e'
stato addestrato, e che le quote delle giocate sopravvivano al calendario
openfootball.

Non richiede catboost: il modulo predict viene importato con un modulo finto,
perche' i due fix verificati qui (pi_* e h2h) sono feature calcolate PRIMA
del modello.

Esegui:  python3.14 scripts/test_inference_matches_training.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

fake = types.ModuleType("catboost")
fake.CatBoostClassifier = object
sys.modules["catboost"] = fake

from predict import compute_features_for_match, FEATURE_ORDER  # noqa: E402

# Le feature devono combaciare su TUTTO lo storico, non solo sull'ultima
# stagione: un bug che si manifesta solo su 2627 passerebbe inosservato.
LEAGUES = [
    ("seriea", "data/serie_a_ratings.csv", "data/serie_a_features.csv"),
    ("eredivisie", "data/eredivisie_ratings.csv", "data/eredivisie_features.csv"),
]

# `promoted` e' escluso perche' usa una semantica documentata e diversa
# (predict.is_promoted lavora con l'insieme squadre del CALENDARIO, il
# training con la stagione precedente): su 2627 coincide, e la differenza
# sulle stagioni passate e' intenzionale, non un disallineamento.
CHECKED = [c for c in FEATURE_ORDER if not c.endswith("_promoted")]

failures = 0

for key, ratings_path, features_path in LEAGUES:
    hist = pd.read_csv(ROOT / ratings_path, parse_dates=["Date"], low_memory=False)
    feats = pd.read_csv(ROOT / features_path, parse_dates=["Date"], low_memory=False)

    # Le righe 2627 sono quelle che contano per il sito; sul resto dello
    # storico si campiona, perche' il confronto e' O(n) e il codice sotto
    # e' identico.
    current = feats[feats["Date"] >= "2026-08-01"]
    sample = feats.sample(n=min(300, len(feats)), random_state=1)
    for label, rows in (("26/27", current), ("storico (campione)", sample)):
        bad = {}
        # iterrows, non itertuples: pandas rinomina i campi itertuples che
        # non sono identificatori validi, e un getattr silenzioso su un
        # campo rinominato restituiva la colonna sbagliata invece di
        # fallire. Un test che legge il dato sbagliato verde e' peggio di
        # nessun test.
        for _, r in rows.iterrows():
            d = compute_features_for_match(hist, r.HomeTeam, r.AwayTeam, r.Date)
            for c in CHECKED:
                tv = r[c]
                iv = d.get(c)
                tv_nan = tv is None or (isinstance(tv, float) and np.isnan(tv))
                iv_nan = iv is None or (isinstance(iv, float) and np.isnan(iv))
                if tv_nan and iv_nan:
                    continue
                if tv_nan != iv_nan or abs(float(tv) - float(iv)) > 1e-6:
                    bad[c] = bad.get(c, 0) + 1
        n_bad = sum(bad.values())
        status = "OK  " if n_bad == 0 else "FAIL"
        print(f"[{status}] {key:11s} {label:20s} righe {len(rows):5d}  "
              f"discrepanze {n_bad}  {bad if bad else ''}")
        failures += n_bad

# ── Quote: la fonte del calendario non puo' far perdere le quote ─────────
# Si legge il RAW di football-data, non il dataset dei rating: il ratings
# CSV non contiene le colonne B365. E' la fonte da cui predict_fixtures
# pesca le quote delle giocate, quindi e' il posto giusto in cui verificare
# che la fonte le abbia davvero.
print()
RAW = {
    "seriea": "data/serie_a_raw.csv",
    "eredivisie": "data/eredivisie_raw.csv",
}
for key, _, _ in LEAGUES:
    raw = pd.read_csv(ROOT / RAW[key], low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    played = raw[raw["Date"] >= "2026-08-01"]
    ok = played[["B365H", "B365D", "B365A"]].apply(
        lambda c: c.map(lambda v: isinstance(v, (int, float)) and np.isfinite(v))
    ).all(axis=1)
    status = "OK  " if ok.all() else "FAIL"
    print(f"[{status}] {key:11s} giocate 26/27 con quote B365 in football-data: "
          f"{int(ok.sum())}/{len(played)}")
    if not ok.all():
        failures += int((~ok).sum())

print()
if failures:
    print(f"FALLITO: {failures} discrepanze")
    sys.exit(1)
print("TUTTO OK: inferenza allineata al training, quote disponibili.")
