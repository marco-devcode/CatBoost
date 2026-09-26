#!/usr/bin/env python3
"""
Ricalcola le probabilita' SOLO delle partite la cui stima e' stata prodotta dal
modello addestrato sul dataset troncato da 22 righe (dopo il commit 515455e),
e non su quello trainato su 5642 partite.

Perimetro: 20 partite (GG4 + GG5), gia' incontrate come "cambiata" tra i commit.
Fuori perimetro: GG1-GG3, la cui prob proviene ancora dal training da 4508 righe
del commit 68c0b24, e tutte le partite senza risultato (gia' ricalcolate dalla
pipeline con il modello buono).

Cosa tocca: solo prob, pred, features, kelly. NON tocca risultato/esito/ok,
che restano il dato reale della partita.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "webapp" / "public" / "predictions.json"

MODEL_PATH = MODELS / "catboost_seriea_v1.cbm"
RATINGS = DATA / "serie_a_ratings.csv"
CLEAN = DATA / "serie_a_clean.csv"
WINDOW = 5

FEATURE_ORDER = [
    "pi_home_H", "pi_away_H", "pi_home_A", "pi_away_A",
    "home_form_5", "away_form_5",
    "home_goal_diff_5", "away_goal_diff_5",
    "home_rest_days", "away_rest_days", "rest_advantage",
    "h2h_home_winrate",
    "home_goals_scored_5", "home_goals_conceded_5",
    "away_goals_scored_5", "away_goals_conceded_5",
    "home_promoted", "away_promoted",
]
LABELS = ["H", "D", "A"]
KELLY_FRACTION = 0.25

# Le 20 partite da ricalcolare (GG4 + GG5), identificate come "prob scritta
# durante l'era modello-22" tracciando i commit.
TARGET_KEYS = {
    "2026-09-11|Venezia|Fiorentina",
    "2026-09-12|Atalanta|Cagliari",
    "2026-09-12|Genoa|Frosinone",
    "2026-09-12|Lazio|Milan",
    "2026-09-13|Lecce|Monza",
    "2026-09-13|Napoli|Bologna",
    "2026-09-13|Sassuolo|Juventus",
    "2026-09-14|Como|Parma",
    "2026-09-14|Inter|Udinese",
    "2026-09-14|Torino|Roma",
    "2026-09-18|Monza|Sassuolo",
    "2026-09-19|Bologna|Torino",
    "2026-09-19|Roma|Inter",
    "2026-09-19|Udinese|Cagliari",
    "2026-09-19|Venezia|Lazio",
    "2026-09-20|Fiorentina|Napoli",
    "2026-09-20|Frosinone|Como",
    "2026-09-20|Juventus|Atalanta",
    "2026-09-20|Milan|Lecce",
    "2026-09-20|Parma|Genoa",
}


def key(m):
    return f"{m['date']}|{m['home']}|{m['away']}"


def kelly_fraction(p, odds, fraction=KELLY_FRACTION):
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, ((b * p) - (1 - p)) / b * fraction)


def compute_features(hist, home, away, match_date, h2h_cache, current_season_teams):
    """Replica di predict.compute_features_for_match (stesse feature, stesso ordine)."""
    past = hist[hist["Date"] < match_date].copy()
    if len(past) == 0:
        past = hist.copy()

    feats = {}

    for team, role in [(home, "home"), (away, "away")]:
        sh = past[past["HomeTeam"] == team]
        sa = past[past["AwayTeam"] == team]
        last = pd.concat([sh, sa]).sort_values("Date").tail(1)
        if len(last) == 0:
            feats[f"pi_{role}_H"] = 0.0
            feats[f"pi_{role}_A"] = 0.0
        else:
            lr = last.iloc[0]
            was_home = (lr["HomeTeam"] == team)
            if role == "home":
                feats["pi_home_H"] = lr["pi_home_H"] if was_home else lr["pi_away_H"]
                feats["pi_home_A"] = lr["pi_home_A"] if was_home else lr["pi_away_A"]
            else:
                feats["pi_away_H"] = lr["pi_home_H"] if was_home else lr["pi_away_H"]
                feats["pi_away_A"] = lr["pi_home_A"] if was_home else lr["pi_away_A"]

    def team_form(team, is_home):
        tcol = "HomeTeam" if is_home else "AwayTeam"
        scol = "FTHG" if is_home else "FTAG"
        acol = "FTAG" if is_home else "FTHG"
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0:
            return 1.0
        pts = np.where(sub[scol].values > sub[acol].values, 3,
                       np.where(sub[scol].values == sub[acol].values, 1, 0))
        return float(np.mean(pts))

    feats["home_form_5"] = team_form(home, True)
    feats["away_form_5"] = team_form(away, False)

    def team_gd(team, is_home):
        tcol = "HomeTeam" if is_home else "AwayTeam"
        scol = "FTHG" if is_home else "FTAG"
        acol = "FTAG" if is_home else "FTHG"
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0:
            return 0.0
        return float(np.mean(sub[scol].values - sub[acol].values))

    feats["home_goal_diff_5"] = team_gd(home, True)
    feats["away_goal_diff_5"] = team_gd(away, False)

    def rest(team):
        sub = past[(past["HomeTeam"] == team) | (past["AwayTeam"] == team)]
        if len(sub) == 0:
            return 7.0
        return float((match_date - sub["Date"].max()).days)

    hd, ad = rest(home), rest(away)
    feats["home_rest_days"] = hd
    feats["away_rest_days"] = ad
    feats["rest_advantage"] = hd - ad

    def roll_goals(team, is_home):
        tcol = "HomeTeam" if is_home else "AwayTeam"
        scol = "FTHG" if is_home else "FTAG"
        acol = "FTAG" if is_home else "FTHG"
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0:
            return np.nan, np.nan
        return float(sub[scol].mean()), float(sub[acol].mean())

    hs, hc = roll_goals(home, True)
    a_s, a_c = roll_goals(away, False)
    feats["home_goals_scored_5"] = hs
    feats["home_goals_conceded_5"] = hc
    feats["away_goals_scored_5"] = a_s
    feats["away_goals_conceded_5"] = a_c

    k = (min(home, away), max(home, away))
    if h2h_cache is not None and k in h2h_cache:
        feats["h2h_home_winrate"] = h2h_cache[k]
    else:
        sub = past[((past["HomeTeam"] == home) & (past["AwayTeam"] == away)) |
                   ((past["HomeTeam"] == away) & (past["AwayTeam"] == home))]
        if len(sub) == 0:
            feats["h2h_home_winrate"] = 0.5
        else:
            hf = sub[sub["HomeTeam"] == home]
            feats["h2h_home_winrate"] = (
                float((hf["FTR"] == "H").sum() / len(hf)) if len(hf) else 0.5
            )
            if h2h_cache is not None:
                h2h_cache[k] = feats["h2h_home_winrate"]

    last_season = sorted(hist["season"].unique())[-1]
    last_teams = set(hist[hist["season"] == last_season]["HomeTeam"]) | \
        set(hist[hist["season"] == last_season]["AwayTeam"])

    def promoted(t):
        if t not in current_season_teams:
            return 0.0
        return 0.0 if t in last_teams else 1.0

    feats["home_promoted"] = promoted(home)
    feats["away_promoted"] = promoted(away)
    return feats


def json_safe(o):
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"modello mancante: {MODEL_PATH}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"modello caricato: {MODEL_PATH}")

    hist = pd.read_csv(RATINGS, parse_dates=["Date"])
    print(f"storico: {len(hist)} partite, {hist['season'].nunique()} stagioni")

    data = json.loads(OUT.read_text(encoding="utf-8"))

    # squadre della stagione corrente: dal calendario di tutte le giornate
    teams = set()
    for ms in data["giornate"].values():
        for m in ms:
            teams.add(m["home"])
            teams.add(m["away"])
    print(f"squadre stagione corrente: {len(teams)}")

    h2h_cache = {}
    changed = 0
    for g in sorted(data["giornate"], key=int):
        for m in data["giornate"][g]:
            k = key(m)
            if k not in TARGET_KEYS:
                continue
            feats = compute_features(
                hist, m["home"], m["away"],
                pd.Timestamp(m["date"]), h2h_cache, teams,
            )
            X = pd.DataFrame([{c: feats.get(c, 0.0) for c in FEATURE_ORDER}])
            probs = model.predict_proba(X)[0]
            old = m.get("prob")
            m["prob"] = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
            m["pred"] = LABELS[int(np.argmax(probs))]
            m["features"] = {c: (None if (isinstance(feats.get(c), float)
                                         and math.isnan(feats.get(c)))
                                 else round(float(feats.get(c, 0.0)), 3))
                             for c in FEATURE_ORDER}
            # ok rifinito sulla nuova previsione, il risultato resta quello reale
            if m.get("esito"):
                m["ok"] = bool(m["pred"] == m["esito"])
            # kelly ricalcolato sulla nuova stima
            bm = m.get("bm")
            if bm:
                p = m["prob"][m["pred"]] / 100.0
                kf = kelly_fraction(p, bm.get(m["pred"]))
                m["kelly"] = ({"pick": m["pred"], "model_p": round(p, 3),
                               "odd": bm.get(m["pred"]),
                               "stake_pct": round(kf * 100, 2)} if kf > 0 else None)
            changed += 1
            print(f"  {m['date']} {m['home']:12s}-{m['away']:12s} "
                  f"{old} -> {m['prob']}  pred={m['pred']} ok={m.get('ok')}")

    # stats ricalcolate
    tot = sum(1 for ms in data["giornate"].values() for m in ms if m.get("ok") is not None)
    cor = sum(1 for ms in data["giornate"].values() for m in ms if m.get("ok") is True)
    data["stats"] = {"corrette": cor, "total": tot}
    if isinstance(data["meta"].get("stats"), dict):
        data["meta"]["stats"] = {"corrette": cor, "total": tot}

    OUT.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False,
                              allow_nan=False), encoding="utf-8")
    print(f"\nricalcolate {changed} partite")
    print(f"stats: {cor}/{tot}")


if __name__ == "__main__":
    main()
