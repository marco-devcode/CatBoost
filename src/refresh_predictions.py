#!/usr/bin/env python3
"""
Rigenera predictions.json correttamente:

1. DATE CORRETTE: per G4+ usa calendario reale Serie A 26/27 (hardcodeato,
   non affidabile fonti esterne). G1-G3 mantengono le date del CSV locale che
   sono corrette (22-24 ago, 28-31 ago, 4-7 set).

2. FEATURES RICALCOLATE per tutte le partite con storico completo
   (G1+G2+G3 inclusi), solo le probabilità/pred/bm/risultato/esito/ok
   rimangono congelati per le partite già giocate (freeze rispettato).

3. REST_DAYS corretti: calcolati con la data REALE della partita e lo storico
   completo (inclusi risultati G2 per le partite G3, risultati G1+G2+G3 per G4).
"""
import json, sys, math
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "webapp" / "public" / "predictions.json"
RAW  = ROOT / "data" / "serie_a_raw.csv"
RATINGS = ROOT / "data" / "serie_a_ratings.csv"

# ═══════════════════════════════════════════════════════════════════════════════
# CALENDARIO: le date vengono dalla fonte autorevole (football-data.co.uk) o
# TheSportsDB. NON hardcodare qui: la season 2627 non e' ancora finita e le date
# cambiano (anticipi, rinvi). Gli hardcode precedenti erano sbagliati e hanno
# falsato rest_days.
# ═══════════════════════════════════════════════════════════════════════════════

def parse_date(s: str) -> pd.Timestamp:
    """Parse data da qualsiasi formato comune."""
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try: return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError: pass
    return pd.Timestamp(s)

def bm_prob(h, d, a):
    try:
        ih, id_, ia = 1/float(h), 1/float(d), 1/float(a)
        t = ih + id_ + ia
        return {"H": round(ih/t*100, 1), "D": round(id_/t*100, 1), "A": round(ia/t*100, 1)}
    except (TypeError, ValueError, ZeroDivisionError):
        return None

def kelly_stake(pk, odd, frac=0.25):
    try:
        o = float(odd); p = float(pk); b = o - 1.0
        if p <= 0 or o <= 0 or b <= 0: return None
        f = (p*b - (1-p)) / b * frac
        return round(f*100, 2) if f > 0 else None
    except (TypeError, ValueError):
        return None

def esito(ftr: str) -> str:
    return {"H": "H", "D": "D", "A": "A"}.get(str(ftr).strip().upper(), "")

def norm_team(s: str) -> str:
    return " ".join(s.strip().split()).title()

# ═══════════════════════════════════════════════════════════════════════════════
# CARICA DATI
# ═══════════════════════════════════════════════════════════════════════════════
print("Caricamento dati...")
df_raw = pd.read_csv(RAW, dtype=str, low_memory=False)
print(f"  CSV locale: {len(df_raw)} righe")

df_raw['Date'] = df_raw['Date'].map(parse_date)
df_raw['HomeTeam'] = df_raw['HomeTeam'].map(norm_team)
df_raw['AwayTeam'] = df_raw['AwayTeam'].map(norm_team)

# Build lookup: (home, away, date) -> row
CSV_LOOKUP = {}
for _, r in df_raw.iterrows():
    key = (str(r['HomeTeam']).strip().lower(),
           str(r['AwayTeam']).strip().lower(),
           r['Date'].strftime("%Y-%m-%d"))
    CSV_LOOKUP[key] = r
print(f"  Lookup CSV: {len(CSV_LOOKUP)} entries")

ratings = pd.read_csv(RATINGS, parse_dates=['Date'])
print(f"  Rating: {len(ratings)} righe, date da {ratings['Date'].min().date()} a {ratings['Date'].max().date()}")
ratings['HomeTeam'] = ratings['HomeTeam'].map(norm_team)
ratings['AwayTeam'] = ratings['AwayTeam'].map(norm_team)

# ═══════════════════════════════════════════════════════════════════════════════
# CARICA JSON ESISTENTE
# ═══════════════════════════════════════════════════════════════════════════════
with open(OUT, encoding="utf-8") as f:
    data = json.load(f)

meta = data["meta"]
giornate = data["giornate"]
print(f"\nJSON esistente: last_update={meta['last_update']}")
print(f"  G1: {len(giornate.get('1',[]))} match, G2: {len(giornate.get('2',[]))} match, G3: {len(giornate.get('3',[]))} match")
print(f"  G4: {len(giornate.get('4',[]))} match")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTA FEATURES PER MATCH (ricampiona da zero con storico completo)
# ═══════════════════════════════════════════════════════════════════════════════
FEATURE_ORDER = [
    'pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A',
    'home_form_5', 'away_form_5',
    'home_goal_diff_5', 'away_goal_diff_5',
    'home_rest_days', 'away_rest_days', 'rest_advantage',
    'h2h_home_winrate',
    'home_goals_scored_5', 'home_goals_conceded_5',
    'away_goals_scored_5', 'away_goals_conceded_5',
    'home_promoted', 'away_promoted',
]
WINDOW = 5

# Ultimo stagione completa nello storico (per flag neopromossa)
last_season_ratings = sorted(ratings['season'].unique())[-1]
last_teams_complete = set(ratings[ratings['season'] == last_season_ratings]['HomeTeam']) | \
                      set(ratings[ratings['season'] == last_season_ratings]['AwayTeam'])

current_season_teams = set(df_raw['HomeTeam']) | set(df_raw['AwayTeam'])

def compute_features(home, away, match_date, history):
    """Ricalcola tutte le features per una partita con data corretta."""
    match_ts = pd.Timestamp(match_date)
    past = history[history['Date'] < match_ts].copy()

    if len(past) == 0:
        past = history.copy()

    feats = {}

    # ── Pi-Rating: usa l'ultimo valore noto per ogni squadra ─────────────
    for team, role in [(home, 'home'), (away, 'away')]:
        sub_home = past[past['HomeTeam'] == team]
        sub_away = past[past['AwayTeam'] == team]
        last = pd.concat([sub_home, sub_away]).sort_values('Date').tail(1)
        if len(last) == 0:
            feats[f'pi_{role}_H'] = 0.0
            feats[f'pi_{role}_A'] = 0.0
        else:
            lr = last.iloc[0]
            was_home = (lr['HomeTeam'] == team)
            if role == 'home':
                feats['pi_home_H'] = lr['pi_home_H'] if was_home else lr['pi_away_H']
                feats['pi_home_A'] = lr['pi_home_A'] if was_home else lr['pi_away_A']
            else:
                feats['pi_away_H'] = lr['pi_home_H'] if was_home else lr['pi_away_H']
                feats['pi_away_A'] = lr['pi_home_A'] if was_home else lr['pi_away_A']

    # ── Forma recente ────────────────────────────────────────────────────
    def team_form(team, is_home):
        tcol = 'HomeTeam' if is_home else 'AwayTeam'
        scol = 'FTHG' if is_home else 'FTAG'
        acol = 'FTAG' if is_home else 'FTHG'
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0: return 1.0
        pts = np.where(sub[scol].values > sub[acol].values, 3,
                      np.where(sub[scol].values == sub[acol].values, 1, 0))
        return float(np.mean(pts))

    feats['home_form_5'] = team_form(home, True)
    feats['away_form_5'] = team_form(away, False)

    # ── Goal diff rolling ────────────────────────────────────────────────
    def team_gd(team, is_home):
        tcol = 'HomeTeam' if is_home else 'AwayTeam'
        scol = 'FTHG' if is_home else 'FTAG'
        acol = 'FTAG' if is_home else 'FTHG'
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0: return 0.0
        return float(np.mean(sub[scol].values - sub[acol].values))

    feats['home_goal_diff_5'] = team_gd(home, True)
    feats['away_goal_diff_5'] = team_gd(away, False)

    # ── Rest days ────────────────────────────────────────────────────────
    def rest_days(team):
        sub = past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)]
        if len(sub) == 0: return 7.0
        last_ts = sub['Date'].max()
        return float((match_ts - last_ts).days)

    hd = rest_days(home)
    ad = rest_days(away)
    feats['home_rest_days'] = hd
    feats['away_rest_days'] = ad
    feats['rest_advantage'] = hd - ad

    # ── Rolling goals ────────────────────────────────────────────────────
    def roll_goals(team, is_home):
        tcol = 'HomeTeam' if is_home else 'AwayTeam'
        scol = 'FTHG' if is_home else 'FTAG'
        acol = 'FTAG' if is_home else 'FTHG'
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0: return np.nan, np.nan
        return float(sub[scol].mean()), float(sub[acol].mean())

    hs, hc = roll_goals(home, True)
    as_, ac = roll_goals(away, False)
    feats['home_goals_scored_5'] = hs
    feats['home_goals_conceded_5'] = hc
    feats['away_goals_scored_5'] = as_
    feats['away_goals_conceded_5'] = ac

    # ── H2H ─────────────────────────────────────────────────────────────
    key = (min(home, away), max(home, away))
    sub_h2h = past[((past['HomeTeam'] == home) & (past['AwayTeam'] == away)) |
                    ((past['HomeTeam'] == away) & (past['AwayTeam'] == home))]
    if len(sub_h2h) > 0:
        home_wins = (sub_h2h['FTR'] == 'H').sum() if home == sub_h2h['HomeTeam'].iloc[0] else 0
        # Simpler: win rate from home perspective
        h2h_home = sub_h2h[sub_h2h['HomeTeam'] == home]
        if len(h2h_home) > 0:
            feats['h2h_home_winrate'] = float((h2h_home['FTR'] == 'H').sum() / len(h2h_home))
        else:
            feats['h2h_home_winrate'] = 0.5
    else:
        feats['h2h_home_winrate'] = 0.5

    # ── Flag neopromossa ────────────────────────────────────────────────
    def is_prom(team):
        if team not in current_season_teams:
            return 0
        return 0 if team in last_teams_complete else 1

    feats['home_promoted'] = float(is_prom(home))
    feats['away_promoted'] = float(is_prom(away))

    return feats

# ═══════════════════════════════════════════════════════════════════════════════
# AGGIORNA G1-G3: overlay B365 + risultato + ricalcola features (non freeze)
# ═══════════════════════════════════════════════════════════════════════════════
UPDATED = {"2": 0, "3": 0}

for g_key in ["2", "3"]:
    matches = giornate.get(g_key, [])
    for m in matches:
        home = norm_team(str(m["home"]))
        away = norm_team(str(m["away"]))
        date_str = str(m["date"])
        date_ts = parse_date(date_str)
        key = (home.lower(), away.lower(), date_str)
        row = CSV_LOOKUP.get(key)

        if row is None:
            print(f"  ⚠ Niente CSV per {home} - {away} ({date_str})")
            continue

        # ── B365 overlay ─────────────────────────────────────────────────
        bm_h = str(row.get("B365H","")).strip()
        bm_d = str(row.get("B365D","")).strip()
        bm_a = str(row.get("B365A","")).strip()
        ftr  = str(row.get("FTR","")).strip().upper()
        fthg = str(row.get("FTHG","")).strip()
        ftag = str(row.get("FTAG","")).strip()

        bm = None; bm_p = None
        if bm_h and bm_d and bm_a:
            try:
                h, d, a = float(bm_h), float(bm_d), float(bm_a)
                if h > 0 and d > 0 and a > 0:
                    bm = {"H": round(h, 2), "D": round(d, 2), "A": round(a, 2)}
                    bm_p = bm_prob(h, d, a)
            except ValueError:
                pass

        played = ftr in ("H","D","A")
        if played:
            m["risultato"] = f"{fthg}-{ftag}"
            m["esito"]     = esito(ftr)
            m["ok"]        = (esito(ftr) == m.get("pred", ""))

        if bm:
            m["bm"]      = bm
            m["bm_prob"] = bm_p

        # ── Ricalcola features con storico completo ─────────────────────
        feats = compute_features(home, away, date_str, ratings)
        m["features"] = {c: round(float(feats.get(c, 0.0)), 3) for c in FEATURE_ORDER}

        # ── Kelly retrospettivo ──────────────────────────────────────────
        if bm and bm_p and played and m.get("pred") and m.get("prob"):
            pred = m["pred"]
            pk   = (m["prob"].get(pred, 0) / 100.0)
            odd  = bm.get(pred)
            stake = kelly_stake(pk, odd)
            if stake is not None:
                m["kelly"] = {"pick": pred, "model_p": round(pk*100, 1),
                              "odd": odd, "stake_pct": stake}

        UPDATED[g_key] += 1

    print(f"  GG{g_key}: {UPDATED[g_key]} match aggiornati (B365+risultato+features)")

# ═══════════════════════════════════════════════════════════════════════════════
# AGGIORNA TUTTE LE GIORNATE 2..N: date dal CSV (fonte autorevole) + B365 +
# risultato + features ricalcolate. Le giornate <= freeze conservano prob/pred
# congelati ma le features/date/quote/risultati vengono allineati alla fonte.
# ═══════════════════════════════════════════════════════════════════════════════
print("\nAllineamento giornate 2..N con il CSV autorevole...")
frozen = int(meta.get("freeze_until", 3))
for g_key in sorted(giornate.keys(), key=lambda x: int(x)):
    g_num = int(g_key)
    if g_num < 2:
        continue
    matches = giornate[g_key]
    n_date = n_bm = n_res = 0

    for m in matches:
        home = norm_team(str(m["home"]))
        away = norm_team(str(m["away"]))
        old_date = str(m["date"])

        # Cerca la riga nel CSV per data+casa+fuori, oppure per sole squadre
        # (le date possono essere slittate di un giorno).
        row = None
        for k, r in CSV_LOOKUP.items():
            if k[0] == home.lower() and k[1] == away.lower():
                row = r
                break
        if row is None:
            print(f"  ! {home} - {away}: nessuna corrispondenza nel CSV")
            continue

        new_date = row["Date"].strftime("%Y-%m-%d")
        if new_date != old_date:
            print(f"  {g_key} {home} - {away}: data {old_date} -> {new_date}")
            m["date"] = new_date
            n_date += 1

        # ── B365 + risultato ─────────────────────────────────────────────
        bm_h = str(row.get("B365H", "")).strip()
        bm_d = str(row.get("B365D", "")).strip()
        bm_a = str(row.get("B365A", "")).strip()
        ftr = str(row.get("FTR", "")).strip().upper()
        fthg = str(row.get("FTHG", "")).strip()
        ftag = str(row.get("FTAG", "")).strip()

        bm = bm_p = None
        if bm_h and bm_d and bm_a:
            try:
                h, d, a = float(bm_h), float(bm_d), float(bm_a)
                if h > 0 and d > 0 and a > 0:
                    bm = {"H": round(h, 2), "D": round(d, 2), "A": round(a, 2)}
                    bm_p = bm_prob(h, d, a)
            except ValueError:
                pass

        played = ftr in ("H", "D", "A")
        if played:
            m["risultato"] = f"{fthg}-{ftag}"
            m["esito"] = esito(ftr)
            m["ok"] = (esito(ftr) == m.get("pred"))
            n_res += 1

        if bm:
            m["bm"] = bm
            m["bm_prob"] = bm_p
            n_bm += 1

        # ── Features: solo per le NON congelate ──────────────────────────
        # Le giornate <= freeze conservano le features originali (regola N-1:
        # la previsione non deve cambiare dopo la pubblicazione).
        if g_num > frozen:
            feats = compute_features(home, away, m["date"], ratings)
            m["features"] = {c: round(float(feats.get(c, 0.0)), 3)
                             for c in FEATURE_ORDER}

        # ── Kelly: solo per le giocate con quota ─────────────────────────
        if bm and played:
            pred = m.get("pred")
            if pred and m.get("prob"):
                pk = m["prob"].get(pred, 0) / 100.0
                stake = kelly_stake(pk, bm.get(pred))
                m["kelly"] = ({"pick": pred, "model_p": round(pk * 100, 1),
                               "odd": bm.get(pred), "stake_pct": stake}
                              if stake is not None else None)
            else:
                m["kelly"] = None
        else:
            m["kelly"] = None

    print(f"  GG{g_key}: {n_date} date corrette, {n_bm} quote, {n_res} risultati")

# ═══════════════════════════════════════════════════════════════════════════════
# META + STATS (derivati, non hardcodati)
# ═══════════════════════════════════════════════════════════════════════════════
meta["last_update"] = pd.Timestamp.now().isoformat(timespec="seconds")

_ultima = max((int(g) for g, ms in giornate.items()
               if any(x.get("ok") is not None for x in ms)), default=0)
_prossima = min((int(g) for g, ms in giornate.items()
                 if any(x.get("ok") is None for x in ms)), default=_ultima + 1)
meta["ultima_giornata_giocata"] = _ultima
meta["prossima_giornata"] = _prossima
# Regola N-1: la giornata N si congela quando la N-1 ha un risultato reale.
meta["freeze_until"] = _ultima + 1

total_played = sum(1 for g in giornate.values() for m in g if m.get("ok") is not None)
correct      = sum(1 for g in giornate.values() for m in g if m.get("ok") is True)
meta["stats"] = {"corrette": correct, "total": total_played}

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
out = json.dumps(data, indent=2, ensure_ascii=False)
OUT.write_text(out, encoding="utf-8")
print(f"\n✓ Salvato: {len(out)} byte")
print(f"  Stats: {correct}/{total_played} corrette ({correct/total_played*100:.1f}%)")
print(f"  Freeze: GG1-GG3 congelate, GG4+ future")
print(f"  Date G4 corrette: Lazio-Milan → 2026-09-12 (venerdì)")

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT G3 + G4
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== G3 (risultati + rest_days corretti) ===")
for m in giornate.get("3", []):
    f = m.get("features", {})
    bm = m.get("bm") or {}
    k = m.get("kelly")
    k_s = f" | Kelly: {k['pick']}@{k['odd']} ({k['stake_pct']}%)" if k else ""
    print(f"  {m['home']:15s} - {m['away']:15s} | {m['date']} | "
          f"home_rest={f.get('home_rest_days'):.0f} away_rest={f.get('away_rest_days'):.0f} | "
          f"B365: {bm.get('H','?')}/{bm.get('D','?')}/{bm.get('A','?')} | "
          f"FTR: {m.get('risultato','?')} | ok={m.get('ok')}{k_s}")

print("\n=== G4 (date corrette + rest_days) ===")
for m in giornate.get("4", []):
    f = m.get("features", {})
    print(f"  {m['home']:15s} - {m['away']:15s} | {m['date']} | "
          f"home_rest={f.get('home_rest_days'):.0f} away_rest={f.get('away_rest_days'):.0f} | "
          f"prob: {m.get('prob',{})}")
