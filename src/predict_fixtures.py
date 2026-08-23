# src/predict_fixtures.py
"""
Genera webapp/public/predictions.json per la dashboard Serie A 26/27.

Fonti calendario (in ordine di preferenza):
  1. football-data.co.uk  (CSV stagione I1)
  2. TheSportsDB          (API gratuita, eventi stagione 2026-2027)

Per ogni partita del calendario:
  - GIA GIOCATA  -> salva risultato reale + se la predizione era corretta (✅/❌)
  - FUTURA       -> calcola le feature (forma, Pi-Rating, H2H, riposo) e le % H/D/A
"""
import sys
import io
import json
import time
import unicodedata
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
WEBAPP_PUBLIC = ROOT / "webapp" / "public"
OUT_JSON = WEBAPP_PUBLIC / "predictions.json"

FIXTURES_URL = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"
# TheSportsDB: la chiave pubblica di test "123" restituisce gli eventi completi
# per-giornata via eventsround.php (il bulk eventsseason.php e' limitato a ~5-15).
TSDB_KEY = "123"
TSDB_LEAGUE_ID = "4332"   # Serie A
TSDB_ROUNDS = 38
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
SEASON_FD = "2627"        # formato football-data
SEASON_TSDB = "2026-2027" # formato TheSportsDB

MODEL_PATH = MODELS_DIR / "catboost_seriea_v1.cbm"

# Nomi squadre TheSportsDB -> nomi storico football-data
ALIASES = {
    "internazionale": "Inter", "inter milan": "Inter", "fc internazionale": "Inter",
    "ac milan": "Milan", "milan fc": "Milan",
    "ssc napoli": "Napoli", "napoli fc": "Napoli",
    "ss lazio": "Lazio", "lazio fc": "Lazio",
    "as roma": "Roma", "roma fc": "Roma",
    "acf fiorentina": "Fiorentina", "fiorentina fc": "Fiorentina",
    "atalanta bc": "Atalanta", "atalanta fc": "Atalanta",
    "bologna fc": "Bologna", "bologna fc 1909": "Bologna",
    "genoa cfc": "Genoa", "genoa cricket club": "Genoa",
    "torino fc": "Torino",
    "udinese calcio": "Udinese",
    "cagliari calcio": "Cagliari",
    "parma calcio": "Parma", "parma fc": "Parma",
    "hellas verona": "Verona", "verona fc": "Verona",
    "us lecce": "Lecce", "lecce fc": "Lecce",
    "empoli fc": "Empoli",
    "frosinone calcio": "Frosinone",
    "venezia fc": "Venezia", "venezia calcio": "Venezia",
    "ac monza": "Monza", "monza fc": "Monza",
    "us sassuolo calcio": "Sassuolo", "sassuolo calcio": "Sassuolo",
    "como 1907": "Como",
    "pisa sporting club": "Pisa", "ac pisa": "Pisa", "pisa sc": "Pisa",
    "cremonese": "Cremonese", "us cremonese": "Cremonese",
    "spezia calcio": "Spezia", "us spezia": "Spezia",
    "juventus fc": "Juventus", "juventus": "Juventus",
}


def norm_team(name: str) -> str:
    """Normalizza il nome squadra al formato dello storico football-data."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = " ".join(s.lower().strip().split())
    if s in ALIASES:
        return ALIASES[s]
    # togli prefissi/suffissi comuni e anno di fondazione
    stop = {"fc", "calcio", "ac", "as", "ss", "ssc", "us", "asd", "bc",
            "cfc", "sc", "spa", "1907", "1909", "1893", "1899", "1926"}
    tokens = [t for t in s.split() if t not in stop]
    base = " ".join(tokens) if tokens else s
    return ALIASES.get(base, base.title())


def _http_get(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = 10 * attempt  # rate-limit: attesa lunga
                print(f"  [429 rate-limit] attendo {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [retry {attempt}/{retries}] {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
        except Exception as e:
            print(f"  [retry {attempt}/{retries}] {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


def fetch_football_data(season: str):
    """CSV football-data. None se non ancora disponibile."""
    raw = _http_get(FIXTURES_URL.format(season=season))
    if raw is None:
        return None
    try:
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines='skip', encoding='latin-1')
        return df if 'HomeTeam' in df.columns else None
    except Exception as e:
        print(f"  football-data parse error: {e}")
        return None


def _parse_events(events: list) -> list:
    """Converte eventi TheSportsDB in righe football-data-like."""
    rows = []
    for ev in events:
        home = norm_team(ev.get("strHomeTeam", ""))
        away = norm_team(ev.get("strAwayTeam", ""))
        date = ev.get("dateEvent")
        rnd = ev.get("intRound")
        if not home or not away or not date:
            continue
        hs, as_ = ev.get("intHomeScore"), ev.get("intAwayScore")
        fthg = ftag = ftr = np.nan
        if hs is not None and as_ is not None:
            try:
                fthg, ftag = int(hs), int(as_)
                ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
            except (ValueError, TypeError):
                pass
        rows.append({
            "Date": pd.to_datetime(date),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
            "giornata": int(rnd) if rnd else np.nan,
        })
    return rows


def fetch_thesportsdb(season: str):
    """Calendario completo: una chiamata eventsround.php per ogni giornata.
    La chiave di test '123' non espone il bulk stagionale completo."""
    def fetch_round(rnd: int):
        url = (f"https://www.thesportsdb.com/api/v1/json/{TSDB_KEY}"
               f"/eventsround.php?id={TSDB_LEAGUE_ID}&r={rnd}&s={season}")
        raw = _http_get(url)
        if raw is None:
            return []
        try:
            payload = json.loads(raw.decode('utf-8'))
        except Exception as e:
            print(f"  [gg {rnd}] json error: {e}")
            return []
        return _parse_events(payload.get("events") or [])

    all_rows = []
    missed = []
    for rnd in range(1, TSDB_ROUNDS + 1):
        rows = fetch_round(rnd)
        if not rows:
            missed.append(rnd)
        all_rows.extend(rows)
        time.sleep(0.6)  # rate-limit gentile

    # Seconda passata per le giornate mancate (di solito 429 transitori)
    if missed:
        print(f"  Seconda passata per {len(missed)} giornate: {missed}")
        time.sleep(15)
        for rnd in missed:
            rows = fetch_round(rnd)
            all_rows.extend(rows)
            time.sleep(1.5)

    df = pd.DataFrame(all_rows)
    if len(df):
        # dedup (eventi possono comparire in piu' chiamate)
        df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"  TheSportsDB: {len(df)} partite su {df['giornata'].nunique()} giornate")
    return df if len(df) else None


def assign_giornate(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering cronologico in giornate (solo se la fonte non ha il round)."""
    df = df.sort_values('Date').reset_index(drop=True)
    rounds, cur, size, prev = [], 1, 0, None
    for d in df['Date']:
        if (prev is not None and (d - prev).days > 4) or size >= 10:
            cur += 1
            size = 0
        rounds.append(cur)
        size += 1
        prev = d
    df['giornata'] = rounds
    return df


def main():
    print("=" * 55)
    print(f"PREDIZIONI FIXTURE Serie A {SEASON_FD[:2]}/{SEASON_FD[2:]}")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} non trovato: esegui prima src/model.py")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    sys.path.insert(0, str(ROOT / 'src'))
    from predict import load_history, compute_features_for_match, FEATURE_ORDER

    history = load_history()
    print(f"Storico: {len(history)} partite")

    # ── Calendario: football-data prima, TheSportsDB fallback ─────────
    fx, source = None, None
    fd = fetch_football_data(SEASON_FD)
    if fd is not None and len(fd):
        fx, source = fd, "football-data.co.uk"
    else:
        print("  football-data non disponibile -> fallback TheSportsDB")
        tsdb = fetch_thesportsdb(SEASON_TSDB)
        if tsdb is not None:
            fx, source = tsdb, "thesportsdb.com"

    if fx is None or not len(fx):
        print("Nessuna fonte calendario disponibile: scrivo JSON vuoto.")
        WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, 'w') as f:
            json.dump({
                "meta": {
                    "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "stagione": SEASON_FD,
                    "nota": "Calendario non ancora disponibile dalle fonti",
                },
                "stats": {"corrette": 0, "total": 0},
                "giornate": {},
            }, f, indent=2, ensure_ascii=False)
        return

    print(f"Fonte calendario: {source}")
    fx['Date'] = pd.to_datetime(fx['Date'])
    fx = fx.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    if 'giornata' not in fx.columns or fx['giornata'].isna().any():
        fx = assign_giornate(fx.drop(columns=['giornata'], errors='ignore'))
    fx['giornata'] = fx['giornata'].astype(int)
    print(f"Fixture: {len(fx)} partite, {fx['giornata'].max()} giornate")

    # ── Predizioni ─────────────────────────────────────────────────────
    LABELS = ['H', 'D', 'A']
    giornate: dict[str, list] = {}
    for _, row in fx.iterrows():
        g = str(row['giornata'])
        m = {"home": row['HomeTeam'], "away": row['AwayTeam'],
             "date": row['Date'].strftime("%Y-%m-%d")}
        feats = compute_features_for_match(history, m["home"], m["away"], row['Date'])
        X = pd.DataFrame([{c: feats.get(c, 0.0) for c in FEATURE_ORDER}])
        probs = model.predict_proba(X)[0]
        m["prob"] = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
        m["pred"] = LABELS[int(np.argmax(probs))]

        ftr = row.get('FTR')
        if isinstance(ftr, str) and ftr.strip().upper() in ('H', 'D', 'A'):
            ftr = ftr.strip().upper()
            m["risultato"] = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
            m["esito"] = ftr
            m["ok"] = bool(m["pred"] == ftr)
        giornate.setdefault(g, []).append(m)

    tutte = [m for ms in giornate.values() for m in ms]
    giocate = [m for m in tutte if m.get("ok") is not None]
    corrette = sum(1 for m in giocate if m["ok"])
    ultima_gg = max((int(g) for g, ms in giornate.items()
                     if any(m.get("ok") is not None for m in ms)), default=0)
    prossima_gg = min((int(g) for g, ms in giornate.items()
                       if any(m.get("ok") is None for m in ms)), default=ultima_gg + 1)

    out = {
        "meta": {
            "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
            "stagione": SEASON_FD,
            "fonte": source,
            "ultima_giornata_giocata": ultima_gg,
            "prossima_giornata": prossima_gg,
        },
        "stats": {"corrette": corrette, "total": len(giocate)},
        "giornate": dict(sorted(giornate.items(), key=lambda kv: int(kv[0]))),
    }

    WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nGiornate: {len(giornate)} | Giocate: {len(giocate)} "
          f"| Corrette: {corrette} ({(corrette/max(len(giocate),1)*100):.0f}%)")
    print(f"Salvato: {OUT_JSON}")


if __name__ == "__main__":
    main()
