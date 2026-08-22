# src/download_data.py
"""
Scarica le stagioni Serie A e le salva concatenate in data/serie_a_raw.csv.

Pipeline dati (due fonti):
  1. Storico 2010-2024  -> football-data.co.uk  (CSV gratuiti, codice I1)
  2. Nuove stagioni 2024/25, 2025/26, 2026/27 -> api.thestatsapi.com
     (require Bearer token in env THESTATSAPI_KEY; vedi llms.txt)

Output: colonne Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, season
(formato identico a clean_data.py / ratings.py / features.py).
"""
import sys
import os
import time
import json
import urllib.request
import urllib.error
import io
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "serie_a_raw.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1) STORICO football-data.co.uk (2010-2024)
# ─────────────────────────────────────────────────────────────────────────────
HIST_SEASONS = [
    '1011', '1112', '1213', '1314', '1415',
    '1516', '1617', '1718', '1819', '1920',
    '2021', '2122', '2223', '2324',
]
HIST_BASE = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"
HIST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
}

def _fetch_url(url, headers=None, retries=3, delay=3.0):
    last = Exception("unknown error")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # network/HTTP
            last = exc
            if attempt < retries:
                time.sleep(delay * attempt)
    raise last


def load_historical() -> pd.DataFrame:
    """Scarica 2010-2024 da football-data.co.uk."""
    dfs = []
    for s in HIST_SEASONS:
        try:
            raw = _fetch_url(HIST_BASE.format(season=s), HIST_HEADERS)
            df = pd.read_csv(io.BytesIO(raw), on_bad_lines='skip', encoding='latin-1')
            if 'Season' in df.columns:
                df['season'] = df['Season'].astype(str)
            else:
                df['season'] = s
            dfs.append(df)
            print(f"  OK  Storico {s}: {len(df)} partite")
        except Exception as e:
            print(f"  ERR Storico {s} fallita: {e}")
        time.sleep(1.0)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2) NUOVE STAGIONI via api.thestatsapi.com
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "https://api.thestatsapi.com/api"
TARGET_SEASONS = ["24/25", "25/26", "26/27"]  # stagioni da recuperare via API


def _api_get(path: str, api_key: str, params=None) -> dict:
    url = f"{API_BASE}{path}"
    if params:
        from urllib.parse import urlencode
        url += ("&" if "?" in url else "?") + urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    raw = _fetch_url(url, {"Authorization": f"Bearer {api_key}"}, retries=3, delay=3.0)
    return json.loads(raw.decode("utf-8"))


def _norm_team(name: str) -> str:
    """Normalizza i nomi squadra per allinearsi allo storico football-data."""
    return (name or "").strip()


def _derive_ftr(score_home, score_away) -> str:
    if score_home is None or score_away is None:
        return None
    if score_home > score_away:
        return "H"
    if score_home < score_away:
        return "A"
    return "D"


def load_from_api(api_key: str) -> pd.DataFrame:
    """Recupera le nuove stagioni da thestatsapi e le mappa nel formato standard."""
    if not api_key:
        print("  [SKIP] THESTATSAPI_KEY non impostata: nessuna nuova stagione recuperata.")
        return pd.DataFrame()

    # Trova competition "Serie A" (Italia)
    comps = _api_get("/football/competitions", api_key, {"search": "Serie A", "country": "Italy"})
    comp_id = None
    for c in comps.get("data", []):
        if c.get("country") == "Italy" and "serie a" in c.get("name", "").lower():
            comp_id = c["id"]
            break
    if not comp_id:
        # fallback: primo match contenente 'Serie A'
        for c in comps.get("data", []):
            if "serie a" in c.get("name", "").lower():
                comp_id = c["id"]
                break
    if not comp_id:
        print("  [ERR] Competition 'Serie A' non trovata su thestatsapi.")
        return pd.DataFrame()

    print(f"  Competition Serie A: {comp_id}")

    seasons = _api_get(f"/football/competitions/{comp_id}/seasons", api_key)
    season_map = {s["year"]: s["id"] for s in seasons.get("data", [])}

    rows = []
    for target in TARGET_SEASONS:
        sid = season_map.get(target)
        if not sid:
            print(f"  [SKIP] Stagione {target} non disponibile su thestatsapi (ancora non giocata?).")
            continue
        print(f"  Recupero stagione {target} ({sid})...")
        page = 1
        collected = 0
        while True:
            data = _api_get(
                f"/football/matches",
                api_key,
                {"competition_id": comp_id, "season_id": sid,
                 "status": "finished", "per_page": 100, "page": page},
            )
            matches = data.get("data", [])
            if not matches:
                break
            for m in matches:
                h = _norm_team(m.get("home_team", {}).get("name"))
                a = _norm_team(m.get("away_team", {}).get("name"))
                sc = m.get("score", {}) or {}
                gh = sc.get("home")
                ga = sc.get("away")
                if gh is None and "final_score" in sc:
                    gh = sc["final_score"].get("home")
                    ga = sc["final_score"].get("away")
                ftr = _derive_ftr(gh, ga)
                if not h or not a or ftr is None or gh is None or ga is None:
                    continue
                try:
                    gh_i, ga_i = int(gh), int(ga)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "Date": m.get("utc_date", "")[:10],
                    "HomeTeam": h,
                    "AwayTeam": a,
                    "FTHG": gh_i,
                    "FTAG": ga_i,
                    "FTR": ftr,
                    "season": target.replace("/", ""),
                })
                collected += 1
            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.5)
        print(f"    OK  {target}: {collected} partite")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    frames = []

    print("=== 1) Storico football-data.co.uk (2010-2024) ===")
    hist = load_historical()
    if len(hist):
        frames.append(hist)
        print(f"  Storico totale: {len(hist)} righe")

    print("\n=== 2) Nuove stagioni via thestatsapi ===")
    api_key = os.environ.get("THESTATSAPI_KEY", "")
    api_df = load_from_api(api_key)
    if len(api_df):
        frames.append(api_df)
        print(f"  API totale: {len(api_df)} righe")

    if not frames:
        raise RuntimeError("Nessun dato scaricato. Controlla connessione / API key.")

    serie_a = pd.concat(frames, ignore_index=True)
    serie_a.to_csv(OUT_PATH, index=False)
    print(f"\nTotale grezzo  : {len(serie_a)} righe")
    print(f"Stagioni      : {sorted(serie_a['season'].unique())}")
    print(f"Salvato in     : {OUT_PATH}")


if __name__ == "__main__":
    main()
