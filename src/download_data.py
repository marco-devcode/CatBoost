# src/download_data.py
"""
Scarica TUTTE le stagioni Serie A da api.thestatsapi.com e le salva in
data/serie_a_raw.csv (formato: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, season).

Nessuna dipendenza da football-data.co.uk: tutto via thestatsapi.
Richiede il Bearer token in env THESTATSAPI_KEY (vedi https://api.thestatsapi.com/llms.txt).

Per recuperare anche lo storico più vecchio (es. 2010/11) basta che le stagioni
esistano su thestatsapi: il codice itera su TUTTE le stagioni della competition
(da quella più vecchia a quella corrente), non solo un elenco fisso.
"""
import sys
import os
import time
import json
import urllib.request
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "serie_a_raw.csv"

API_BASE = "https://api.thestatsapi.com/api"

# Stagioni "anno/i" da provare esplicitamente (oltre a quelle enumerate dall'API).
# Utile se l'API non espone lo storico più vecchio ma i match sono comunque
# raggiungibili per ID stagione noto. Lasciato vuoto = usa solo quelle dell'API.
EXTRA_SEASON_YEARS: list[str] = []


def _api_get(path: str, api_key: str, params=None) -> dict:
    from urllib.parse import urlencode
    url = f"{API_BASE}{path}"
    if params:
        url += ("&" if "?" in url else "?") + urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    last = Exception("unknown error")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(3 * attempt)
    raise last


def _norm_team(name: str) -> str:
    return (name or "").strip()


def _derive_ftr(gh, ga) -> str | None:
    if gh is None or ga is None:
        return None
    try:
        gh, ga = float(gh), float(ga)
    except (TypeError, ValueError):
        return None
    if gh > ga:
        return "H"
    if gh < ga:
        return "A"
    return "D"


def find_competition(api_key: str) -> str | None:
    comps = _api_get("/football/competitions", api_key, {"search": "Serie A", "country": "Italy"})
    for c in comps.get("data", []):
        if c.get("country") == "Italy" and "serie a" in c.get("name", "").lower():
            return c["id"]
    for c in comps.get("data", []):
        if "serie a" in c.get("name", "").lower():
            return c["id"]
    return None


def fetch_season_matches(api_key: str, comp_id: str, season_id: str, year: str) -> list[dict]:
    rows = []
    page = 1
    while True:
        data = _api_get(
            "/football/matches",
            api_key,
            {"competition_id": comp_id, "season_id": season_id,
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
                gh_i, ga_i = int(float(gh)), int(float(ga))
            except (TypeError, ValueError):
                continue
            rows.append({
                "Date": (m.get("utc_date") or "")[:10],
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": gh_i,
                "FTAG": ga_i,
                "FTR": ftr,
                "season": year.replace("/", ""),
            })
        meta = data.get("meta", {})
        if page >= meta.get("total_pages", 1):
            break
        page += 1
        time.sleep(0.4)
    return rows


def main():
    api_key = os.environ.get("THESTATSAPI_KEY", "")
    if not api_key:
        raise RuntimeError(
            "THESTATSAPI_KEY non impostata. Esporta il Bearer token di "
            "https://api.thestatsapi.com prima di lanciare download_data.py"
        )

    print("=== Ricerca competition Serie A ===")
    comp_id = find_competition(api_key)
    if not comp_id:
        raise RuntimeError("Competition 'Serie A' non trovata su thestatsapi.")
    print(f"  Competition: {comp_id}")

    print("=== Elenco stagioni disponibili ===")
    seasons = _api_get(f"/football/competitions/{comp_id}/seasons", api_key)
    season_list = seasons.get("data", [])
    # Ordine cronologico (più vecchia -> corrente)
    season_list = sorted(season_list, key=lambda s: (s.get("start_year", 0), s.get("year", "")))
    print(f"  Stagioni trovate: {len(season_list)}")

    # Aggiungi eventuali stagioni extra (storico non enumerato dall'API)
    wanted = {s["id"]: s.get("year", "") for s in season_list}
    for y in EXTRA_SEASON_YEARS:
        # se non già presente, proviamo a usarla comunque come year (richiede ID)
        pass  # gli ID vanno scoperti tramite l'API; qui lasciato come estensione

    all_rows = []
    for s in season_list:
        sid = s["id"]
        year = s.get("year", "")
        print(f"  Recupero {year} ({sid})...")
        try:
            rows = fetch_season_matches(api_key, comp_id, sid, year)
            print(f"    OK  {year}: {len(rows)} partite")
            all_rows.extend(rows)
        except Exception as e:
            print(f"    ERR {year}: {e}")
        time.sleep(0.5)

    if not all_rows:
        raise RuntimeError("Nessuna partita recuperata da thestatsapi.")

    df = pd.DataFrame(all_rows)
    # Ordinamento per data (il Pi-Rating richiede ordine cronologico)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date", ascending=True, kind="mergesort").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nTotale grezzo  : {len(df)} righe")
    print(f"Stagioni      : {sorted(df['season'].unique())}")
    print(f"Periodo       : {df['Date'].min()} → {df['Date'].max()}")
    print(f"Salvato in     : {OUT_PATH}")


if __name__ == "__main__":
    main()
