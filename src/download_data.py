# src/download_data.py
"""
Scarica le stagioni Serie A 2010-2024 da football-data.co.uk
e le salva concatenate in data/serie_a_raw.csv
"""
import sys
import time
import pandas as pd
import urllib.request
import io
from pathlib import Path

# Forza UTF-8 in output (necessario su Windows con encoding cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# --- Percorsi robusti indipendenti dalla CWD ---
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "serie_a_raw.csv"

SEASONS = [
    '1011', '1112', '1213', '1314', '1415',
    '1516', '1617', '1718', '1819', '1920',
    '2021', '2122', '2223', '2324'
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0 Safari/537.36'
}

def fetch_season(season: str, retries: int = 3, delay: float = 3.0) -> pd.DataFrame:
    """Scarica un CSV con retry e User-Agent per evitare 10054/403."""
    url = BASE_URL.format(season=season)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            df = pd.read_csv(io.BytesIO(raw), on_bad_lines='skip', encoding='latin-1')
            return df
        except Exception as exc:
            if attempt < retries:
                print(f"  [retry {attempt}/{retries}] {season}: {exc}")
                time.sleep(delay * attempt)
            else:
                raise

dfs = []
for s in SEASONS:
    try:
        df = fetch_season(s)
        df['season'] = s
        dfs.append(df)
        print(f"  OK  Stagione {s}: {len(df)} partite")
    except Exception as e:
        print(f"  ERR Stagione {s} fallita: {e}")
    time.sleep(1.0)   # pausa cortesia tra stagioni

if not dfs:
    raise RuntimeError("Nessuna stagione scaricata. Controlla la connessione.")

serie_a = pd.concat(dfs, ignore_index=True)
serie_a.to_csv(OUT_PATH, index=False)
print(f"\nTotale grezzo  : {len(serie_a)} righe")
print(f"Stagioni OK    : {len(dfs)}/{len(SEASONS)}")
print(f"Salvato in     : {OUT_PATH}")