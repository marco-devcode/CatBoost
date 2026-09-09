# src/download_data.py
"""
Scarica le stagioni Serie A da football-data.co.uk e le salva concatenate in
data/serie_a_raw.csv.

Formato football-data.co.uk:
  - Serie A = codice I1
  - URL: https://www.football-data.co.uk/mmz4281/{season}/I1.csv
  - Colonne utili: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season,
    B365H/B365D/B365A (quote), HST/AST (tiri in porta)

Le stagioni sono nel formato YY1YY2 (es. 2425 = 2024/25).
"""
import sys
import time
import pandas as pd
import urllib.request
import io
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "serie_a_raw.csv"

# Storico + nuove stagioni. Aggiungi qui sotto le stagioni future quando escono
# (es. 2627 per il 2026/27, se football-data le pubblica).
SEASONS = [
    '1011', '1112', '1213', '1314', '1415',
    '1516', '1617', '1718', '1819', '1920',
    '2021', '2122', '2223', '2324',
    '2425', '2526', '2627',
]

# Serie A = I1 su football-data.co.uk
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/124.0 Safari/537.36'
}


def fetch_season(season: str, retries: int = 3, delay: float = 3.0) -> pd.DataFrame:
    """Scarica un CSV con retry e User-Agent per evitare 10054/403."""
    url = BASE_URL.format(season=season)
    last = Exception("unknown error")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            df = pd.read_csv(io.BytesIO(raw), on_bad_lines='skip', encoding='latin-1')
            return df
        except Exception as exc:
            last = exc
            if attempt < retries:
                print(f"  [retry {attempt}/{retries}] {season}: {exc}")
                time.sleep(delay * attempt)
    raise last


def main():
    # Se il file raw esiste già e contiene partite della stagione 2026/27 (date >= 2026),
    # usalo direttamente e salta il download (evita 503 da football-data.co.uk)
    if OUT_PATH.exists():
        try:
            existing = pd.read_csv(OUT_PATH, dtype=str, low_memory=False)
            # Cerca date 2026 (formato DD/MM/YYYY o YYYY-MM-DD)
            if 'Date' in existing.columns:
                dates = existing['Date'].astype(str)
                has_2026 = dates.str.contains(r'2026|/26$').any()
                if has_2026:
                    print(f"  OK  File locale trovato con stagione 2026/27 ({len(existing)} righe) — salto download")
                    print(f"\\nTotale grezzo  : {len(existing)} righe")
                    print(f"Stagioni OK    : {sorted(existing['Div'].unique()) if 'Div' in existing.columns else 'N/A'}")
                    print(f"Salvato in     : {OUT_PATH}")
                    return
        except Exception as e:
            print(f"  WARN Impossibile leggere file locale: {e}, provo download...")

    dfs = []
    for s in SEASONS:
        try:
            df = fetch_season(s)
            # Normalizza la colonna stagione -> formato YY1YY2
            if 'Season' in df.columns:
                df['season'] = df['Season'].astype(str).str.replace('-', '', regex=False)
            else:
                df['season'] = s
            dfs.append(df)
            print(f"  OK  Stagione {s}: {len(df)} partite")
        except Exception as e:
            print(f"  ERR Stagione {s} fallita (forse non ancora disponibile): {e}")
        time.sleep(1.0)   # pausa cortesia tra stagioni

    if not dfs:
        raise RuntimeError("Nessuna stagione scaricata. Controlla la connessione.")

    serie_a = pd.concat(dfs, ignore_index=True)
    serie_a.to_csv(OUT_PATH, index=False)
    print(f"\nTotale grezzo  : {len(serie_a)} righe")
    print(f"Stagioni OK    : {sorted(serie_a['season'].unique())}")
    print(f"Salvato in     : {OUT_PATH}")


if __name__ == "__main__":
    print("=== Download Serie A da football-data.co.uk ===")
    main()
