# src/download_data.py
"""
Scarica lo storico COMPLETO di un campionato da football-data.co.uk e lo
salva in data/<prefix>_raw.csv.

Formato football-data.co.uk (URL: .../mmz4281/{season}/{CODE}.csv):
  - Serie A = I1, Eredivisie = N1
  - Colonne utili: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, Season,
    B365H/B365D/B365A (quote), HST/AST (tiri in porta)

Le stagioni sono nel formato YY1YY2 (es. 2425 = 2024/25) e arrivano da
leagues.League.seasons: questo script non ha piu' hardcode di lega.

Uso:
    python src/download_data.py --league seriea
    python src/download_data.py --league eredivisie
"""
import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}


def fetch_season(url: str, season: str, retries: int = 3,
                 delay: float = 3.0) -> pd.DataFrame:
    """Scarica un CSV con retry e User-Agent per evitare 10054/403.

    Nota: www.football-data.co.uk risponde 302 verso football-data.co.uk.
    urllib segue i redirect di default, quindi NON disabilitarli: senza -L
    si scarica un file vuoto e si conclude che la fonte e' down.
    """
    last = Exception("unknown error")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            return pd.read_csv(io.BytesIO(raw), on_bad_lines="skip",
                               encoding="latin-1")
        except Exception as exc:
            last = exc
            if attempt < retries:
                print(f"  [retry {attempt}/{retries}] {season}: {exc}")
                time.sleep(delay * attempt)
    raise last


def main(league):
    # Lo storico completo e' la base del modello: addestrarlo sulle sole
    # partite della stagione corrente non basta a stimare i pesi su 17 feature
    # e produce previsioni arbitrarie. Quindi scarichiamo TUTTE le stagioni
    # elencate in League.seasons, ogni volta, senza short-circuit sul
    # contenuto del file locale (un guard di quel tipo ha troncato lo storico
    # da 5600 a 22 righe, in silenzio, due volte).
    dfs = []
    for s in league.seasons:
        try:
            df = fetch_season(league.fd_url.format(season=s), s)
            # Normalizza la colonna stagione -> formato YY1YY2
            if "Season" in df.columns:
                df["season"] = (df["Season"].astype(str)
                                .str.replace("-", "", regex=False))
            else:
                df["season"] = s
            dfs.append(df)
            print(f"  OK  Stagione {s}: {len(df)} partite")
        except Exception as e:
            print(f"  ERR Stagione {s} fallita "
                  f"(forse non ancora disponibile): {e}")
        time.sleep(0.4)   # pausa cortesia tra stagioni

    if not dfs:
        raise RuntimeError("Nessuna stagione scaricata. Controlla la connessione.")

    raw = pd.concat(dfs, ignore_index=True)
    raw.to_csv(league.raw_path, index=False)
    print(f"\nTotale grezzo  : {len(raw)} righe")
    print(f"Stagioni OK    : {sorted(raw['season'].unique())}")
    print(f"Salvato in     : {league.raw_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Download storico da football-data.co.uk")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    args = ap.parse_args()
    lg = get_league(args.league)
    print(f"=== Download {lg.name} ({lg.fd_code}) da football-data.co.uk ===")
    main(lg)
