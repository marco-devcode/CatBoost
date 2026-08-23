# src/predict_fixtures.py
"""
Genera webapp/public/predictions.json per la dashboard Serie A 26/27.

Per ogni partita del calendario 26/27 (football-data.co.uk):
  - GIA GIOCATA  -> salva risultato reale + se la predizione era corretta (✅/❌)
  - FUTURA       -> calcola le feature (forma, Pi-Rating, H2H, riposo) e le % H/D/A

Le partite vengono raggruppate in GIORNATE con clustering cronologico
(nuova giornata se >4 giorni di distanza dalla precedente o 10 partite chiuse).
"""
import sys
import io
import json
import time
import urllib.request
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
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
SEASON = "2627"

MODEL_PATH = MODELS_DIR / "catboost_seriea_v1.cbm"


def fetch_fixtures(season: str, retries: int = 3):
    """Scarica il CSV della stagione. Ritorna None se non ancora disponibile."""
    url = FIXTURES_URL.format(season=season)
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
            return pd.read_csv(io.BytesIO(raw), on_bad_lines='skip', encoding='latin-1')
        except Exception as e:
            print(f"  [retry {attempt}/{retries}] {season}: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


def assign_giornate(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering cronologico in giornate: nuova giornata se gap > 4 giorni
    oppure 10 partite gia' assegnate."""
    df = df.sort_values('Date').reset_index(drop=True)
    rounds = []
    cur = 1
    size = 0
    prev_date = None
    for _, row in df.iterrows():
        d = row['Date']
        new_round = (
            (prev_date is not None and (d - prev_date).days > 4) or size >= 10
        )
        if new_round:
            cur += 1
            size = 0
        rounds.append(cur)
        size += 1
        prev_date = d
    df['giornata'] = rounds
    return df


def main():
    print("=" * 55)
    print(f"PREDIZIONI FIXTURE Serie A {SEASON[:2]}/{SEASON[2:]}")
    print("=" * 55)

    # ── Modello ────────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} non trovato: esegui prima src/model.py (step Train)"
        )
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    # ── Storico per le feature ─────────────────────────────────────────
    sys.path.insert(0, str(ROOT / 'src'))
    from predict import load_history, compute_features_for_match, FEATURE_ORDER

    history = load_history()
    print(f"Storico: {len(history)} partite")

    # ── Calendario ─────────────────────────────────────────────────────
    fx = fetch_fixtures(SEASON)
    if fx is None:
        print("Calendario non ancora disponibile: scrivo JSON vuoto.")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, 'w') as f:
            json.dump({
                "meta": {
                    "last_update": pd.Timestamp.now().isoformat(),
                    "stagione": SEASON,
                    "nota": "Calendario 26/27 non ancora pubblicato da football-data.co.uk",
                },
                "stats": {"corrette": 0, "total": 0},
                "giornate": {},
            }, f, indent=2, ensure_ascii=False)
        print(f"Salvato: {OUT_JSON}")
        return

    fx['Date'] = pd.to_datetime(fx['Date'], dayfirst=True, errors='coerce')
    fx = fx.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    fx = assign_giornate(fx)
    print(f"Fixture: {len(fx)} partite, {fx['giornata'].max()} giornate")

    # ── Predizioni ─────────────────────────────────────────────────────
    LABELS = ['H', 'D', 'A']
    giocatori = {}
    for _, row in fx.iterrows():
        g = str(row['giornata'])
        m = {
            "home": row['HomeTeam'],
            "away": row['AwayTeam'],
            "date": row['Date'].strftime("%Y-%m-%d"),
        }
        feats = compute_features_for_match(history, m["home"], m["away"], row['Date'])
        X = pd.DataFrame([{c: feats.get(c, 0.0) for c in FEATURE_ORDER}])
        probs = model.predict_proba(X)[0]
        m["prob"] = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
        m["pred"] = LABELS[int(np.argmax(probs))]

        # Partita giocata? (gol presenti)
        ftr = row.get('FTR')
        if isinstance(ftr, str) and ftr.strip().upper() in ('H', 'D', 'A'):
            ftr = ftr.strip().upper()
            m["risultato"] = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
            m["esito"] = ftr
            m["ok"] = bool(m["pred"] == ftr)
        giocatori.setdefault(g, []).append(m)

    # ── Meta + stats ───────────────────────────────────────────────────
    tutte = [m for ms in giocatori.values() for m in ms]
    giocate = [m for m in tutte if m.get("ok") is not None]
    corrette = sum(1 for m in giocate if m["ok"])
    ultima_gg = max((int(g) for g, ms in giocatori.items()
                     if any(m.get("ok") is not None for m in ms)), default=0)
    prossima_gg = min((int(g) for g, ms in giocatori.items()
                       if any(m.get("ok") is None for m in ms)), default=ultima_gg + 1)

    out = {
        "meta": {
            "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
            "stagione": SEASON,
            "ultima_giornata_giocata": ultima_gg,
            "prossima_giornata": prossima_gg,
        },
        "stats": {"corrette": corrette, "total": len(giocate)},
        "giornate": dict(sorted(giocatori.items(), key=lambda kv: int(kv[0]))),
    }

    WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nGiornate: {len(giocatori)} | Giocate: {len(giocate)} "
          f"| Corrette: {corrette} ({(corrette/max(len(giocate),1)*100):.0f}%)")
    print(f"Salvato: {OUT_JSON}")


if __name__ == "__main__":
    main()
