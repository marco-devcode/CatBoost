"""Ogni nome del CALENDARIO deve combaciare con un nome dello STORICO.

Il bug colto: attivando openfootball come fonte del calendario Serie A, i
nomi sono diversi da football-data ("Internazionale Milano" vs "Inter",
"Parma 1913" vs "Parma"). La chiave del freeze non combacia, la stessa
partita viene trattata come nuova e ricalcolata: `corrette` e' passato da
24/50 a 21/50 SENZA che nessuna partita cambiasse davvero.

Questo test e' la rete: prende il calendario vero (openfootball), applica
normalize() e verifica che ogni squadra risulti anche nello storico. Se
openfootball introduce un nome nuovo senza alias, fallisce qui e non dopo
una pipeline intera.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from leagues import LEAGUES, get_league  # noqa: E402
import sources  # noqa: E402

STORAGE = "https://raw.githubusercontent.com/openfootball/football.json/master"

fail = 0
for key, lg in LEAGUES.items():
    if not lg.openfootball_url:
        print(f"{key}: openfootball non configurato, skip")
        continue

    # 1. squadre del calendario openfootball, normalizzate
    # Se la fonte non risponde, il test NON deve fallire: un problema di
    # rete non e' un problema del codice, e far fallire la CI per quello
    # rende la pipeline inaffidabile. Si avvisa e si salta la lega.
    try:
        with urllib.request.urlopen(lg.openfootball_url, timeout=60) as r:
            cal = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  SKIP {key}: calendario non raggiungibile ({type(e).__name__}: {e})")
        continue
    matches = cal.get("matches", cal)
    cal_names = set()
    for m in matches:
        for side in ("team1", "team2"):
            t = m.get(side)
            if isinstance(t, dict):
                t = t.get("name")
            if t:
                cal_names.add(sources.norm_team(t, lg))

    # 2. squadre dello storico (data/serie_a_raw.csv o eredivisie_raw.csv)
    raw = ROOT / "data" / f"{lg.data_prefix}_raw.csv"
    hist = set()
    if raw.exists():
        import csv
        with raw.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for col in ("HomeTeam", "AwayTeam"):
                    v = (row.get(col) or "").strip()
                    if v:
                        hist.add(v)
    if not hist:
        print(f"{key}: storico assente in {raw.name}, skip il confronto")
        continue

    mancanti = sorted(cal_names - hist)
    print(f"\n{key}: {len(cal_names)} squadre nel calendario, "
          f"{len(hist)} nello storico")
    if mancanti:
        fail += 1
        print(f"  FAIL {len(mancanti)} squadre del calendario non nello storico:")
        for n in mancanti:
            print(f"        {n}")
        print("        -> aggiungere l'alias in leagues.py, altrimenti il "
              "freeze non riconosce la partita")
    else:
        print("  OK   tutte le squadre del calendario esistono nello storico")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
sys.exit(1 if fail else 0)
