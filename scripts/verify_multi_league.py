"""Verifica pre-push: allineamento feature, fonti, path per lega, freeze.

Non sostituisce la pipeline: e' il controllo veloce che gira in locale per
intercettare gli errori che la pipeline intercetterebbe solo dopo 10 minuti.
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # radice del repo
sys.path.insert(0, str(ROOT / "src"))
from leagues import LEAGUES, get_league  # noqa: E402

fail = []
ok = []


def check(cond, msg):
    (ok if cond else fail).append(msg)


# 1. py_compile su tutti gli script
r = subprocess.run([sys.executable, "-m", "py_compile"] +
                   [str(p) for p in sorted((ROOT / "src").glob("*.py"))],
                   capture_output=True, text=True)
check(r.returncode == 0, f"py_compile: {r.stderr.strip()[:120] or 'OK'}")

# 2. FEATURE_COLS == FEATURE_ORDER, senza importare predict (catboost manca)
def literal_from(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None

fc = literal_from(ROOT / "src" / "features.py", "FEATURE_COLS")
fo = literal_from(ROOT / "src" / "predict.py", "FEATURE_ORDER")
check(fc == fo, f"FEATURE_COLS == FEATURE_ORDER ({len(fc or [])} vs {len(fo or [])})")
check(len(fc or []) == 17, f"17 feature (trovate {len(fc or [])})")
check("rest_advantage" not in (fc or []),
      "rest_advantage esclusa (ridondante deterministica)")
check("home_promoted" in (fc or []) and "away_promoted" in (fc or []),
      "flag neopromossa presenti")

# 3. Nessun hardcode di lega residuo negli script.
#    compute_ratings.py e' lo script TANTUM del ricalcolo selettivo: resta
#    volutamente solo Serie A (allowlist di partite esplicita), quindi e'
#    escluso dal controllo come leagues.py.
TANTUM_OK = {"recompute_22era.py"}
import re
hard = []
for p in sorted((ROOT / "src").glob("*.py")):
    if p.name in ("leagues.py",) or p.name in TANTUM_OK:
        continue
    for i, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        code = line.split("#")[0]
        if re.search(r"catboost_seriea|\"serie_a_raw\"|'serie_a_raw'|/I1\.csv", code):
            hard.append(f"{p.name}:{i}")
check(not hard, f"nessun hardcode di lega fuori leagues.py ({hard})")

# 4. Path distinti per lega
seen = {}
for lg in LEAGUES.values():
    for kind, path in (("split", lg.split_path("X_train_v2")),
                       ("model", lg.model_path),
                       ("json", lg.json_path)):
        seen.setdefault(path, []).append(f"{lg.key}:{kind}")
dups = {p: v for p, v in seen.items() if len(v) > 1}
check(not dups, f"nessun path condiviso fra leghe ({list(dups.values())})")

# 5. Ogni lega ha il suo JSON su disco.
#    La Eredivisie nasce con la pipeline: il file non esiste finche' il
#    modello non e' stato addestrato in CI (il .cbm e' gitignored, quindi in
#    locale non si puo' generare il JSON). E' un'attesa, non un errore: senza
#    questa distinzione il check fallirebbe sempre in locale e insegnerebbe
#    a ignorarlo.
SERIE_A_JSON_MUST_EXIST = True
for lg in LEAGUES.values():
    exists = lg.json_path.exists()
    if lg.key == "seriea" or exists:
        check(exists, f"JSON presente: {lg.json_name}")
    else:
        ok.append(f"JSON non ancora generato: {lg.json_name} "
                  f"(atteso: la pipeline lo crea al primo run)")
    if exists:
        try:
            d = json.loads(lg.json_path.read_text())
            def boom(x):
                raise ValueError(f"NaN/Infinity non valido: {x}")
            json.loads(lg.json_path.read_text(), parse_constant=boom)
            check("giornate" in d and "stats" in d,
                  f"{lg.key}: struttura JSON ok ({len(d.get('giornate', {}))} giornate)")
        except Exception as e:
            fail.append(f"{lg.key}: JSON non rigoroso — {e}")

# 6. Serie A: freeze e storico non regressi
lg_a = get_league("seriea")
if lg_a.json_path.exists():
    a = json.loads(lg_a.json_path.read_text())
    check(a["stats"].get("total", 0) >= 50,
          f"Serie A: partite giocate >= 50 ({a['stats'].get('total')})")
    check(a["meta"].get("freeze_until", 0) >= 1,
          f"Serie A: freeze_until impostato ({a['meta'].get('freeze_until')})")

# 7. Eredivisie: dataset e split
lg_e = get_league("eredivisie")
if lg_e.split_path("X_train_v2").exists():
    import csv
    with open(lg_e.split_path("X_train_v2")) as f:
        rows = sum(1 for _ in f) - 1
    check(rows > 1000, f"Eredivisie: X_train_v2 ha {rows} righe (>1000)")

if lg_e.features_path.exists():
    import csv
    with open(lg_e.features_path) as f:
        header = f.readline().strip().split(",")
    for c in ("home_promoted", "away_promoted", "home_rest_days"):
        check(c in header, f"Eredivisie: colonna {c} nelle feature")

print(f"\n{'='*60}")
for m in ok:
    print(f"  OK    {m}")
for m in fail:
    print(f"  FAIL  {m}")
print(f"{'='*60}")
print(f"{len(ok)} ok, {len(fail)} falliti")
sys.exit(1 if fail else 0)
