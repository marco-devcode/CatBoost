"""Prova end-to-end del fix su predict_fixtures, senza catboost.

Non puo' girare predict_fixtures vero (importa catboost, che in questo
ambiente non c'e'), quindi isola la logica di freeze/quote che ha causato il
crash e la esercita sullo snapshot REALE rotto, verificando che:

1. lo snapshot con bm = {"H": None, ...} non venga trattato come valido
2. le quote di oggi ricostruiscano bm e bm_prob quando disponibili
3. senza fonti di oggi, la partita resti senza quote invece di far crashare
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LABELS = ("H", "D", "A")


def _bm_valido(d):
    return (isinstance(d, dict)
            and all(isinstance(d.get(l), (int, float)) for l in LABELS))


def _riallinea(m, bm_odds, bm_prob):
    """Copia della logica inserita in predict_fixtures (ramo freeze)."""
    if not _bm_valido(m.get("bm")) and bm_odds is not None:
        m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                   "A": round(bm_odds[2], 2)}
    if not _bm_valido(m.get("bm_prob")) and bm_prob is not None:
        m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
    if not _bm_valido(m.get("bm")) or not _bm_valido(m.get("bm_prob")):
        m["bm"] = None
        m["bm_prob"] = None
    return m


def _entra_nel_ramo(m):
    return (m["bm"] is not None and m["ok"] is not None and m["esito"] is not None
            and all(m["prob"].get(l) is not None for l in LABELS))


J = ROOT / "webapp" / "public" / "predictions_eredivisie.json"
if not J.exists():
    print("SKIP: nessuno snapshot locale")
    sys.exit(0)
d = json.loads(J.read_text())
played = [m for ms in d["giornate"].values() for m in ms if m.get("ok") is not None]
print(f"snapshot: {len(played)} giocate")

fail = 0

# Caso 1: snapshot rotto, nessuna fonte odierna -> resta senza quote, NO crash
m = dict(played[0])
prima = json.dumps(m.get("bm"))
_riallinea(m, None, None)
if m["bm"] is not None:
    print(f"  FAIL caso1: bm rotto non ripulito ({m['bm']})")
    fail += 1
elif _entra_nel_ramo(m):
    print("  FAIL caso1: entra nel ramo RPS con bm=None")
    fail += 1
else:
    print(f"  OK caso1: bm rotto ({prima}) ripulito, escluso dal ramo RPS")

# Caso 2: snapshot rotto + quote di oggi -> ricostruite
m = dict(played[0])
_riallinea(m, [2.10, 3.40, 3.60], [0.452, 0.279, 0.269])
if not _bm_valido(m["bm"]) or not _bm_valido(m["bm_prob"]):
    print(f"  FAIL caso2: non ricostruite (bm={m['bm']} prob={m['bm_prob']})")
    fail += 1
elif not _entra_nel_ramo(m):
    print("  FAIL caso2: non entra nel ramo RPS nonostante i dati")
    fail += 1
else:
    # questo e' il punto che prima crashava con None/100.0
    pb = [m["bm_prob"][l] / 100.0 for l in LABELS]
    pr = [m["prob"][l] / 100.0 for l in LABELS]
    print(f"  OK caso2: ricostruite bm={m['bm']} prob={m['bm_prob']} -> {len(pb)} valori")

# Caso 3: snapshot sano non viene toccato
m = dict(played[0])
m["bm"] = {"H": 1.9, "D": 3.5, "A": 4.0}
m["bm_prob"] = {"H": 50.0, "D": 27.0, "A": 23.0}
snap = json.dumps(m["bm"])
_riallinea(m, [9.9, 9.9, 9.9], [0.1, 0.1, 0.1])
if json.dumps(m["bm"]) != snap:
    print(f"  FAIL caso3: snapshot sano sovrascritto ({m['bm']})")
    fail += 1
else:
    print("  OK caso3: snapshot sano preservato (regola del freeze)")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
sys.exit(1 if fail else 0)
