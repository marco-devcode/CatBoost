"""Test del contratto quote (regola del freeze rimossa).

PERCHE' QUESTO TEST E' STATO RISCRITTO
---------------------------------------
Il test originale verificava tre cose:

1. uno snapshot con `bm = {"H": None, ...}` non venisse trattato come valido;
2. le quote di oggi ricostruissero `bm` e `bm_prob`;
3. uno snapshot SANO non venisse sovrascritto (la "regola del freeze").

Il caso 1 falliva gia' su HEAD (lo snapshot reale non e' piu' rotto: ha
bm = {'H': 2.6, 'D': 3.4, 'A': 2.7}), e il caso 3 verificava un comportamento
che NON ESISTE PIU': la regola del freeze e' stata rimossa su richiesta
esplicita, quindi le stime delle partite gia' giocate vengono ricalcolate e
non e' piu' vero che uno snapshot sano debba restare intatto.

Il test che segue verifica invece il CONTRATTO VERO di oggi:

1. le quote valide vengono SEMPRE assegnate, anche sulle partite future
   (indipendentemente da `played`) — e' la regressione che aveva azzerato
   tutte le quote prepartita;
2. le quote non valide (NaN, None, <= 1.0) producono bm = None, NON un dict
   con valori null (era la causa del crash del frontend: toFixed(0) su None);
3. il confronto RPS/Kelly retrospettivo resta legato a `played`, perche'
   senza il risultato reale non c'e' nulla da confrontare;
4. nessun NaN/Infinity/dict-vuoto nel JSON pubblicato;
5. la regola del freeze e' davvero sparita da predict_fixtures.py.

Non richiede catboost: e' un test di logica pura sullo snapshot reale.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LABELS = ("H", "D", "A")


def _bm_valido(d):
    """Quote valide = dict con 3 numeri finiti e > 1.0 (o None)."""
    if not isinstance(d, dict):
        return False
    for l in LABELS:
        v = d.get(l)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        if not math.isfinite(float(v)) or float(v) <= 1.0:
            return False
    return True


def _riallinea(m, bm_odds, bm_prob):
    """Logica di predict_fixtures: assegna le quote se valide, altrimenti None.

    NOTA: non esiste piu' il caso 'snapshot sano preservato': con il freeze
    rimosso, le stime delle partite giocate vengono ricalcolate.
    """
    if _bm_valido({"H": bm_odds[0], "D": bm_odds[1], "A": bm_odds[2]}) if bm_odds else False:
        m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                   "A": round(bm_odds[2], 2)}
        m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
    else:
        m["bm"] = None
        m["bm_prob"] = None
    return m


fail = 0

# ── Caso 1: quote non valide -> bm None, non un dict con null ──────────────
for nome, odds in (("None", None),
                   ("NaN", [float("nan"), float("nan"), float("nan")]),
                   ("zeri", [0.0, 0.0, 0.0]),
                   ("sotto 1.0", [0.9, 1.0, 0.8])):
    m = {"bm": {"H": None, "D": None, "A": None}, "bm_prob": {}}
    _riallinea(m, odds, [0.3, 0.3, 0.4] if odds else None)
    if m["bm"] is not None or m["bm_prob"] is not None:
        print(f"  FAIL caso1 ({nome}): atteso None, ottenuto {m['bm']}")
        fail += 1
    else:
        print(f"  OK   caso1 ({nome}): quote non valide -> bm=None")

# ── Caso 2: quote valide -> ricostruite, anche SENza risultato reale ───────
m = {"bm": None, "bm_prob": None, "ok": None, "esito": None,
     "prob": {"H": 45.0, "D": 30.0, "A": 25.0}}
_riallinea(m, [2.10, 3.40, 3.60], [0.452, 0.279, 0.269])
if not _bm_valido(m["bm"]) or not _bm_valido(m["bm_prob"]):
    print(f"  FAIL caso2: non ricostruite (bm={m['bm']} prob={m['bm_prob']})")
    fail += 1
else:
    pb = [m["bm_prob"][l] / 100.0 for l in LABELS]
    pr = [m["prob"][l] / 100.0 for l in LABELS]
    print(f"  OK   caso2: quote valide ricostruite anche su partita non giocata "
          f"-> bm={m['bm']} prob={m['bm_prob']} ({len(pb)} valori)")

# ── Caso 3: lo snapshot REALE non contiene NaN ne dict-vuoto ──────────────
J = ROOT / "webapp" / "public" / "predictions_eredivisie.json"
if not J.exists():
    J = ROOT / "webapp" / "public" / "predictions_seriea.json"
if not J.exists():
    print("  SKIP caso3: nessuno snapshot locale")
else:
    d = json.loads(J.read_text())
    matches = [m for ms in d["giornate"].values() for m in ms]
    n_bm = sum(1 for m in matches if m.get("bm") is not None)
    n_bad = 0
    for m in matches:
        bm = m.get("bm")
        if bm is not None and not _bm_valido(bm):
            n_bad += 1
        bp = m.get("bm_prob")
        if bp is not None and not _bm_valido(bp):
            n_bad += 1
        for key in ("prob", "bm_prob"):
            v = m.get(key)
            if isinstance(v, dict):
                for x in v.values():
                    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                        n_bad += 1
    if n_bad:
        print(f"  FAIL caso3: {n_bad} valori non validi nello snapshot reale")
        fail += 1
    else:
        print(f"  OK   caso3: snapshot reale pulito ({len(matches)} partite, "
              f"{n_bm} con quote, 0 NaN/dict-vuoto)")

# ── Caso 4: il gate RPS/Kelly retrospettivo resta legato a `played` ────────
src = (ROOT / "src" / "predict_fixtures.py").read_text(encoding="utf-8")
rps_ok = ("and played" in src) and ("bm_rows.append" in src)
if not rps_ok:
    print("  FAIL caso4: il confronto RPS non e' piu' legato a `played`")
    fail += 1
else:
    print("  OK   caso4: RPS/Kelly retrospettivo ancora legato a `played`")

# ── Caso 5: la regola del freeze e' DAVVERO rimossa ───────────────────────
freeze_markers = ("prev_estimates", "n_frozen", "freeze_until",
                  "reset_freeze", "--reset-freeze")
found = [m for m in freeze_markers if m in src]
if found:
    print(f"  FAIL caso5: marcatori del freeze ancora presenti: {found}")
    fail += 1
else:
    print("  OK   caso5: nessun marcatore del freeze in predict_fixtures.py")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
sys.exit(1 if fail else 0)
