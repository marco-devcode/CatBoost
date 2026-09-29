"""Il caso di oggi: NaN che arriva dal CALENDARIO, non dallo snapshot.

openfootball non ha le colonne B365. Per una partita senza quote,
`bm_odds` arriva come [nan, nan, nan] invece che None. Il gate
`bm_odds is not None` lo accetta, costruisce il dict, round(nan, 2)
restituisce nan, _json_safe lo converte in null, e il frontend fa
toFixed(0) su None: pagina bianca.

Misurato: 310 partite future della Serie A con bm = {"H": null, ...}.

Il test esegue `_bm_valid` estratta dal sorgente e verifica i casi che
arrivano davvero dai dati, non sintattici.
"""
import ast
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "predict_fixtures.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))

fn = next((n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "_bm_valid"), None)
if fn is None:
    print("FAIL: _bm_valid non trovata a livello di modulo")
    sys.exit(1)

ns = {"math": math}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<sorgente>", "exec"), ns)
check = ns["_bm_valid"]
print("_bm_valid estratta dal sorgente (livello di modulo)")

NAN = float("nan")
fail = 0
casi = [
    # (nome, valore, atteso)
    ("openfootball senza quote", {"H": NAN, "D": NAN, "A": NAN}, False),
    ("NaN parziale", {"H": 1.9, "D": NAN, "A": 3.1}, False),
    ("dict None (lo snapshot rotto)", {"H": None, "D": None, "A": None}, False),
    ("quote vere da TheOddsAPI", {"H": 1.9, "D": 3.4, "A": 4.1}, True),
    ("quote intere", {"H": 2, "D": 3, "A": 4}, True),
    ("inf", {"H": float("inf"), "D": 3.0, "A": 2.0}, False),
    ("None intero", None, False),
    ("chiave mancante", {"H": 1.9}, False),
]
for nome, val, atteso in casi:
    got = check(val)
    ok = got == atteso
    fail += 0 if ok else 1
    print(f"  {'OK  ' if ok else 'FAIL'} {nome:30} atteso={atteso!s:5} ottenuto={got}")

# Il caso specifico di oggi: cosa succede se il gate usa `is not None`
bm_odds = [NAN, NAN, NAN]
gate_vecchio = bm_odds is not None
gate_nuovo = check({"H": bm_odds[0], "D": bm_odds[1], "A": bm_odds[2]})
print(f"\ngate vecchio (is not None) : {gate_vecchio}  <- accetta NaN, genera il dict-vuoto")
print(f"gate nuovo (_bm_valid)    : {gate_nuovo}  <- deve rifiutare NaN")
# fallisce SOLO se il gate nuovo accetta il NaN: il vecchio accettarlo e' il
# bug da cui siamo partiti, non una condizione da verificare.
if gate_nuovo:
    fail += 1
    print("  FAIL: _bm_valid accetta NaN, il dict-vuoto tornerebbe in circolo")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
sys.exit(1 if fail else 0)
