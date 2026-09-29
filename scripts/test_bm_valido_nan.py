"""Il caso che ha fatto fallire ogni fix precedente: NaN, non None.

`isinstance(nan, (int, float))` e' True, quindi il guard accettava NaN come
valido, `round(nan, 2)` restituiva nan, `_json_safe` lo convertiva in null
e il dict-vuoto ricompariva a ogni run. Solo `math.isfinite` lo scarta.

Il test esegue la definizione REALE estratta dal sorgente via ast, non una
copia: una copia puo' confermare un fix che nel sorgente non c'e', ed e'
esattamente quello che e' successo con le run precedenti.
"""
import ast
import math
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "predict_fixtures.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))

fn = next((n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "_bm_valid"), None)
if fn is None:
    print("FAIL: _bm_valido non trovata in predict_fixtures.py")
    raise SystemExit(1)

ns = {"math": math, "LABELS": ["H", "D", "A"]}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<sorgente>", "exec"), ns)
valido = ns["_bm_valid"]
print(f"_bm_valido estratta dal sorgente: {valido}")

fail = 0
casi = [
    ("dict-vuoto da JSON", {"H": None, "D": None, "A": None}, False),
    ("NaN (il caso vero)", {"H": float("nan"), "D": float("nan"), "A": float("nan")}, False),
    ("nan parziale", {"H": 1.9, "D": float("nan"), "A": 3.1}, False),
    ("inf", {"H": float("inf"), "D": 3.0, "A": 2.0}, False),
    ("valido", {"H": 1.9, "D": 3.4, "A": 4.1}, True),
    ("None", None, False),
    ("vuoto", {}, False),
    ("chiave mancante", {"H": 1.9, "D": 3.4}, False),
]
for nome, val, atteso in casi:
    got = valido(val)
    ok = got == atteso
    fail += 0 if ok else 1
    print(f"  {'OK  ' if ok else 'FAIL'} {nome:22} atteso={atteso!s:5} ottenuto={got}")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
raise SystemExit(1 if fail else 0)
