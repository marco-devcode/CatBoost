#!/usr/bin/env python3
"""Verifica il Fractional Kelly Dinamico (src/kelly_dynamic.py).

Non richiede catboost ne' pandas: il modulo e' puro, quindi il test gira in
qualsiasi ambiente dove la pipeline gira.

Casi coperti (richiesti dalla specifica):
  - p = 0.60, O = 1.50  -> c_dynamic = 0.50
  - p = 0.25, O = 6.00  -> c_dynamic = 0.10 e troncamento al MAX_CAP 0.025
  - nessun valore restituito negativo o sopra il cap del suo scaglione

Esegui:  python3.14 scripts/test_kelly_dynamic.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kelly_dynamic import (  # noqa: E402
    KELLY_DYNAMIC_CONFIG,
    calculate_dynamic_kelly_stake,
    dynamic_kelly_breakdown,
    dynamic_kelly_max_cap,
    dynamic_kelly_multiplier,
)

failures = []


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label}: {got} (atteso {want})")
    if not ok:
        failures.append(label)


def check_true(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


print("== Configurazione ==")
check_true("soglie di quota", KELLY_DYNAMIC_CONFIG["odds_thresholds"] == (1.80, 3.00, 5.00),
           str(KELLY_DYNAMIC_CONFIG["odds_thresholds"]))
check_true("c_dynamic", KELLY_DYNAMIC_CONFIG["c_dynamic"] == (0.50, 0.35, 0.20, 0.10),
           str(KELLY_DYNAMIC_CONFIG["c_dynamic"]))
check_true("config immutabile",
           isinstance(KELLY_DYNAMIC_CONFIG, type(KELLY_DYNAMIC_CONFIG)) and
           not hasattr(KELLY_DYNAMIC_CONFIG, "__setitem__"))

print("\n== Scaglioni di c_dynamic ==")
for odds, want_c in [(1.01, 0.50), (1.50, 0.50), (1.80, 0.50), (1.81, 0.35),
                     (3.00, 0.35), (3.01, 0.20), (5.00, 0.20), (5.01, 0.10),
                     (6.00, 0.10), (20.0, 0.10)]:
    check(f"O={odds}", dynamic_kelly_multiplier(odds), want_c)

print("\n== Caso 1 della specifica: p=0.60, O=1.50 ==")
# NB: con O=1.50 il break-even e' p = 1/1.50 = 0.667, quindi p=0.60 e' SOTTO
# break-even: c_dynamic e' 0.50 come richiesto, ma lo stake e' 0 per il
# guardrail. E' il comportamento corretto, non un errore: la specifica chiede
# di verificare il moltiplicatore, non di puntare una scommessa senza edge.
bd = dynamic_kelly_breakdown(0.60, 1.50)
check("c_dynamic", bd["c_dynamic"], 0.50)
check("b = O-1", bd["b"], 0.50)
check("f_full", bd["f_full"], -0.20)
check("f_dynamic", bd["f_dynamic"], -0.10)
check("stake_fraction (guardrail)", bd["stake_fraction"], 0.0)
check_true("nessun troncamento", bd["capped"] is False)

# Sullo stesso scaglione (O <= 1.80 -> c=0.50) con edge reale: qui si vede
# il moltiplicatore applicato. p=0.75 -> f_full=0.25, *0.50 = 0.125, che
# supera il MAX_CAP 0.05 di quello scaglione -> troncato al 5%.
bd = dynamic_kelly_breakdown(0.75, 1.50)
check("c_dynamic", bd["c_dynamic"], 0.50)
check("f_full", bd["f_full"], 0.25)
check("f_dynamic", bd["f_dynamic"], 0.125)
check("max_cap dello scaglione", bd["max_cap"], 0.05)
check_true("troncato al MAX_CAP 5%", bd["capped"] is True)
check("stake_fraction", bd["stake_fraction"], 0.05)

print("\n== Caso 2 della specifica: p=0.25, O=6.00 ==")
bd = dynamic_kelly_breakdown(0.25, 6.00)
check("c_dynamic", bd["c_dynamic"], 0.10)
check("max_cap", bd["max_cap"], 0.025)
# f_full = (0.25*5 - 0.75)/5 = (1.25-0.75)/5 = 0.10 ; 0.10*0.10 = 0.01
check("f_full", bd["f_full"], 0.10)
check("f_dynamic", bd["f_dynamic"], 0.01)
check_true("NON troncato: 1% < cap 2.5%", bd["capped"] is False,
           f"f_dynamic={bd['f_dynamic']} cap={bd['max_cap']}")
# Il troncamento si vede forzando p alto sulla stessa quota.
bd = dynamic_kelly_breakdown(0.45, 6.00)
# f_full = (0.45*5 - 0.55)/5 = (2.25-0.55)/5 = 0.34 ; *0.10 = 0.034 > 0.025
check("f_dynamic sopra il cap", bd["f_dynamic"], 0.034)
check_true("troncato al MAX_CAP", bd["capped"] is True)
check("stake_fraction == cap", bd["stake_fraction"], 0.025)

print("\n== Invarianti: mai negativo, mai sopra il cap ==")
grid_p = [i / 100 for i in range(1, 100)]
grid_o = [1.01, 1.10, 1.50, 1.80, 1.81, 2.00, 2.50, 3.00, 3.01, 3.50,
          4.00, 5.00, 5.01, 6.00, 8.00, 15.0, 100.0]
negativi = sopraccapo = 0
for p in grid_p:
    for o in grid_o:
        stake = calculate_dynamic_kelly_stake(p, o)
        cap = dynamic_kelly_max_cap(o)
        if stake < 0.0:
            negativi += 1
        if stake > cap + 1e-12:
            sopraccapo += 1
check_true(f"griglia {len(grid_p)}x{len(grid_o)}: nessun negativo", negativi == 0,
           f"{negativi} casi")
check_true(f"griglia {len(grid_p)}x{len(grid_o)}: nessuno sopra il cap",
           sopraccapo == 0, f"{sopraccapo} casi")

print("\n== Casi limite e robustezza ==")
check_true("edge assente (p == p_bm) -> 0 puntata",
           calculate_dynamic_kelly_stake(0.25, 2.00) == 0.0)
check_true("quota <= 1 -> 0 puntata", calculate_dynamic_kelly_stake(0.90, 1.0) == 0.0)
check_true("p fuori dominio -> 0 puntata", calculate_dynamic_kelly_stake(1.5, 2.0) == 0.0)
check_true("input non numerico -> 0 puntata, nessuna eccezione",
           calculate_dynamic_kelly_stake("abc", None) == 0.0)  # type: ignore[arg-type]
check_true("stake sempre <= cap anche su p -> 1",
           all(calculate_dynamic_kelly_stake(0.999, o) <= dynamic_kelly_max_cap(o) + 1e-12
               for o in grid_o))

print()
if failures:
    print(f"KELLY DINAMICO: {len(failures)} CONTROLLI FALLITI -> {failures}")
    sys.exit(1)
print("KELLY DINAMICO: tutti i controlli passati.")