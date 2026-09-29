"""Verifica che le funzioni chiamate in predict_fixtures siano raggiungibili.

Il bug di oggi: il gate del ramo future chiamava `_bm_valido` (con la o),
definita SOLO dentro il ramo snapshot. Nel ramo future era un NameError,
intercettato, e la pipeline restava verde producendo 310 dict-vuoto.

Un'analisi di scope completa e' pesante; qui si fa il caso utile: ogni
funzione chiamata con un nome che SOMIGLIA a una def di modulo deve essere
usata solo dove quella def e' visibile. Concreto: nessuna def annidata deve
essere chiamata da fuori dal proprio scope, e nessuna chiamata puo' usare un
nome che e' una def annidata se il chiamante e' fuori dallo scope che la
definisce.

Semplificato e mirato: segnala ogni chiamata a una def ANNIDATA il cui nome
differisce di 1 char da una def di modulo (errore di battitura), e verifica
che ogni def di modulo chiamata esista.
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "predict_fixtures.py"
tree = ast.parse(SRC.read_text(encoding="utf-8"))

module_defs = {n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
local_defs = {n.name for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} - module_defs

# 1. funzioni di modulo chiamate ma non esistenti
all_called = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        all_called.add(node.func.id)

helper_like = {c for c in all_called
               if c.startswith("_") and c.islower() and c not in ("__init__",)}
undefined_helpers = {c for c in helper_like if c not in module_defs and c not in local_defs}

# 2. def annidata chiamata con nome vicino a una di modulo
typos = []
for ld in sorted(local_defs):
    for md in sorted(module_defs):
        if ld != md and len(ld) == len(md) and sum(a != b for a, b in zip(ld, md)) == 1:
            typos.append((ld, md))

fail = 0
print(f"def di modulo: {len(module_defs)}")
print(f"def annidate : {len(local_defs)} ({sorted(local_defs)})")

if undefined_helpers:
    fail += 1
    print(f"\n  FAIL helper chiamati ma non definiti: {sorted(undefined_helpers)}")
else:
    print("\n  OK   tutti gli helper chiamati sono definiti")

if typos:
    fail += 1
    print(f"  FAIL def annidate con nome a 1 char da una di modulo:")
    for ld, md in typos:
        print(f"         {ld!r} vs {md!r} — probabile errore di battitura")
else:
    print("  OK   nessuna def locale sospetta (nome a 1 char da una di modulo)")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
sys.exit(1 if fail else 0)
