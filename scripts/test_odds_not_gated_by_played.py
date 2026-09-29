"""Le quote prepartita finivano nel ramo sbagliato.

Il bug: la condizione era `if bm_odds is not None and played`. Ma le uniche
partite con quote sono le FUTURE (TheOddsAPI non fornisce lo storico), quindi
`played` era False e le 8/20 quote finivano nell'`else` che le azzerava.
Risultato: 0 quote nel JSON, mentre il log diceva "Quote prepartita: 8
partite future" — la fusione col calendario funzionava, lo smistamento
successivo la buttava via.

Il test verifica la CONDIZIONE, non una copia della logica: `played` non deve
comparire nel gate che assegna le quote.
"""
import ast
import re
import sys
from pathlib import Path


def re_has_played(test: str) -> bool:
    return re.search(r"\bplayed\b", test) is not None

SRC = Path(__file__).resolve().parent.parent / "src" / "predict_fixtures.py"
src = SRC.read_text(encoding="utf-8")
tree = ast.parse(src)

fail = 0

# 1. Nessun gate che assegna bm deve richiedere `played`.
#    Cerca: if <cond> and played  (o played and <cond>) followed da m["bm"] =
bad_gates = []
for node in ast.walk(tree):
    if isinstance(node, ast.If):
        test = ast.unparse(node.test)
        # solo il corpo del nodo, NON orelse: in una catena if/elif/else
        # ast.unparse(node) include l'else, e l'elif (bm_odds is None ->
        # m["bm"] = None) farebbe scattare il check sullo stato buono.
        body = "\n".join(ast.unparse(st) for st in node.body)
        if 'm["bm"]' not in body and "m['bm']" not in body:
            continue
        if not re_has_played(test):
            continue
        # Gate LEGITTIMI: il ramo snapshot del freeze (k in prev_estimates),
        # che non assegna quote ma ripristina lo snapshot.
        if "prev_estimates" in test:
            print(f"  info gate legittimo riga {node.lineno}: if {test}")
            continue
        # Il confronto RPS sulle giocate (bm_rows.append) e' legittimo SOLO se
        # il ramo NON contiene un gate su played che ASSEGNA le quote. Il
        # caso del bug: `if bm_odds is not None and played:` che faceva
        # sia bm_rows.append sia m["bm"] = {...}, e la sua `else` azzerava
        # tutto — quindi le future perdevano le quote.
        #
        # Nota: `elif bm_odds is None:` nella catena contiene ancora
        # m["bm"] = None, ma NON e' il problema (e' pulizia, non assegnazione
        # condizionata da played). Per questo si controlla il test principale
        # del nodo, non l'intero corpo concatenato della catena If/If.
        if "bm_rows.append" in body and node.test is not None \
                and re.search(r'm\[.{0,2}bm.{0,2}\]\s*=', body):
            bad_gates.append(
                f"riga {node.lineno}: il gate RPS assegna anche le quote "
                f"(if {test}) — le future finiscono scartate")
            continue
        if "bm_rows.append" in body:
            print(f"  info gate legittimo riga {node.lineno}: solo RPS, "
                  f"assegnazione quote separata")
            continue
        bad_gates.append(f"riga {node.lineno}: if {test}")
print(f"gate che richiedono `played` per assegnare le quote: {len(bad_gates)}")
for g in bad_gates:
    print(f"  FAIL {g}")
    fail += 1
if not bad_gates:
    print("  OK   nessun gate lega l'assegnazione delle quote a `played`")

# 2. Il ramo RPS/Kelly retrospettivo DEVE continuare a richiedere played,
#    altrimenti si confronterebbe il modello con il bookmaker su partite senza
#    risultato reale (e il JSON avrebbe RPS inventato).
rps_requires_played = "and played" in src and "bm_rows.append" in src
print(f"\nRPS/Kelly retrospettivo ancora legato a `played`: {rps_requires_played}")
if not rps_requires_played:
    print("  ATTENZIONE: verifica manuale necessaria sul gate del RPS")

print(f"\n{'TUTTI I CASI PASSANO' if not fail else f'{fail} FALLITI'}")
sys.exit(1 if fail else 0)
