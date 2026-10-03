#!/usr/bin/env python3
"""Il contratto JSON del Kelly dinamico deve coincidere con webapp/lib/types.ts.

I due produttori del blocco `meta.kelly` (src/predict_fixtures.py e
src/evaluate.py) e i tipi che lo consumano (webapp/lib/types.ts) vivono in
file diversi: un campo rinominato da una parte e non dall'altra non fallisce
in Python, e in TypeScript fallisce solo se il file lo dichiara required —
mentre a runtime la pagina mostra NaN. Questo controllo incrocia i due lati
con l'AST, senza eseguire la pipeline (catboost non e' installato in locale).

Esegui:  python3.14 scripts/test_kelly_contract.py
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TYPES = ROOT / "webapp" / "lib" / "types.ts"

failures = []


def check(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


def dict_keys(path: Path, needle: str) -> set[str]:
    """Chiavi letterali del dict dict[...] costruito in un assegnamento che
    contiene `needle`. Solo literal: i campi calcolati sono esclusi."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # ast.unparse normalizza le virgolette: il needle va cercato come token
    # nudo, non come letterale con le sue quote.
    needle = needle.strip('"\'')
    found: set[str] = set()
    # Si seleziona il dict che ha il needle come CHIAVE, non lo statement che
    # lo contiene: a predict_fixtures il dict meta.kelly sta dentro
    # `meta_extra["kelly"] = {...}`, e il dict che lo contiene (l'intero
    # payload JSON) ha "meta"/"stats"/"giornate" come chiavi, non il needle.
    # Per `bets.append({...})` il dict e' direttamente un argomento della call.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        if needle in keys:
            found.update(keys)
    return found


def ts_interface(name: str) -> tuple[set[str], set[str]]:
    """(campi required, campi optional) dell'interfaccia TS `name`."""
    text = TYPES.read_text(encoding="utf-8")
    m = re.search(rf"export interface {name}\s*\{{(.*?)\n\}}", text, re.S)
    if not m:
        raise SystemExit(f"interfaccia {name} non trovata in {TYPES}")
    body = m.group(1)
    # rimuovi i commenti per non leggere campi citati solo nelle note
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    required, optional = set(), set()
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith(";"):
            continue
        is_optional = "?" in line.split(":")[0]
        field = line[:-1].split(":")[0].strip().rstrip("?").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            continue
        (optional if is_optional else required).add(field)
    return required, optional


print("== Tipi TypeScript ==")
kelly_req, kelly_opt = ts_interface("KellySummary")
bet_req, bet_opt = ts_interface("KellyBet")
print(f"  KellySummary: {len(kelly_req)} required, {len(kelly_opt)} optional")
print(f"  KellyBet:     {len(bet_req)} required, {len(bet_opt)} optional")

print("\n== predict_fixtures.py: meta.kelly e scommesse_dettaglio ==")
pf_summary = dict_keys(SRC / "predict_fixtures.py", '"stake_totale_pct"')
pf_bets = dict_keys(SRC / "predict_fixtures.py", '"bankroll_dopo"')
print(f"  meta.kelly: {sorted(pf_summary)}")
print(f"  bet:       {sorted(pf_bets)}")

print("\n== evaluate.py: kelly_metrics ==")
ev_summary = dict_keys(SRC / "evaluate.py", '"stake_totale_pct"')
print(f"  kelly_metrics: {sorted(ev_summary)}")

print("\n== Ogni campo REQUIRED deve essere prodotto da entrambi i lati ==")
# `capped` in predict_fixtures e' un conteggio, in evaluate si chiama
# n_troncate_al_cap: sono due nomi diversi per lo stesso fatto, quindi il
# campo required corrispondente e' dichiarato optional nei due tipi.
ALIAS = {"capped": {"capped", "n_troncate_al_cap"}}
for produttore, produced in (("predict_fixtures", pf_summary), ("evaluate", ev_summary)):
    for campo in sorted(kelly_req):
        if campo in ALIAS:
            check(f"{produttore} produce uno di {sorted(ALIAS[campo])}",
                  bool(produced & ALIAS[campo]), str(sorted(produced & ALIAS[campo])))
        else:
            check(f"{produttore} produce '{campo}'", campo in produced)

print("\n== Nessun campo 'fraction' (sostituito dal sizing dinamico) ==")
check("types.ts non dichiara più 'fraction'", "fraction" not in kelly_req | kelly_opt)
for nome, path in (("predict_fixtures", SRC / "predict_fixtures.py"),
                   ("evaluate", SRC / "evaluate.py"),
                   ("types.ts", TYPES)):
    testi = [i for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
             if re.search(r"""\bfraction\b\s*[:=]""", ln)]
    check(f"{nome} non assegna più 'fraction'", not testi, str(testi))

print("\n== Ogni campo REQUIRED di KellyBet è prodotto da predict_fixtures ==")
for campo in sorted(bet_req):
    if campo in bet_opt:
        continue
    check(f"bet produce '{campo}'", campo in pf_bets)

print("\n== I due produttori non si contraddicono sul summary ==")
# Un campo required presente in un solo produttore farebbe fallire l'altro
# lato a runtime (NaN sulla pagina), quindi ogni campo che NON e' in comune
# deve essere dichiarato optional nei tipi.
solo_pf = sorted((pf_summary - ev_summary) - set(ALIAS["capped"]))
solo_ev = sorted(ev_summary - pf_summary)
check("predict_fixtures non inventa campi assenti in evaluate",
      all(c in kelly_opt for c in solo_pf), str(solo_pf))
check("evaluate non inventa campi assenti in predict_fixtures",
      all(c in kelly_opt for c in solo_ev), str(solo_ev))
mancanti = sorted((kelly_req - pf_summary) - set(ALIAS["capped"]))
check("nessun campo required senza produttore", not mancanti, str(mancanti))

print()
if failures:
    print(f"KELLY CONTRACT: {len(failures)} CONTROLLI FALLITI -> {failures}")
    sys.exit(1)
print("KELLY CONTRACT: i due lati del contratto coincidono.")