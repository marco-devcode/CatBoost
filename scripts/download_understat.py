#!/usr/bin/env python3
"""
Scarica l'xG da UNDERSTAT (fonte aperta, senza API key).

Endpoint: GET https://understat.com/getLeagueData/{league}/{season}
La pagina della lega carica i dati con AJAX; l'endpoint va richiesto con
gli header Referer e X-Requested-With, e arriva compresso in gzip.

Perche' Understat: e' l'unica fonte aperta che da' l'EXPECTED GOALS per ogni
tiro sulle 5 leghe europee dal 2014-15. I tiri in porta (HST/AST) del
football-data sono un proxy grezzo: due tiri in porta valgono 1.0 o 0.1 gol a
seconda della qualita' dell'occasione, e l'xG distingue i due casi.

Dal 2026 il formato della risposta e' {"teams": {...}, "players": [...],
"dates": [...]}. Noi salviamo due tabelle:

  1) PARTITE   (data, casa, fuori, xG casa/fuori, gol, previsione understat)
  2) STORIA    (data, squadra, xG, xG subiti, npxG, npxGA, PPDA, deep
                completions, deep subiti, gol fatti/subiti, xPTS)

Lo storico per squadra e' gia' pronto per le feature rolling: basta lo
shift(1) sulla serie ordinata per data.

Uso:  python scripts/download_understat.py --league Serie_A --from-year 2014 --to-year 2026
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0 Safari/537.36")


def nrm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s).lower()
    s = re.sub(r"\b(fc|cf|afc|as|ac|ss|ssc|us|ud|cd|rc|sc|bs|cl)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch_json(url: str, referer: str, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Referer": referer, "Accept": "*/*",
                "X-Requested-With": "XMLHttpRequest", "Accept-Encoding": "gzip",
            })
            with urllib.request.urlopen(req, timeout=45) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"download fallito {url}: {last}")


def team_index(teams):
    idx = {}
    for t in teams:
        n = nrm(t)
        if not n:
            continue
        idx.setdefault(n, t)
        for tok in n.split():
            idx.setdefault(tok, t)
    return idx


def resolve(name, index):
    n = nrm(name)
    if n in index:
        return index[n]
    toks = set(n.split())
    best = None
    for cand, orig in index.items():
        ct = set(cand.split())
        if toks & ct and (toks <= ct or ct <= toks):
            if best is None or len(cand) > len(best[0]):
                best = (cand, orig)
    return best[1] if best else name


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="Serie_A")
    ap.add_argument("--from-year", type=int, default=2014, dest="from_year")
    ap.add_argument("--to-year", type=int, default=2026, dest="to_year")
    args = ap.parse_args()

    dataset_teams = set()
    raw_path = ROOT / "data" / "serie_a_raw.csv"
    if raw_path.exists():
        with open(raw_path, encoding="latin-1") as f:
            for r in csv.DictReader(f):
                dataset_teams.add((r.get("HomeTeam") or "").strip())
                dataset_teams.add((r.get("AwayTeam") or "").strip())
    index = team_index(dataset_teams)

    matches, history = [], []
    for season in range(args.from_year, args.to_year + 1):
        ref = f"https://understat.com/league/{args.league}/{season}"
        url = f"https://understat.com/getLeagueData/{args.league}/{season}"
        try:
            data = fetch_json(url, ref)
        except Exception as e:  # noqa: BLE001
            print(f"  {season}: FALLITO ({e})")
            continue
        teams = data.get("teams", {})
        dates = data.get("dates", [])
        # indice history per (squadra grezza, data) -> npxG, ppda, deep:
        # npxG (xG senza rigori) e' un segnale piu' pulito dell'xG grezzo
        hist_idx = {}
        for t in teams.values():
            for e in t.get("history", []):
                hist_idx[(t.get("title", ""), (e.get("date") or "")[:10])] = e
        for m in dates:
            h = (m.get("h") or {})
            a = (m.get("a") or {})
            if not h.get("title") or not a.get("title"):
                continue
            he = hist_idx.get((h["title"], (m.get("datetime") or "")[:10]), {})
            ae = hist_idx.get((a["title"], (m.get("datetime") or "")[:10]), {})
            fc = m.get("forecast") or {}
            matches.append({
                "date": (m.get("datetime") or "")[:10],
                "home": resolve(h["title"], index),
                "away": resolve(a["title"], index),
                "raw_home": h["title"], "raw_away": a["title"],
                "season": season,
                "xG_home": round(_f((m.get("xG") or {}).get("h")), 4),
                "xG_away": round(_f((m.get("xG") or {}).get("a")), 4),
                "goals_home": int(_f((m.get("goals") or {}).get("h"))),
                "goals_away": int(_f((m.get("goals") or {}).get("a"))),
                "fc_home": _f(fc.get("w")), "fc_draw": _f(fc.get("d")),
                "fc_away": _f(fc.get("l")),
                "npxG_home": round(_f(he.get("npxG")), 4),
                "npxG_away": round(_f(ae.get("npxG")), 4),
                "ppda_home": round(_f((he.get("ppda") or {}).get("att")) /
                                   max(_f((he.get("ppda") or {}).get("def")), 1), 4),
                "ppda_away": round(_f((ae.get("ppda") or {}).get("att")) /
                                   max(_f((ae.get("ppda") or {}).get("def")), 1), 4),
                "deep_home": int(_f(he.get("deep"))),
                "deep_away": int(_f(ae.get("deep"))),
            })
        for t in teams.values():
            team = resolve(t.get("title", ""), index)
            for e in t.get("history", []):
                ppda = e.get("ppda") or {}
                ppa = e.get("ppda_allowed") or {}
                history.append({
                    "date": (e.get("date") or "")[:10],
                    "team": team, "raw_team": t.get("title", ""),
                    "season": season, "h_a": e.get("h_a", ""),
                    "xG": round(_f(e.get("xG")), 4),
                    "xGA": round(_f(e.get("xGA")), 4),
                    "npxG": round(_f(e.get("npxG")), 4),
                    "npxGA": round(_f(e.get("npxGA")), 4),
                    "ppda_att": int(_f(ppda.get("att"))), "ppda_def": int(_f(ppda.get("def"))),
                    "ppda_allowed_att": int(_f(ppa.get("att"))),
                    "ppda_allowed_def": int(_f(ppa.get("def"))),
                    "deep": int(_f(e.get("deep"))),
                    "deep_allowed": int(_f(e.get("deep_allowed"))),
                    "scored": int(_f(e.get("scored"))), "missed": int(_f(e.get("missed"))),
                    "xpts": round(_f(e.get("xpts")), 4),
                })
        print(f"  {season}: {len(dates)} partite, {sum(len(t.get('history', [])) for t in teams.values())} righe storia")
        time.sleep(1.0)

    if not matches:
        print("NESSUN DATO scaricato")
        raise SystemExit(1)

    out_m = ROOT / "data" / "understat_seriea.csv"
    with open(out_m, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
        w.writeheader()
        w.writerows(matches)
    out_h = ROOT / "data" / "understat_history_seriea.csv"
    with open(out_h, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"\n  {len(matches)} partite -> {out_m}")
    print(f"  {len(history)} righe di storia -> {out_h}")


if __name__ == "__main__":
    main()
