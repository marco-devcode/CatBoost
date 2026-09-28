# src/sources.py
"""
Fonti dati per calendario e quote, per campionato.

Tre fonti, ognuna con un ruolo distinto. Non sono intercambiabili:

1. football-data.co.uk  -> storico + quote delle partite GIA GIOCATE.
   Non pubblica le partite future della stagione corrente.

2. openfootball (GitHub, public domain, nessuna chiave) -> calendario
   COMPLETO della stagione corrente, giocate e future, con la giornata
   esplicita. E' la fonte delle future per le leghe che non hanno TheSportsDB.
   Limite: i nomi squadra sono diversi dallo storico (AFC Ajax vs Ajax).

3. TheSportsDB (key free) -> calendario futuro, ma solo per 5 leghe
   (Premier, Championship, Scottish, Bundesliga, Serie A). L'Eredivisie non
   c'e', quindi non e' usabile come fonte generale.

4. TheOddsAPI -> quote PREPARTITA dei bookmaker. Copre solo l'orizzonte
   vicino (le quote esistono per le partite che si avvicinano, non per tutto
   il calendario): e' il comportamento normale del mercato, non un bug.
   Una richiesta copre tutti gli eventi con quote di una lega.

Regola di normalizzazione: ogni nome squadra esterno viene riportato al
formato canonico dello storico tramite League.aliases. Senza questo passo
il modello vede squadre diverse da quelle su cui e' stato addestrato e le
previsioni diventano rumore.
"""
from __future__ import annotations

import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# Suffissi societari da rimuovere quando il nome non e' in aliases
_STOPWORDS = {"fc", "calcio", "ac", "as", "ss", "ssc", "us", "asd", "bc",
              "cfc", "sc", "spa", "sc", "afc", "fsv", "sv", "vfl", "tsv",
              "bsc", "rc", "rcd", "sd", "ud", "cd", "de", "the", "1963",
              "1907", "1909", "1893", "1899", "1926", "'65", "65"}


def norm_team(name: str, league) -> str:
    """Riporta un nome squadra esterno al formato canonico dello storico.

    1. normalizza accenti e case
    2. se il risultato e' in league.aliases -> nome storico
    3. altrimenti rimuove i suffissi societari e riprova
    4. altrimenti Title Case (con fallback warning in predict_fixtures)
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = " ".join(s.lower().strip().split())
    if not s:
        return ""
    if s in league.aliases:
        return league.aliases[s]

    tokens = [t for t in s.split() if t not in _STOPWORDS]
    base = " ".join(tokens) if tokens else s
    if base in league.aliases:
        return league.aliases[base]
    return league.aliases.get(base, base.title())


def _http_get(url: str, retries: int = 3, timeout: int = 30) -> bytes | None:
    """GET con retry. 429 = rate limit (attesa lunga), altri = attesa breve."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = 10 * attempt
                print(f"  [429 rate-limit] attendo {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [retry {attempt}/{retries}] HTTP {e.code}")
            if attempt < retries:
                time.sleep(3 * attempt)
        except Exception as e:
            print(f"  [retry {attempt}/{retries}] {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 1. football-data.co.uk — storico e quote delle giocate
# ═══════════════════════════════════════════════════════════════════════════
def fetch_football_data(league, season: str | None = None):
    """CSV football-data della stagione. None se non disponibile.

    Restituisce solo partite con risultato: football-data pubblica le partite
    della stagione corrente man mano che vengono giocate, quindi il futuro non
    e' qui dentro (per quello serve openfootball).
    """
    season = season or league.season_fd
    raw = _http_get(league.fd_url.format(season=season))
    if raw is None:
        return None
    import io
    try:
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip", encoding="latin-1")
        if "HomeTeam" not in df.columns:
            return None
        df["HomeTeam"] = df["HomeTeam"].map(lambda n: norm_team(n, league))
        df["AwayTeam"] = df["AwayTeam"].map(lambda n: norm_team(n, league))
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        return df.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    except Exception as e:
        print(f"  football-data parse error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 2. openfootball — calendario completo, giocate + future
# ═══════════════════════════════════════════════════════════════════════════
def fetch_openfootball(league):
    """Calendario completo da openfootball/football.json (public domain).

    Formato: {"matches": [{"round": "Matchday 1", "date": "2026-08-07",
    "time": "20:00", "team1": ..., "team2": ...,
    "score": {"ht": [h, a], "ft": [h, a]}}]}

    Attenzione: `score.ft` e' una LISTA [h, a], non un dict, e una partita su
    ~300 lo pubblica come lista piu' che come dict — un parser che assume
    dict() su tutti gli elementi va in crash proprio su quella.
    """
    url = league.openfootball_url
    if not url:
        return None
    raw = _http_get(url)
    if raw is None:
        print(f"  openfootball non disponibile per {league.name}")
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"  openfootball json error: {e}")
        return None

    rows = []
    for m in payload.get("matches") or []:
        home = norm_team(m.get("team1", ""), league)
        away = norm_team(m.get("team2", ""), league)
        date = m.get("date")
        if not home or not away or not date:
            continue
        rnd = str(m.get("round") or "")
        # "Matchday 8" -> 8; se assente, NaN (il chiamante raggrupperà)
        try:
            giornata = int(rnd.split()[-1])
        except (ValueError, IndexError):
            giornata = float("nan")

        fthg = ftag = ftr = float("nan")
        sc = m.get("score")
        if isinstance(sc, dict):
            ft = sc.get("ft")
        elif isinstance(sc, list) and len(sc) == 2:
            ft = sc          # forma lista [h, a]
        else:
            ft = None
        if ft is not None:
            try:
                hg, ag = (ft[0], ft[1])
                if hg is not None and ag is not None:
                    fthg, ftag = int(hg), int(ag)
                    ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
            except (ValueError, TypeError, IndexError):
                pass

        rows.append({
            "Date": pd.Timestamp(date),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
            # openfootball non ha quote: restano NaN, la UI mostra
            # "Quote non disponibili" se TheOddsAPI non le fornisce
            "B365H": float("nan"), "B365D": float("nan"), "B365A": float("nan"),
            "giornata": giornata,
            "time": m.get("time"),
        })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values("Date").reset_index(drop=True)
    played = df["FTR"].notna().sum()
    print(f"  openfootball {league.name}: {len(df)} partite "
          f"({played} giocate, {len(df) - played} future), "
          f"{df['giornata'].nunique()} giornate")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. TheSportsDB — calendario futuro (solo leghe nel free tier)
# ═══════════════════════════════════════════════════════════════════════════
TSDB_KEY = "123"


def _parse_tsdb_events(events: list, league) -> list:
    rows = []
    for ev in events:
        home = norm_team(ev.get("strHomeTeam", ""), league)
        away = norm_team(ev.get("strAwayTeam", ""), league)
        date = ev.get("dateEvent")
        rnd = ev.get("intRound")
        if not home or not away or not date:
            continue
        hs, as_ = ev.get("intHomeScore"), ev.get("intAwayScore")
        fthg = ftag = ftr = float("nan")
        if hs is not None and as_ is not None:
            try:
                fthg, ftag = int(hs), int(as_)
                ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
            except (ValueError, TypeError):
                pass
        rows.append({
            "Date": pd.Timestamp(date),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
            "B365H": float("nan"), "B365D": float("nan"), "B365A": float("nan"),
            "giornata": int(rnd) if rnd else float("nan"),
            "time": ev.get("strTime") or "20:00",
        })
    return rows


def fetch_thesportsdb(league, season_tsd: str | None = None):
    """Calendario completo da TheSportsDB, una chiamata per giornata.

    Disponibile SOLO se league.tsdb_league e' valorizzato: il piano free
    espone 5 leghe e l'Eredivisie non e' tra queste. Non e' un ID sbagliato:
    il free tier elenca Premier, Championship, Scottish, Bundesliga, Serie A.
    """
    if not league.tsdb_league:
        return None
    season_tsd = season_tsd or f"{league.season_fd[:2]}-{league.season_fd[2:]}"
    season_tsd = f"20{season_tsd}"

    def fetch_round(rnd: int):
        url = (f"https://www.thesportsdb.com/api/v1/json/{TSDB_KEY}"
               f"/eventsround.php?id={league.tsdb_league}&r={rnd}&s={season_tsd}")
        raw = _http_get(url)
        if raw is None:
            return []
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return []
        return _parse_tsdb_events(payload.get("events") or [], league)

    all_rows, missed = [], []
    for rnd in range(1, league.tsdb_rounds + 1):
        rows = fetch_round(rnd)
        if not rows:
            missed.append(rnd)
        all_rows.extend(rows)
        time.sleep(0.6)

    if missed:
        print(f"  Seconda passata per {len(missed)} giornate: {missed}")
        time.sleep(15)
        for rnd in missed:
            all_rows.extend(fetch_round(rnd))
            time.sleep(1.5)

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  TheSportsDB {league.name}: {len(df)} partite, "
          f"{df['giornata'].nunique()} giornate")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. TheOddsAPI — quote prepartita
# ═══════════════════════════════════════════════════════════════════════════
# La chiave NON e' nel repo: si legge da una variabile d'ambiente o da un file
# fuori dal tree git. In CI arriva come secret.
ODDS_BASE = "https://api.the-odds-api.com/v4"
ODDS_HOST = "api.the-odds-api.com"   # NON theoddsapi.com: dominio diverso


def _odds_key() -> str | None:
    """Chiave TheOddsAPI da env o da file locale. Nessun fallback hardcoded."""
    key = os.environ.get("THEODDSAPI_KEY")
    if key:
        return key.strip()
    # Path fuori dal repo: la chiave non deve mai entrare nel git.
    for cand in (Path.home() / ".hermes" / "secrets" / "theoddsapi.key",):
        try:
            if cand.exists():
                return cand.read_text().strip()
        except OSError:
            pass
    return None


def fetch_odds(league, bookmaker: str | None = None) -> dict:
    """Quote prepartita 1X2 da TheOddsAPI.

    Ritorna {(home, away, data_iso): {"H":..,"D":..,"A":..}} con i nomi gia'
    normalizzati allo storico, cosi' predict_fixtures puo' fare il join diretto.

    Copre solo gli eventi con quote disponibili (tipicamente le prossime
    giornate): e' il comportamento del mercato, non un limite del piano.
    Restituisce {} se manca la chiave o la lega non e' coperta.
    """
    if not league.odds_sport_key:
        return {}
    key = _odds_key()
    if not key:
        print("  [odds] chiave TheOddsAPI non trovata: quote prepartita "
              "disponibili solo sulle partite gia' giocate")
        return {}

    bookmaker = bookmaker if bookmaker is not None else league.odds_bookmaker
    # `regions` (plurale) e' obbligatorio: senza l'API risponde 422
    # MISSING_REGION, non un errore di auth. Se si chiede un bookmaker
    # specifico si passa `bookmakers` al suo posto.
    scope = f"bookmakers={bookmaker}" if bookmaker else "regions=eu"
    url = (f"{ODDS_BASE}/sports/{league.odds_sport_key}/odds/"
           f"?apiKey={key}&markets=h2h&oddsFormat=decimal&{scope}")
    raw = _http_get(url, retries=2)
    if raw is None:
        print("  [odds] TheOddsAPI non raggiungibile")
        return {}
    try:
        events = json.loads(raw.decode("utf-8"))
    except Exception as e:
        print(f"  [odds] json error: {e}")
        return {}
    if not isinstance(events, list):
        print(f"  [odds] risposta non lista: {str(events)[:160]}")
        return {}

    out = {}
    unresolved = set()
    for ev in events:
        home = norm_team(ev.get("home_team", ""), league)
        away = norm_team(ev.get("away_team", ""), league)
        date = str(ev.get("commence_time", ""))[:10]
        if not home or not away or not date:
            continue
        picked = None
        bookmakers = ev.get("bookmakers") or []
        # Se un bookmaker e' richiesto, cercalo; altrimenti media del mercato
        # (il "sharp" medio e' piu' stabile del singolo bookmaker migliore).
        for bm in bookmakers:
            if bookmaker and bm.get("key") != bookmaker:
                continue
            for mk in bm.get("markets") or []:
                if mk.get("key") != "h2h":
                    continue
                vals = {o.get("name"): o.get("price") for o in mk.get("outcomes") or []}
                if not vals:
                    continue
                d = vals.get("Draw") or vals.get(draw_variant(vals))
                h = vals.get(home) or vals.get(fuzzy(vals, home))
                a = vals.get(away) or vals.get(fuzzy(vals, away))
                if h and d and a:
                    picked = (h, d, a)
                    break
            if picked:
                break
        if not picked:
            # Fallback: media di tutti i bookmaker disponibili
            acc = {"H": [], "D": [], "A": []}
            for bm in bookmakers:
                for mk in bm.get("markets") or []:
                    if mk.get("key") != "h2h":
                        continue
                    vals = {o.get("name"): o.get("price") for o in mk.get("outcomes") or []}
                    d = vals.get("Draw")
                    h = vals.get(home) or vals.get(fuzzy(vals, home))
                    a = vals.get(away) or vals.get(fuzzy(vals, away))
                    if h and d and a:
                        acc["H"].append(h); acc["D"].append(d); acc["A"].append(a)
            if acc["H"]:
                picked = (sum(acc["H"]) / len(acc["H"]),
                          sum(acc["D"]) / len(acc["D"]),
                          sum(acc["A"]) / len(acc["A"]))
        if picked:
            out[(home, away, date)] = {"H": round(picked[0], 2),
                                       "D": round(picked[1], 2),
                                       "A": round(picked[2], 2)}
        else:
            unresolved.add(f"{home}-{away}")

    print(f"  [odds] TheOddsAPI {league.name}: {len(out)} eventi con quote "
          f"({len(unresolved)} senza 1X2)")
    return out


def draw_variant(vals: dict):
    """Il pareggio puo' arrivare come 'Draw', 'Tie', 'X' o 'D'."""
    for cand in ("Draw", "Tie", "X", "D", "draw"):
        if cand in vals:
            return cand
    return None


def fuzzy(vals: dict, target: str):
    """Ultimo tentativo: nome con prefissi/suffissi diversi.

    'SC Heerenveen' su football-data puo' arrivare come 'Heerenveen' dalle
    quote. Confronta i token per trovare la corrispondenza.
    """
    t = set(str(target).lower().replace(".", " ").split())
    for name in vals:
        n = set(str(name).lower().replace(".", " ").split())
        if t & n and (t - n) <= {"sc", "fc", "afc", "sbv", "vfc", "az", "psv"}:
            return name
    return None
