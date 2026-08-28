# src/predict_fixtures.py
"""
Genera webapp/public/predictions.json per la dashboard Serie A 26/27.

Fonti calendario (in ordine di preferenza):
  1. football-data.co.uk  (CSV stagione I1, include quote B365)
  2. TheSportsDB          (API gratuita, eventi stagione 2026-2027)

Per ogni partita del calendario:
  - GIA GIOCATA  -> mantiene le stime H/D/A ORIGINALI (freeze): non vengono
                    ricalcolate quando escono i nuovi risultati. Si aggiornano
                    solo risultato reale + esito (ok/❌).
  - FUTURA       -> calcola le feature (forma, Pi-Rating, H2H, riposo,
                    flag neopromossa) e le % H/D/A.

In piu' (per la dashboard):
  - Confronto BOOKMAKER: quote B365 normalizzate in probabilita', e RPS del
    modello vs RPS del bookmaker sulle partite gia' giocate.
  - Kelly simulation: frazione di Kelly (25%) sulle giocate in cui il modello
    batte il bookmaker, con bankroll simulato e ROI.

Le stime delle partite passate restano STABILI nel tempo: lo storico cresce ma
non "sgonfia" le probabilita' gia' pubblicate.
"""
import sys
import io
import json
import time
import unicodedata
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
WEBAPP_PUBLIC = ROOT / "webapp" / "public"
OUT_JSON = WEBAPP_PUBLIC / "predictions.json"

FIXTURES_URL = "https://www.football-data.co.uk/mmz4281/{season}/I1.csv"
# TheSportsDB: la chiave pubblica di test "123" restituisce gli eventi completi
# per-giornata via eventsround.php (il bulk eventsseason.php e' limitato a ~5-15).
TSDB_KEY = "123"
TSDB_LEAGUE_ID = "4332"   # Serie A
TSDB_ROUNDS = 38
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
SEASON_FD = "2627"        # formato football-data
SEASON_TSDB = "2026-2027" # formato TheSportsDB

MODEL_PATH = MODELS_DIR / "catboost_seriea_v1.cbm"
KELLY_FRACTION = 0.25      # Kelly ridotto al 25% per gestione del rischio

# Nomi squadre TheSportsDB -> nomi storico football-data
ALIASES = {
    "internazionale": "Inter", "inter milan": "Inter", "fc internazionale": "Inter",
    "ac milan": "Milan", "milan fc": "Milan",
    "ssc napoli": "Napoli", "napoli fc": "Napoli",
    "ss lazio": "Lazio", "lazio fc": "Lazio",
    "as roma": "Roma", "roma fc": "Roma",
    "acf fiorentina": "Fiorentina", "fiorentina fc": "Fiorentina",
    "atalanta bc": "Atalanta", "atalanta fc": "Atalanta",
    "bologna fc": "Bologna", "bologna fc 1909": "Bologna",
    "genoa cfc": "Genoa", "genoa cricket club": "Genoa",
    "torino fc": "Torino",
    "udinese calcio": "Udinese",
    "cagliari calcio": "Cagliari",
    "parma calcio": "Parma", "parma fc": "Parma",
    "hellas verona": "Verona", "verona fc": "Verona",
    "us lecce": "Lecce", "lecce fc": "Lecce",
    "empoli fc": "Empoli",
    "frosinone calcio": "Frosinone",
    "venezia fc": "Venezia", "venezia calcio": "Venezia",
    "ac monza": "Monza", "monza fc": "Monza",
    "us sassuolo calcio": "Sassuolo", "sassuolo calcio": "Sassuolo",
    "como 1907": "Como",
    "pisa sporting club": "Pisa", "ac pisa": "Pisa", "pisa sc": "Pisa",
    "cremonese": "Cremonese", "us cremonese": "Cremonese",
    "spezia calcio": "Spezia", "us spezia": "Spezia",
    "juventus fc": "Juventus", "juventus": "Juventus",
}


def norm_team(name: str) -> str:
    """Normalizza il nome squadra al formato dello storico football-data."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = " ".join(s.lower().strip().split())
    if s in ALIASES:
        return ALIASES[s]
    # togli prefissi/suffissi comuni e anno di fondazione
    stop = {"fc", "calcio", "ac", "as", "ss", "ssc", "us", "asd", "bc",
            "cfc", "sc", "spa", "1907", "1909", "1893", "1899", "1926"}
    tokens = [t for t in s.split() if t not in stop]
    base = " ".join(tokens) if tokens else s
    return ALIASES.get(base, base.title())


def _http_get(url: str, retries: int = 3) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = 10 * attempt  # rate-limit: attesa lunga
                print(f"  [429 rate-limit] attendo {wait}s...")
                time.sleep(wait)
                continue
            print(f"  [retry {attempt}/{retries}] {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
        except Exception as e:
            print(f"  [retry {attempt}/{retries}] {e}")
            if attempt < retries:
                time.sleep(3 * attempt)
    return None


def fetch_football_data(season: str):
    """CSV football-data. None se non ancora disponibile.

    Estrae anche le quote B365 (H/D/A) se presenti nel CSV.
    """
    raw = _http_get(FIXTURES_URL.format(season=season))
    if raw is None:
        return None
    try:
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines='skip', encoding='latin-1')
        if 'HomeTeam' not in df.columns:
            return None
        # Normalizza i nomi squadra (football-data usa gia' i nomi canonici,
        # ma per sicurezza applichiamo norm_team)
        df['HomeTeam'] = df['HomeTeam'].map(norm_team)
        df['AwayTeam'] = df['AwayTeam'].map(norm_team)
        return df
    except Exception as e:
        print(f"  football-data parse error: {e}")
        return None


def _parse_events(events: list) -> list:
    """Converte eventi TheSportsDB in righe football-data-like.

    TheSportsDB NON espone le quote B365, quindi bm resta None.
    """
    rows = []
    for ev in events:
        home = norm_team(ev.get("strHomeTeam", ""))
        away = norm_team(ev.get("strAwayTeam", ""))
        date = ev.get("dateEvent")
        rnd = ev.get("intRound")
        if not home or not away or not date:
            continue
        hs, as_ = ev.get("intHomeScore"), ev.get("intAwayScore")
        fthg = ftag = ftr = np.nan
        if hs is not None and as_ is not None:
            try:
                fthg, ftag = int(hs), int(as_)
                ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
            except (ValueError, TypeError):
                pass
        rows.append({
            "Date": pd.to_datetime(date),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": fthg, "FTAG": ftag, "FTR": ftr,
            "B365H": np.nan, "B365D": np.nan, "B365A": np.nan,
            "giornata": int(rnd) if rnd else np.nan,
        })
    return rows


def fetch_thesportsdb(season: str):
    """Calendario completo: una chiamata eventsround.php per ogni giornata.
    La chiave di test '123' non espone il bulk stagionale completo."""
    def fetch_round(rnd: int):
        url = (f"https://www.thesportsdb.com/api/v1/json/{TSDB_KEY}"
               f"/eventsround.php?id={TSDB_LEAGUE_ID}&r={rnd}&s={season}")
        raw = _http_get(url)
        if raw is None:
            return []
        try:
            payload = json.loads(raw.decode('utf-8'))
        except Exception as e:
            print(f"  [gg {rnd}] json error: {e}")
            return []
        return _parse_events(payload.get("events") or [])

    all_rows = []
    missed = []
    for rnd in range(1, TSDB_ROUNDS + 1):
        rows = fetch_round(rnd)
        if not rows:
            missed.append(rnd)
        all_rows.extend(rows)
        time.sleep(0.6)  # rate-limit gentile

    # Seconda passata per le giornate mancate (di solito 429 transitori)
    if missed:
        print(f"  Seconda passata per {len(missed)} giornate: {missed}")
        time.sleep(15)
        for rnd in missed:
            rows = fetch_round(rnd)
            all_rows.extend(rows)
            time.sleep(1.5)

    df = pd.DataFrame(all_rows)
    if len(df):
        # dedup (eventi possono comparire in piu' chiamate)
        df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"  TheSportsDB: {len(df)} partite su {df['giornata'].nunique()} giornate")
    return df if len(df) else None


def assign_giornate(df: pd.DataFrame) -> pd.DataFrame:
    """Clustering cronologico in giornate (solo se la fonte non ha il round)."""
    df = df.sort_values('Date').reset_index(drop=True)
    rounds, cur, size, prev = [], 1, 0, None
    for d in df['Date']:
        if (prev is not None and (d - prev).days > 4) or size >= 10:
            cur += 1
            size = 0
        rounds.append(cur)
        size += 1
        prev = d
    df['giornata'] = rounds
    return df


def _match_key(home: str, away: str, date: str) -> str:
    return f"{date}|{home}|{away}"


def bookmaker_prob(home_odd, draw_odd, away_odd):
    """Normalizza le quote B365 rimuovendo l'overround -> probabilita' 1X2."""
    try:
        h, d, a = float(home_odd), float(draw_odd), float(away_odd)
        if h <= 1.0 or d <= 1.0 or a <= 1.0:
            return None
        raw = [1.0 / h, 1.0 / d, 1.0 / a]
        tot = sum(raw)
        if tot <= 0:
            return None
        return [x / tot for x in raw]  # [pH, pD, pA]
    except (TypeError, ValueError):
        return None


def kelly_fraction(p_model, odds, fraction=KELLY_FRACTION):
    """Kelly ridotto: frazione di bankroll da puntare. <=0 se non conviene."""
    b = odds - 1.0
    q = 1.0 - p_model
    if b <= 0:
        return 0.0
    k = (b * p_model - q) / b
    return max(0.0, k * fraction)


def ranked_probability_score(y_true, y_proba):
    """RPS: piu' basso e' meglio. y_true: 0/1/2, y_proba: lista [pH,pD,pA]."""
    n = len(y_proba[0]) if hasattr(y_proba, '__len__') else 3
    rps = 0.0
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n)[true])
        cum_pred = np.cumsum(prob)
        rps += np.mean((cum_pred - cum_true) ** 2)
    return rps / max(len(y_true), 1)


def main():
    print("=" * 55)
    print(f"PREDIZIONI FIXTURE Serie A {SEASON_FD[:2]}/{SEASON_FD[2:]}")
    print("=" * 55)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} non trovato: esegui prima src/model.py")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    # Timestamp del modello allenato: se cambia rispetto al run precedente,
    # il freeze delle stime passate viene ignorato (una correzione di bug nel
    # modello deve propagarsi). Altrimenti le stime giocate restano stabili.
    model_trained_at = pd.Timestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec="seconds")

    sys.path.insert(0, str(ROOT / 'src'))
    from predict import (load_history, compute_features_for_match, FEATURE_ORDER,
                         is_promoted)

    history = load_history()
    print(f"Storico: {len(history)} partite")

    # ── Calendario ────────────────────────────────────────────────────
    # 1) football-data.co.uk: porta le giornate DISPONIBILI + quote B365.
    #    All'inizio della stagione ha SOLO le prime giornate giocate.
    # 2) TheSportsDB: ha l'INTERO calendario 26/27 (tutte le 38 giornate,
    #    con date + intRound), ma SENZA quote. Lo usiamo per integrare le
    #    giornate future non ancora presenti su football-data, cosi' la
    #    dashboard mostra le prediction di CatBoost sulle prossime giornate.
    fx, source = None, None
    fd = fetch_football_data(SEASON_FD)
    if fd is not None and len(fd):
        fx, source = fd, "football-data.co.uk"
        # Normalizza Date in datetime64 (FD la legge come stringa DD/MM/YYYY)
        fx['Date'] = pd.to_datetime(fx['Date'], dayfirst=True, errors='coerce')
        # Assicura colonna giornata (FD non l'ha: la deriviamo da TSDB dopo)
        if 'giornata' not in fx.columns:
            fx['giornata'] = np.nan
        # Integra le giornate future mancanti da TheSportsDB
        print("  Integrazione giornate future da TheSportsDB...")
        tsdb = fetch_thesportsdb(SEASON_TSDB)
        if tsdb is not None:
            tsdb['Date'] = pd.to_datetime(tsdb['Date'], errors='coerce')
            # Chiave di dedup su data(giorno)+squadre, formato normalizzato
            def _key(row):
                return (pd.Timestamp(row['Date']).strftime('%Y-%m-%d'),
                        str(row['HomeTeam']), str(row['AwayTeam']))
            fd_keys = set(_key(r) for _, r in fx.iterrows())
            add_mask = [(_key(r) not in fd_keys) for _, r in tsdb.iterrows()]
            add = tsdb[add_mask]
            if len(add):
                # Mappa round di TSDB (affidabile per tutte le 38) su TUTTE le
                # righe (anche FD) cosi' le partite giocate restano nella loro
                # giornata reale invece di essere rimappate dal clustering.
                round_map = {_key(r): int(r['giornata']) for _, r in tsdb.iterrows()
                             if pd.notna(r['giornata'])}
                fx['giornata'] = fx.apply(
                    lambda r: round_map.get(_key(r), np.nan), axis=1)
                fx = pd.concat([fx, add], ignore_index=True)
                print(f"    +{len(add)} partite future aggiunte (senza quote B365)")
    else:
        print("  football-data non disponibile -> fallback TheSportsDB")
        tsdb = fetch_thesportsdb(SEASON_TSDB)
        if tsdb is not None:
            tsdb['Date'] = pd.to_datetime(tsdb['Date'], errors='coerce')
            fx, source = tsdb, "thesportsdb.com"

    if fx is None or not len(fx):
        print("Nessuna fonte calendario disponibile: scrivo JSON vuoto.")
        WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, 'w') as f:
            json.dump({
                "meta": {
                    "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "stagione": SEASON_FD,
                    "nota": "Calendario non ancora disponibile dalle fonti",
                },
                "stats": {"corrette": 0, "total": 0},
                "giornate": {},
            }, f, indent=2, ensure_ascii=False)
        return

    print(f"Fonte calendario: {source}")
    fx = fx.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    # Giornata: priorità a chi ce l'ha (TSDB intRound), altrimenti clustering
    if 'giornata' not in fx.columns:
        fx['giornata'] = np.nan
    fx['giornata'] = pd.to_numeric(fx['giornata'], errors='coerce')
    if fx['giornata'].isna().any():
        fx = fx.drop(columns=['giornata'])
        fx = assign_giornate(fx)
    fx['giornata'] = fx['giornata'].astype(int)
    # Deduplicazione finale per (data, squadre) a prova di bug nel merge
    fx = fx.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first').reset_index(drop=True)
    print(f"Fixture: {len(fx)} partite, {fx['giornata'].max()} giornate")

    # ── Soglia di freeze (regola N-1) ───────────────────────────────────
    # Una giornata N e' "bloccabile" (snapshot non piu ricalcolato) non appena
    # la giornata N-1 ha almeno un risultato reale. Quindi le giornate con
    # numero <= (ultima_giornata_giocata + 1) sono CONGELATE: le loro
    # percentuali non si aggiornano piu (no hindsight bias).
    # Calcoliamo l'ultima giornata con risultato reale dal calendario odierno.
    fx_with_result = fx[fx['FTR'].apply(lambda x: isinstance(x, str) and x.strip().upper() in ('H', 'D', 'A'))]
    if len(fx_with_result):
        ultima_giocata = int(fx_with_result['giornata'].max())
    else:
        ultima_giocata = 0
    freeze_until = ultima_giocata + 1   # giornate 1..freeze_until sono congelate
    print(f"  Freeze (regola N-1): giornate 1..{freeze_until} bloccate "
          f"(ultima giocata = {ultima_giocata})")

    # Squadre della Serie A corrente: derivate dal calendario stesso (fonte
    # autorevole), NON hardcodate. Serve per il flag neopromossa: chi e' in
    # questa lista ma ASSENTE dall'ultima stagione storica completa -> promosso.
    current_season_teams = set(fx['HomeTeam']) | set(fx['AwayTeam'])
    print(f"Squadre stagione corrente ({len(current_season_teams)}): "
          f"{sorted(current_season_teams)}")

    # ── Freeze stime passate ──────────────────────────────────────────
    # Carica le stime gia' pubblicate. Se il modello NON e' cambiato rispetto
    # al run precedente, le partite GIA' GIOCATE mantengono le loro prob/pred
    # (non driftano quando escono i nuovi risultati). Se il modello e' stato
    # riallenato (correzione di bug), il freeze viene ignorato e tutto si
    # ricalcola col modello nuovo.
    prev_estimates = {}
    prev_model_trained_at = None
    freeze_active = False
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON) as f:
                old = json.load(f)
            prev_model_trained_at = old.get("meta", {}).get("model_trained_at")
            for g, ms in old.get("giornate", {}).items():
                for m in ms:
                    if m.get("ok") is not None:  # gia' giocata nella vecchia run
                        prev_estimates[_match_key(m["home"], m["away"], m["date"])] = m
            freeze_active = (prev_model_trained_at == model_trained_at)
            n = len(prev_estimates)
            print(f"  Freeze {'ATTIVO' if freeze_active else 'DISATTIVO (modello cambiato)'}:"
                  f" {n} stime di partite gia' giocate")
        except Exception as e:
            print(f"  [warn] predictions.json precedente illeggibile: {e}")
            freeze_active = False

    # ── Predizioni + bookmaker + kelly ────────────────────────────────
    LABELS = ['H', 'D', 'A']
    h2h_cache = {}
    giornate: dict[str, list] = {}
    bm_rows = []   # (y_true, bm_prob) per RPS bookmaker sulle giocate
    model_rows = []  # (y_true, model_prob) per RPS modello sulle giocate
    kelly_log = []  # simulate scommesse sulle giocate con value

    for _, row in fx.iterrows():
        g = str(row['giornata'])
        m = {"home": row['HomeTeam'], "away": row['AwayTeam'],
             "date": row['Date'].strftime("%Y-%m-%d")}
        ftr = row.get('FTR')
        played = isinstance(ftr, str) and ftr.strip().upper() in ('H', 'D', 'A')

        # Quote bookmaker (solo football-data le porta)
        bm_prob = bookmaker_prob(row.get('B365H'), row.get('B365D'), row.get('B365A'))
        bm_odds = None
        if bm_prob is not None:
            try:
                bm_odds = [float(row['B365H']), float(row['B365D']), float(row['B365A'])]
            except (TypeError, ValueError):
                bm_odds = None

        k = _match_key(m["home"], m["away"], m["date"])
        giornata_n = int(row['giornata'])
        # ── SNAPSHOT / FREEZE ─────────────────────────────────────────────
        # Due casi in cui riusiamo lo snapshot (NON ricalcoliamo):
        #  (a) partita GIA' GIOCATA (ha risultato reale), o
        #  (b) giornata N con N <= freeze_until (la N-1 e' stata giocata):
        #      regola utente "la giornata N si aggiorna solo fino ai risultati
        #      della N-1" -> no hindsight bias.
        # In entrambi i casi, se lo snapshot esiste, lo riusiamo integro.
        if k in prev_estimates and (played or giornata_n <= freeze_until):
            old = prev_estimates[k]
            m["prob"] = old.get("prob")
            m["pred"] = old.get("pred")
            m["bm"] = old.get("bm")
            m["bm_prob"] = old.get("bm_prob")
            m["kelly"] = old.get("kelly")
            m["risultato"] = old.get("risultato")
            m["esito"] = old.get("esito")
            m["ok"] = old.get("ok")
            m["features"] = old.get("features")
            # Se lo snapshot non aveva le quote ma il CSV odierno le porta
            # (es. football-data), le valorizziamo dal CSV (non dal freeze).
            if m["bm"] is None and bm_odds is not None:
                m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                           "A": round(bm_odds[2], 2)}
                m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
        else:
            # Partita FUTURA (o mai vista): ricalcola col modello ATTUALE.
            feats = compute_features_for_match(history, m["home"], m["away"], row['Date'], h2h_cache, current_season_teams)
            X = pd.DataFrame([{c: feats.get(c, 0.0) for c in FEATURE_ORDER}])
            probs = model.predict_proba(X)[0]
            m["prob"] = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
            m["pred"] = LABELS[int(np.argmax(probs))]
            m["features"] = {c: round(float(feats.get(c, 0.0)), 3) for c in FEATURE_ORDER}

            if played:
                ftr = ftr.strip().upper()
                m["risultato"] = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
                m["esito"] = ftr
                m["ok"] = bool(m["pred"] == ftr)

            # Confronto bookmaker / Kelly solo sulle giocate (serve risultato)
            if bm_odds is not None and played:
                assert bm_prob is not None
                m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                           "A": round(bm_odds[2], 2)}
                m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
                y_idx = {'H': 0, 'D': 1, 'A': 2}[ftr]
                bm_rows.append((y_idx, bm_prob))
                model_rows.append((y_idx, [m["prob"][l] / 100 for l in LABELS]))
            else:
                m["bm"] = None
                m["bm_prob"] = None

            # Kelly retrospettivo: tra le 3 giocate, punta su quella con edge
            # positivo (p_model > p_bookmaker) usando la quota reale.
            if bm_odds is not None and played:
                best_k = -1
                best_info = None
                for i, lab in enumerate(LABELS):
                    pk = m["prob"][lab] / 100.0
                    pb = bm_prob[i]
                    if pk > pb:  # il modello stima piu' probabile del bookmaker
                        kf = kelly_fraction(pk, bm_odds[i])
                        if kf > 0 and kf > best_k:
                            best_k = kf
                            best_info = (lab, pk, bm_odds[i], kf)
                if best_info is not None:
                    lab, pk, odd, kf = best_info
                    win = (ftr == lab)
                    kelly_log.append({
                        "home": m["home"], "away": m["away"], "date": m["date"],
                        "pick": lab, "model_p": round(pk, 3), "odd": round(odd, 2),
                        "kelly": round(kf, 4), "win": win,
                    })
            # Partita futura: bookmaker + Kelly (prob/features gia' calcolate)
            if bm_odds is not None and played:
                assert bm_prob is not None
                m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                           "A": round(bm_odds[2], 2)}
                m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
                # Suggerimento Kelly sulle future: giocata con edge migliore
                best_k = -1
                best_info = None
                for i, lab in enumerate(LABELS):
                    pk = m["prob"][lab] / 100.0
                    pb = bm_prob[i]
                    if pk > pb:
                        kf = kelly_fraction(pk, bm_odds[i])
                        if kf > 0 and kf > best_k:
                            best_k = kf
                            best_info = (lab, pk, bm_odds[i], kf)
                if best_info is not None:
                    lab, pk, odd, kf = best_info
                    m["kelly"] = {"pick": lab, "model_p": round(pk, 3),
                                  "odd": round(odd, 2), "stake_pct": round(kf * 100, 2)}
                else:
                    m["kelly"] = None
            else:
                m["bm"] = None
                m["bm_prob"] = None
                m["kelly"] = None

        giornate.setdefault(g, []).append(m)

    tutte = [m for ms in giornate.values() for m in ms]
    giocate = [m for m in tutte if m.get("ok") is not None]
    corrette = sum(1 for m in giocate if m["ok"])
    ultima_gg = max((int(g) for g, ms in giornate.items()
                     if any(m.get("ok") is not None for m in ms)), default=0)
    prossima_gg = min((int(g) for g, ms in giornate.items()
                       if any(m.get("ok") is None for m in ms)), default=ultima_gg + 1)

    # ── Confronto bookmaker (RPS) + Kelly simulation ─────────────────
    meta_extra = {}
    if bm_rows:
        y_bm = [r[0] for r in bm_rows]
        p_bm = [r[1] for r in bm_rows]
        y_md = [r[0] for r in model_rows]
        p_md = [r[1] for r in model_rows]
        rps_bm = ranked_probability_score(y_bm, p_bm)
        rps_md = ranked_probability_score(y_md, p_md)
        meta_extra["bookmaker_rps"] = round(float(rps_bm), 4)
        meta_extra["model_rps"] = round(float(rps_md), 4)
        meta_extra["beats_bookmaker_rps"] = bool(rps_md < rps_bm)
        print(f"  RPS modello={rps_md:.4f} | RPS bookmaker={rps_bm:.4f} "
              f"| modello batte bookmaker: {rps_md < rps_bm}")

    if kelly_log:
        bankroll = 1000.0
        stakes_total = 0.0
        for bet in kelly_log:
            stake = bankroll * bet["kelly"]
            stakes_total += stake
            bankroll += stake * (bet["odd"] - 1) if bet["win"] else -stake
        profit = bankroll - 1000.0
        roi = (profit / 1000.0) * 100.0
        meta_extra["kelly"] = {
            "bankroll_iniziale": 1000.0,
            "bankroll_finale": round(float(bankroll), 2),
            "profitto": round(float(profit), 2),
            "roi_pct": round(float(roi), 2),
            "scommesse": len(kelly_log),
            "stake_totale_pct": round(float(stakes_total / 1000.0) * 100, 1),
            "fraction": KELLY_FRACTION,
        }
        print(f"  Kelly (fraction={KELLY_FRACTION}): {len(kelly_log)} scommesse, "
              f"bankroll finale €{bankroll:.2f} (ROI {roi:.1f}%)")

    out = {
        "meta": {
            "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
            "stagione": SEASON_FD,
            "fonte": source,
            "ultima_giornata_giocata": ultima_gg,
            "prossima_giornata": prossima_gg,
            "freeze_until": freeze_until,
            "freeze_regola": "La giornata N si aggiorna solo fino ai risultati della N-1 (no hindsight)",
            "model_trained_at": model_trained_at,
            **meta_extra,
        },
        "stats": {"corrette": corrette, "total": len(giocate)},
        "giornate": dict(sorted(giornate.items(), key=lambda kv: int(kv[0]))),
    }

    WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nGiornate: {len(giornate)} | Giocate: {len(giocate)} "
          f"| Corrette: {corrette} ({(corrette/max(len(giocate),1)*100):.0f}%)")
    print(f"Salvato: {OUT_JSON}")


if __name__ == "__main__":
    main()
