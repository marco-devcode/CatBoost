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
import os
import sys
import io
import json
import math
import time
import unicodedata
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league, WEBAPP_PUBLIC  # noqa: E402
import sources  # noqa: E402
from kelly_dynamic import (  # noqa: E402
    KELLY_DYNAMIC_CONFIG,
    calculate_dynamic_kelly_stake,
    dynamic_kelly_breakdown,
    select_value_bet_stake,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
WEBAPP_PUBLIC = WEBAPP_PUBLIC

# ── Kelly Dinamico ────────────────────────────────────────────────────────
# Non esiste piu' una KELLY_FRACTION fissa: il sizing lo decide
# `calculate_dynamic_kelly_stake(p, odds)` (src/kelly_dynamic.py), che
# attenuа il Kelly pieno per scaglioni di quota e applica un hard cap.
KELLY_MODE = "dinamico"

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


def _bm_valid(d) -> bool:
    """Un dict di quote/probabilita' e' utilizzabile solo con numeri FINITI.

    Non basta `x is not None` e non basta isinstance(x, float):
    NaN e' un float, quindi isinstance(nan, (int, float)) -> True. Con NaN il
    dict verrebbe costruito, `round(nan, 2)` restituirebbe nan, _json_safe lo
    convertirebbe in null, e il frontend farebbe toFixed(0) su None: pagina
    bianca. isfinite scarta insieme None, nan e inf.

    Vive a livello di modulo perche' serve a DUE rami (snapshot e future):
    definirlo dentro il ramo snapshot lo lasciava "possibly unbound" nel ramo
    future, ed e' esattamente li' che i dict-vuoto sono ricomparsi (310
    partite future della Serie A, che non hanno colonne B365 in openfootball).
    """
    if not isinstance(d, dict):
        return False
    return all(
        isinstance(d.get(l), (int, float)) and math.isfinite(d.get(l))
        for l in ("H", "D", "A")
    )


def _match_key(home: str, away: str, date: str) -> str:
    return f"{date}|{home}|{away}"


def bookmaker_prob(home_odd, draw_odd, away_odd, method=None):
    """Quote B365 -> probabilita' 1X2, rimuovendo il margine (de-vig).

    Il metodo di default NON e' piu' la normalizzazione proporzionale (quella
    storica del repo): e' il metodo "additive", scelto EMPIRICAMENTE su 6135
    partite Serie A con quote B365 reali (scripts/select_devig.py), dove e'
    quello con il log-loss minimo, quindi il piu' CALIBRATO come previsione
    del risultato.

    Numeri (stesso dataset, log-loss sulle partite giocate):
      multiplicative  0.96105  <- metodo storico del repo
      additive        0.95982  <- scelto qui
      power           0.95985
      shin            0.96006
      odds_ratio      0.96022

    Perche' il metodo storico e' sbagliato: il bookmaker non distribuisce il
    margine in modo uniforme (favourite-longshot bias: margine relativo
    maggiore sui longshot), quindi la normalizzazione proporzionale toglie
    TROPPO al favorito e TROPPO POCO al longshot. L'effetto e' che il
    benchmark storico sottostimava la qualita' del bookmaker: il confronto
    "modello vs bookmaker" era falsato in entrambe le direzioni.

    Riferimenti: Shin (1992, 1993) The Economic Journal; Power (1990) IJF;
    Strumbelj (2014) IJF; Eckert (2013) Univ. of Canterbury WP.
    """
    try:
        h, d, a = float(home_odd), float(draw_odd), float(away_odd)
    except (TypeError, ValueError):
        return None
    if h <= 1.0 or d <= 1.0 or a <= 1.0:
        return None
    if method is None:
        try:
            from implied import DE_VIG_METHODS
            method = os.environ.get("HERMES_DE_VIG_METHOD", "additive")
            fn = DE_VIG_METHODS.get(method) or DE_VIG_METHODS["additive"]
            return fn([h, d, a])
        except Exception:
            method = "additive"
    try:
        raw = [1.0 / h, 1.0 / d, 1.0 / a]
        tot = sum(raw)
        if tot <= 0:
            return None
        if method == "multiplicative":
            return [x / tot for x in raw]  # [pH, pD, pA]
        # additive: margine equidistribuito
        m = (tot - 1.0) / 3.0
        out = [x - m for x in raw]
        if any(v <= 0 for v in out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def ranked_probability_score(y_true, y_proba):
    """RPS: piu' basso e' meglio. y_true: 0/1/2, y_proba: lista [pH,pD,pA]."""
    n = len(y_proba[0]) if hasattr(y_proba, '__len__') else 3
    rps = 0.0
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n)[true])
        cum_pred = np.cumsum(prob)
        rps += np.mean((cum_pred - cum_true) ** 2)
    return rps / max(len(y_true), 1)


def main(league):
    print("=" * 55)
    print(f"PREDIZIONI FIXTURE {league.name} "
          f"{league.season_fd[:2]}/{league.season_fd[2:]}")
    print("=" * 55)

    MODEL_PATH = league.model_path
    OUT_JSON = league.json_path
    SEASON_FD = league.season_fd

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} non trovato: esegui prima "
            f"src/model.py --league {league.key}")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    model_trained_at = pd.Timestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec="seconds")

    from predict import (load_history, compute_features_for_match, is_promoted)
    from predict import _build_feature_order, apply_saved_calibration, predict_proba_with_prior

    history = load_history(league)

    # L'elenco delle feature viene derivato dal MODELO STESSO: CatBoost
    # registra i nomi delle feature usate in training, quindi l'inferenza
    # usa esattamente quelle, nell'ordine giusto. Se il training e
    # l'inferenza divergessero, il modello riceverebbe colonne diverse da
    # quelle su cui e' stato addestrato -> output rumore.
    #
    # La fonte resta il modello, non il dataset corrente: cosi' un dataset
    # rigenerato non puo' cambiare silenziosamente le feature del modello
    # gia' allenato.
    model_feature_names = list(model.feature_names_)
    FEATURE_ORDER = model_feature_names
    print(f"Feature dal modello ({len(FEATURE_ORDER)}): {FEATURE_ORDER}")
    print(f"Storico: {len(history)} partite")

    # ── Calendario ────────────────────────────────────────────────────
    # Ordine di preferenza:
    #   1. openfootball  -> calendario COMPLETO (giocate + future) e giornata
    #                       esplicita. E' la fonte primaria.
    #   2. football-data -> stagioni dove openfootball non copre (Serie A puo'
    #                       restare su TSDB, che ha le future).
    #   3. TheSportsDB   -> solo se la lega e' nel free tier.
    print("  Calendario da openfootball...")
    fx = sources.fetch_openfootball(league)
    source = "openfootball"
    fd = None

    if fx is None:
        print("  openfootball non disponibile -> football-data.co.uk")
        fd = sources.fetch_football_data(league)
        if fd is not None and len(fd):
            fx, source = fd, "football-data.co.uk"
            fx['giornata'] = np.nan
            # Integra le giornate future mancanti da TheSportsDB
            if league.tsdb_league:
                print("  Integrazione giornate future da TheSportsDB...")
                tsdb = sources.fetch_thesportsdb(league)
                if tsdb is not None:
                    def _key(row):
                        return (pd.Timestamp(row['Date']).strftime('%Y-%m-%d'),
                                str(row['HomeTeam']), str(row['AwayTeam']))
                    fd_keys = set(_key(r) for _, r in fx.iterrows())
                    add_mask = [(_key(r) not in fd_keys) for _, r in tsdb.iterrows()]
                    add = tsdb[add_mask]
                    if len(add):
                        round_map = {_key(r): int(r['giornata']) for _, r in tsdb.iterrows()
                                     if pd.notna(r['giornata'])}
                        fx['giornata'] = fx.apply(
                            lambda r: round_map.get(_key(r), np.nan), axis=1)
                        fx = pd.concat([fx, add], ignore_index=True)
                        print(f"    +{len(add)} partite future aggiunte")
        else:
            print("  football-data non disponibile -> TheSportsDB")
            tsdb = sources.fetch_thesportsdb(league)
            if tsdb is not None:
                fx, source = tsdb, "thesportsdb.com"

    if fx is None or not len(fx):
        print("Nessuna fonte calendario disponibile: scrivo JSON vuoto.")
        WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
        with open(OUT_JSON, 'w') as f:
            json.dump({
                "meta": {
                    "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
                    "league": league.key,
                    "league_name": league.name,
                    "stagione": SEASON_FD,
                    "nota": "Calendario non ancora disponibile dalle fonti",
                },
                "stats": {"corrette": 0, "total": 0},
                "giornate": {},
            }, f, indent=2, ensure_ascii=False)
        return

    print(f"Fonte calendario: {source}")
    fx = fx.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
    # Giornata: priorita' a chi ce l'ha (openfootball/TSDB), altrimenti clustering
    if 'giornata' not in fx.columns:
        fx['giornata'] = np.nan
    fx['giornata'] = pd.to_numeric(fx['giornata'], errors='coerce')
    if fx['giornata'].isna().any():
        # NON chiamare assign_giornate se solo alcune righe mancano: il
        # clustering cronologico rimapperebbe TUTTE le giornate sbagliato
        # (le partite football-data finirebbero nella giornata sbagliata).
        if fx['giornata'].notna().sum() == 0:
            fx = fx.drop(columns=['giornata'])
            fx = assign_giornate(fx)
        else:
            fx['giornata'] = fx['giornata'].fillna(0).astype(int)
    fx['giornata'] = fx['giornata'].astype(int)
    print(f"Fixture: {len(fx)} partite, {fx['giornata'].nunique()} giornate")

    # ── Quote B365 delle partite GIA GIOCATE (football-data) ─────────
    # La fonte del CALENDARIO e la fonte delle QUOTE sono due ruoli distinti.
    # openfootball dà il calendario completo (tutte le giornate, con date e
    # risultati) ma NON ha le colonne B365:football-data dà le quote delle
    # giocate ma non il calendario futuro. Sostituendo l'una con l'altra si
    # erano perse le quote: `fd` restava None, il blocco di merge sotto non
    # girava e tutte le partite giocate uscivano con bm=None, senza edge e
    # senza Kelly retrospettivo (misurato: 50/50 giocate Serie A e 63/63
    # Eredivisie senza quote, contro 50/50 con quote nel backup precedente).
    #
    # Quindi football-data viene SEMPRE scaricato per le quote, anche quando
    # il calendario arriva da openfootball, e fuso per (data, casa, fuori).
    if fd is None:
        fd_quotes = sources.fetch_football_data(league)
    else:
        fd_quotes = fd

    if fd_quotes is not None and len(fd_quotes):
        qmap = {}
        rmap = {}
        extra_quotes = {}
        for _, r in fd_quotes.iterrows():
            key = (pd.Timestamp(r['Date']).strftime('%Y-%m-%d'),
                   str(r['HomeTeam']), str(r['AwayTeam']))
            try:
                h, d_, a_ = float(r['B365H']), float(r['B365D']), float(r['B365A'])
            except (KeyError, TypeError, ValueError):
                h = d_ = a_ = None
            if h is not None and all(math.isfinite(v) for v in (h, d_, a_)):
                qmap[key] = (h, d_, a_)
            # Pinnacle e media di mercato: nel TRAINING il modello vede
            # anche queste fonti (12 feature di mercato in tutto), quindi
            # l'inferenza deve riceverle dove disponibili, altrimenti gira
            # su colonne diverse da quelle su cui e' stata addestrata.
            # Sono le quote di APERTURA: note prima del match, nessun hindsight.
            for _label, _cols in (("pinnacle", ('PSH', 'PSD', 'PSA')),
                                  ("avg", ('AvgH', 'AvgD', 'AvgA'))):
                try:
                    _v = tuple(float(r[c]) for c in _cols)
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(v) and v > 1.0 for v in _v):
                    extra_quotes.setdefault(key, {})[_label] = _v
            # RISULTATI: football-data.co.uk pubblica ogni partita giocata
            # (FTR/FTHG/FTAG), mentre openfootball NON e' piu' aggiornato
            # (ultimo auto-update 2026-09-22: 50/63 risultati contro i 70/81
            # del CSV). Il join (data, casa, fuori) e' 1:1 e senza divergenze
            # sui risultati in comune, quindi il CSV completa il calendario
            # dove openfootball non ha ancora il risultato.
            ftr_fd = r.get('FTR')
            if isinstance(ftr_fd, str) and ftr_fd.strip().upper() in ('H', 'D', 'A'):
                try:
                    hg, ag = int(r['FTHG']), int(r['FTAG'])
                except (KeyError, TypeError, ValueError):
                    hg = ag = None
                if hg is not None and ag is not None:
                    rmap[key] = (ftr_fd.strip().upper(), hg, ag)
        # La colonna FTR puo' essere float (tutto NaN) se openfootball non ha
        # ancora nessun risultato: assegnare una stringa in una colonna float
        # solleva in pandas>=2. Si passa a object PRIMA di scrivere.
        if 'FTR' not in fx.columns:
            print("  Il calendario non ha la colonna FTR: nessun risultato da "
                  "fondere, restano solo le quote.")
        elif fx['FTR'].dtype != object:
            fx['FTR'] = fx['FTR'].astype(object)
        n_applied = 0
        n_result = 0
        for idx, row in fx.iterrows():
            key = (pd.Timestamp(row['Date']).strftime('%Y-%m-%d'),
                   str(row['HomeTeam']), str(row['AwayTeam']))
            q = qmap.get(key)
            if q is not None:
                fx.at[idx, 'B365H'], fx.at[idx, 'B365D'], fx.at[idx, 'B365A'] = q
                n_applied += 1
            # Pinnacle e media di mercato: stesse quote di apertura, dal CSV.
            # Servono alle 8 feature mkt_pinnacle_*/mkt_avg_* che il modello
            # si aspetta (il training le ha dove il CSV storico le aveva).
            ex = extra_quotes.get(key)
            if ex:
                for _label, _cols in (("pinnacle", ('PSH', 'PSD', 'PSA')),
                                      ("avg", ('AvgH', 'AvgD', 'AvgA'))):
                    if _label in ex:
                        for _c, _v in zip(_cols, ex[_label]):
                            if _c not in fx.columns:
                                fx[_c] = np.nan
                            fx.at[idx, _c] = _v
            res = rmap.get(key)
            if res is not None:
                ftr_v, hg, ag = res
                cur = fx.at[idx, 'FTR']
                if not (isinstance(cur, str)
                        and cur.strip().upper() in ('H', 'D', 'A')):
                    fx.at[idx, 'FTR'] = ftr_v
                    fx.at[idx, 'FTHG'] = hg
                    fx.at[idx, 'FTAG'] = ag
                    n_result += 1
        print(f"  Quote B365 da football-data fuse per chiave: {n_applied}")
        print(f"  RISULTATI da football-data fusi dove openfootball non li ha: {n_result}")
    else:
        print("  football-data non disponibile: le giocate restano senza quote")

    # ── Quote PREPARTITA da TheOddsAPI ──────────────────────────────────
    # Copre solo l'orizzonte vicino (comportamento normale del mercato):
    # le partite future lontane non hanno ancora quote. Le giocate NON
    # vengono sovrascritte: il loro snapshot e' congelato e le quote
    # storiche restano quelle di football-data.
    odds_map = sources.fetch_odds(league)
    n_odds = 0
    for col, lab in (("B365H", "H"), ("B365D", "D"), ("B365A", "A")):
        if col not in fx.columns:
            fx[col] = np.nan
    for idx, row in fx.iterrows():
        ftr = row.get('FTR')
        if isinstance(ftr, str) and ftr.strip().upper() in ('H', 'D', 'A'):
            continue   # gia' giocata: non toccare
        key = (str(row['HomeTeam']), str(row['AwayTeam']),
               pd.Timestamp(row['Date']).strftime('%Y-%m-%d'))
        q = odds_map.get(key)
        if q:
            fx.at[idx, 'B365H'] = q['H']
            fx.at[idx, 'B365D'] = q['D']
            fx.at[idx, 'B365A'] = q['A']
            n_odds += 1
    if n_odds:
        print(f"  Quote prepartita (TheOddsAPI): {n_odds} partite future")

    # ── Ultima giornata giocata (informativa) ─────────────────────────────
    # Il freeze e' stato RIMOSSO: ogni partita viene ricalcolata con il
    # modello di oggi, giocate e future. Resta pero' utile sapere quale e'
    # l'ultima giornata con risultato reale (la dashboard la mostra, e serve
    # al refresh per capire cosa e' gia' stato giocato).
    fx_with_result = fx[fx['FTR'].apply(lambda x: isinstance(x, str) and x.strip().upper() in ('H', 'D', 'A'))]
    fx_ultima_giocata = int(fx_with_result['giornata'].max()) if len(fx_with_result) else 0
    ultima_giocata = fx_ultima_giocata
    print(f"  Ultima giornata giocata: {ultima_giocata} (freeze: DISATTIVATO)")

    # Squadre della Serie A corrente: derivate dal calendario stesso (fonte
    # autorevole), NON hardcodate. Serve per il flag neopromossa: chi e' in
    # questa lista ma ASSENTE dall'ultima stagione storica completa -> promosso.
    current_season_teams = set(fx['HomeTeam']) | set(fx['AwayTeam'])
    print(f"Squadre stagione corrente ({len(current_season_teams)}): "
          f"{sorted(current_season_teams)}")

    # ── Snapshot precedente (solo come riferimento) ───────────────────────
    # Il freeze e' stato rimosso: le partite giocate NON vengono piu' prese
    # dallo snapshot, ma ricalcolate col modello di oggi.
    #
    # Perche' il ricalcolo e' lecito (nessun hindsight): lo split di training
    # e' TEMPORALE (cutoff 2025-08-01), quindi le partite del 2026-27 non
    # sono mai state nel training; e le feature sono calcolate solo su dati
    # PRECEDENTI la partita. Ricalcolare una partita gia' giocata non
    # introduce informazione che non era disponibile al momento della
    # previsione.
    #
    # Resta pero' utile sapere quante stime cambiano rispetto alla
    # pubblicazione precedente: e' il dato che segnala un modello o una
    # feature cambiati.
    prev_published = {}
    prev_model_trained_at = None
    model_changed = False
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON) as f:
                old = json.load(f)
            prev_model_trained_at = old.get("meta", {}).get("model_trained_at")
            for g, ms in old.get("giornate", {}).items():
                for m in ms:
                    prev_published[_match_key(m["home"], m["away"], m["date"])] = m
            model_changed = (prev_model_trained_at != model_trained_at)
            print(f"  Snapshot precedente: {len(prev_published)} partite pubblicate"
                  + (" (modello riaddestrato: le stime si aggiornano)"
                     if model_changed else ""))
        except Exception as e:
            print(f"  [warn] predictions.json precedente illeggibile: {e}")

    # ── Predizioni + bookmaker + kelly ────────────────────────────────
    LABELS = ['H', 'D', 'A']
    h2h_cache = {}
    giornate: dict[str, list] = {}
    bm_rows = []
    model_rows = []
    kelly_log = []
    n_changed = 0

    for _, row in fx.iterrows():
        g = str(row['giornata'])
        m = {"home": row['HomeTeam'], "away": row['AwayTeam'],
             "date": row['Date'].strftime("%Y-%m-%d")}
        ftr = row.get('FTR')
        played = isinstance(ftr, str) and ftr.strip().upper() in ('H', 'D', 'A')

        # Quote bookmaker
        bm_prob = bookmaker_prob(row.get('B365H'), row.get('B365D'), row.get('B365A'))
        bm_odds = None
        if bm_prob is not None:
            try:
                bm_odds = [float(row['B365H']), float(row['B365D']), float(row['B365A'])]
            except (TypeError, ValueError):
                bm_odds = None

        k = _match_key(m["home"], m["away"], m["date"])
        giornata_n = int(row['giornata'])
        # Snapshot precedente: serve solo per contare quante stime cambiano
        # (nessun riuso: il freeze e' rimosso).
        # ── RICALCOLO SEMPRE (FREEZE REMOSSO) ────────────────────────────────
        # Il freeze storico riusava lo snapshot delle partite giocate, cosi'
        # che le percentuali gia' pubblicate non cambiassevano. Rimosso su
        # richiesta esplicita: ogni partita (giocata o futura) viene
        # ricalcolata con il modello e le feature di OGGI.
        #
        # Perche' questo e' lecito (nessun hindsight):
        #  - il training e' TEMPORALE (cutoff 2025-08-01), quindi le partite
        #    del 2026-27 non sono mai state nel training del modello;
        #  - le feature sono calcolate solo su dati PRECEDENTI la partita
        #    (compute_features_for_match usa `df[df['Date'] < match_date]`).
        # Quindi ricalcolare una partita gia' giocata non introduce
        # informazione che non era disponibile al momento della previsione.
        # Le quote del match alimentano le feature di mercato: sono gia' note
        # prima del fischio (apertura dal CSV / TheOddsAPI per le future), quindi
        # non introducono hindsight.
        # TUTTE le fonti di mercato disponibili per questo match, non solo
        # B365: in training il modello vede anche Pinnacle e la media di
        # mercato (Avg), e l'inferenza deve replicare esattamente quel
        # set di feature, altrimenti il modello gira su colonne diverse da
        # quelle su cui e' stato addestrato.
        _odds = {}
        for _label, _cols in (("b365", ('B365H', 'B365D', 'B365A')),
                              ("pinnacle", ('PSH', 'PSD', 'PSA')),
                              ("avg", ('AvgH', 'AvgD', 'AvgA'))):
            try:
                _v = tuple(row.get(c) for c in _cols)
                if all(x is not None and not pd.isna(x) and float(x) > 1.0 for x in _v):
                    _odds[_label] = tuple(float(x) for x in _v)
            except (TypeError, ValueError):
                pass
        feats = compute_features_for_match(history, m["home"], m["away"], row['Date'],
                                           h2h_cache, current_season_teams, odds=_odds)
        # Default NaN per le feature che possono mancare: 0.0 e' un valore
        # FUORI dominio che il training non ha mai visto su quelle colonne
        # (li: NaN = "non lo so"). Con 0.0 il modello riceverebbe un
        # segnale falso su 8-12 colonne di mercato.
        X = pd.DataFrame([{c: (feats.get(c, np.nan) if c.startswith("mkt_")
                                else feats.get(c, 0.0)) for c in FEATURE_ORDER}])
        probs = predict_proba_with_prior(model, X, feats, league=league)
        # Calibrazione Dirichlet: le percentuali pubblicate devono essere
        # quelle CALIBRATE, altrimenti il sito mostrerebbe valori diversi da
        # quelli misurati in evaluate (RPS 0.13044 grezze vs 0.12953 calibrate).
        probs = apply_saved_calibration(league, probs)
        m["prob"] = {LABELS[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
        m["pred"] = LABELS[int(np.argmax(probs))]
        m["features"] = {c: (None if (c.startswith("mkt_") and
                                      (c not in feats or
                                       (isinstance(feats.get(c), float) and
                                        math.isnan(feats.get(c)))))
                         else round(float(feats.get(c, 0.0)), 3))
                         for c in FEATURE_ORDER}

        # Confronto con lo snapshot precedente (informativo): quante stime
        # sono cambiate rispetto a quelle pubblicate prima.
        _prev = prev_published.get(k)
        if _prev and _prev.get("prob"):
            try:
                if any(abs(float(_prev["prob"].get(lab, -1)) - m["prob"][lab]) > 0.05
                       for lab in LABELS):
                    n_changed += 1
            except (TypeError, ValueError):
                pass

        if played:
            ftr = ftr.strip().upper()
            m["risultato"] = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
            m["esito"] = ftr
            m["ok"] = bool(m["pred"] == ftr)

        # Assegnazione quote + miglior giocata Kelly (indipendente da `played`).
        best_info = None
        if _bm_valid({"H": bm_odds[0], "D": bm_odds[1], "A": bm_odds[2]}) if bm_odds else False:
            assert bm_prob is not None
            m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                       "A": round(bm_odds[2], 2)}
            m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
            best_k = 0.0
            for i, lab in enumerate(LABELS):
                pk = m["prob"][lab] / 100.0
                pb = bm_prob[i]
                kf = select_value_bet_stake(pk, pb, bm_odds[i])
                if kf > 0.0 and kf > best_k:
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

        # Confronto bookmaker (RPS) e Kelly retrospettivo: richiedono `played`.
        if bm_odds is not None and played:
            assert bm_prob is not None
            y_idx = {'H': 0, 'D': 1, 'A': 2}[ftr]
            bm_rows.append((y_idx, bm_prob))
            model_rows.append((y_idx, [m["prob"][l] / 100 for l in LABELS]))
            if best_info is not None:
                lab, pk, odd, kf = best_info
                _bm_p = bm_prob[LABELS.index(lab)]
                kelly_log.append({
                    "home": m["home"], "away": m["away"], "date": m["date"],
                    "giornata": g, "esito": ftr,
                    "pick": lab, "model_p": round(pk, 3), "odd": round(odd, 2),
                    "bm_p": round(_bm_p, 4),
                    "kelly": round(kf, 4), "win": (ftr == lab),
                })

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
        # Ricalcola il log con bankroll progressivo, cosi' la pagina puo' mostrare
        # ogni puntata reale (stake in €, quota, esito, bankroll dopo).
        bets = []
        bb = 1000.0
        for i, bet in enumerate(kelly_log, 1):
            stake = bb * bet["kelly"]
            won = bet["win"]
            delta = stake * (bet["odd"] - 1) if won else -stake
            # La puntata e' gia' stata calcolata da calculate_dynamic_kelly_stake
            # sul bankroll di QUESTO step (c_dynamic + hard cap applicati li').
            # Qui si espongono i passaggi, cosi' la pagina puo' mostrare
            # l'aritmetica di ogni puntata invece di un "25%" unico.
            bd = dynamic_kelly_breakdown(bet["model_p"], bet["odd"])
            bets.append({
                "n": i,
                "giornata": bet.get("giornata"),
                "home": bet["home"],
                "away": bet["away"],
                "date": bet["date"],
                "pick": bet["pick"],
                "pick_label": {"H": "Casa", "D": "Pareggio", "A": "Trasferta"}[bet["pick"]],
                "odd": bet["odd"],
                "model_p": round(bet["model_p"] * 100, 1),
                "bm_p": round(bet.get("bm_p", 0) * 100, 1),
                "full_kelly_pct": round(bd["f_full"] * 100, 2) if bd["f_full"] is not None else None,
                "c_dynamic": bd["c_dynamic"],
                "max_cap_pct": round(bd["max_cap"] * 100, 2) if bd["max_cap"] is not None else None,
                "capped": bd["capped"],
                "kelly_pct": round(bet["kelly"] * 100, 2),
                "stake": round(stake, 2),
                "esito": bet.get("esito"),
                "win": bool(won),
                "payout": round(stake * bet["odd"], 2) if won else 0.0,
                "delta": round(delta, 2),
                "bankroll_dopo": round(bb + delta, 2),
            })
            bb += delta
        meta_extra["kelly"] = {
            "bankroll_iniziale": 1000.0,
            "bankroll_finale": round(float(bankroll), 2),
            "profitto": round(float(profit), 2),
            "roi_pct": round(float(roi), 2),
            "scommesse": len(kelly_log),
            "vinte": sum(1 for b in bets if b["win"]),
            "perse": sum(1 for b in bets if not b["win"]),
            "stake_totale_pct": round(float(stakes_total / 1000.0) * 100, 1),
            # Non c'e' piu' una "fraction" unica: il JSON porta i parametri del
            # sizing dinamico, cosi' la UI puo' spiegare la regola senza
            # ripeterla a mano nel testo della pagina.
            "mode": KELLY_MODE,
            "odds_thresholds": list(KELLY_DYNAMIC_CONFIG["odds_thresholds"]),
            "c_dynamic": list(KELLY_DYNAMIC_CONFIG["c_dynamic"]),
            "cap_odds_threshold": KELLY_DYNAMIC_CONFIG["cap_odds_threshold"],
            "max_caps_pct": [round(c * 100, 2) for c in KELLY_DYNAMIC_CONFIG["max_caps"]],
            "capped": sum(1 for b in bets if b["capped"]),
            "scommesse_dettaglio": bets,
        }
        _c = KELLY_DYNAMIC_CONFIG["c_dynamic"]
        print(f"  Kelly dinamico (c_dynamic {_c[0]}/{_c[1]}/{_c[2]}/{_c[3]}): "
              f"{len(kelly_log)} scommesse, bankroll finale €{bankroll:.2f} "
              f"(ROI {roi:.1f}%)")

    # ── JSON-safe: NaN/Infinity NON sono JSON valido. Python li serializza
    # come NaN/Infinity e Safari/WebKit (JSON.parse) rifiuta l'intero file con
    # "The string did not match the expected pattern" (Chrome lo tollera, quindi
    # il bug restava invisibile). Convertiamo in null, che è JSON valido e
    # corrisponde al "valore mancante" che CatBoost si aspetta.
    def _json_safe(o):
        if isinstance(o, dict):
            return {k: _json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_json_safe(v) for v in o]
        if isinstance(o, float):
            return None if (np.isnan(o) or np.isinf(o)) else o
        if o is None or isinstance(o, (bool, int, str)):
            return o
        return o

    out = _json_safe({
        "meta": {
            "last_update": pd.Timestamp.now().isoformat(timespec="seconds"),
            "league": league.key,
            "league_name": league.name,
            "stagione": SEASON_FD,
            "fonte": source,
            "ultima_giornata_giocata": ultima_gg,
            "prossima_giornata": prossima_gg,
            "freeze_attivo": False,
            "freeze_regola": "FREEVE RIMOSSO: ogni partita (giocata o futura) viene ricalcolata col modello di oggi",
            "model_trained_at": model_trained_at,
            **meta_extra,
        },
        "stats": {"corrette": corrette, "total": len(giocate)},
        "giornate": dict(sorted(giornate.items(), key=lambda kv: int(kv[0]))),
    })

    WEBAPP_PUBLIC.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        # allow_nan=False -> solleva invece di scrivere NaN. Belt and braces:
        # _json_safe gia' ripulisce, ma se resta un NaN vogliamo che esploda qui
        # in CI e non sul browser dell'utente.
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"\nGiornate: {len(giornate)} | Giocate: {len(giocate)} "
          f"| Corrette: {corrette} ({(corrette/max(len(giocate),1)*100):.0f}%)")
    print(f"Stime ricalcolate (freeze disattivato): {n_changed} cambiate "
          f"rispetto allo snapshot precedente")
    print(f"Salvato: {OUT_JSON}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Genera le previsioni dal calendario")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    args = ap.parse_args()
    main(get_league(args.league))
