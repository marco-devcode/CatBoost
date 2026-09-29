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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league, WEBAPP_PUBLIC  # noqa: E402
import sources  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
WEBAPP_PUBLIC = WEBAPP_PUBLIC

KELLY_FRACTION = 0.25      # Kelly ridotto al 25% per gestione del rischio

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

    from predict import (load_history, compute_features_for_match, FEATURE_ORDER,
                         is_promoted)

    history = load_history(league)
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

    # ── Quote B365 dalle partite GIA GIOCATE (football-data) ─────────
    # football-data pubblica le quote solo per le partite giocate; per le
    # future arriva TheOddsAPI. openfootball non ha quote.
    if fd is not None and 'B365H' in fx.columns:
        n_b365 = fx['B365H'].notna().sum()
        if n_b365:
            print(f"  Quote B365 (partite giocate): {n_b365}")

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

    # ── Soglia di freeze (regola N-1) ───────────────────────────────────
    # Una giornata N e' "bloccabile" (snapshot non piu ricalcolato) non appena
    # la giornata N-1 ha almeno un risultato reale. Quindi le giornate con
    # numero <= (ultima_giornata_giocata + 1) sono CONGELATE: le loro
    # percentuali non si aggiornano piu (no hindsight bias).
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
    # Il freeze e' SEMPRE attivo per le partite con risultato reale, anche se il
    # modello e' stato riaddestrato nel frattempo. Ricalcolare le percentuali
    # di una partita gia' giocata userebbe informazioni che al momento della
    # previsione non esistevano (hindsight bias), quindi lo snapshot resta.
    # prev_estimates contiene solo partite giocate: la chiave e' data+squadre,
    # cosi' un cambio di data non crea una partita nuova con lo stesso nome.
    prev_estimates = {}
    prev_model_trained_at = None
    model_changed = False
    if OUT_JSON.exists():
        try:
            with open(OUT_JSON) as f:
                old = json.load(f)
            prev_model_trained_at = old.get("meta", {}).get("model_trained_at")
            for g, ms in old.get("giornate", {}).items():
                for m in ms:
                    if m.get("ok") is not None:
                        prev_estimates[_match_key(m["home"], m["away"], m["date"])] = m
            model_changed = (prev_model_trained_at != model_trained_at)
            print(f"  Freeze ATTIVO su {len(prev_estimates)} partite giocate"
                  + (" (modello riaddestrato: gli snapshot restano invariati)"
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
        # ── SNAPSHOT / FREEZE ─────────────────────────────────────────────
        if k in prev_estimates and (played or giornata_n <= freeze_until):
            old = prev_estimates[k]
            m["prob"] = old.get("prob")
            m["pred"] = old.get("pred")
            m["bm"] = old.get("bm")
            m["bm_prob"] = old.get("bm_prob")
            # Lo snapshot puo' contenere bm = {"H": None, "D": None, "A": None}:
            # un dict PRESENTE ma vuoto (NaN serializzato in null). Copiato
            # cosi' com'e' arrivava al frontend, dove `m.bm && m.bm_prob` e'
            # truthy e .toFixed(0) faceva TypeError — pagina bianca su 63
            # partite. La normalizzazione sotto valida i VALORI, quindi qui si
            # azzera subito quello che non e' numerico: niente dict-vuoto in
            # giro.
            m["kelly"] = old.get("kelly")
            m["risultato"] = old.get("risultato")
            m["esito"] = old.get("esito")
            m["ok"] = old.get("ok")
            m["features"] = old.get("features")
            # Se lo snapshot non aveva le quote ma il CSV/le quote di oggi le
            # portano, si ricostruiscono. Il caso reale (Eredivisie, prima
            # run) era peggio: lo snapshot aveva bm = {"H": None, "D": None,
            # "A": None} — un dict PRESENTE ma con valori null, che il test
            # `bm is not None` lascia passare e poi fa None/100.0 -> la
            # pipeline muore dopo aver allenato il modello. Percio' qui si
            # controlla che i valori siano NUMERICI, non che il dict esista.
            def _bm_valido(d):
                return (isinstance(d, dict)
                        and all(isinstance(d.get(l), (int, float)) for l in LABELS))

            if not _bm_valido(m.get("bm")) and bm_odds is not None:
                m["bm"] = {"H": round(bm_odds[0], 2), "D": round(bm_odds[1], 2),
                           "A": round(bm_odds[2], 2)}
            if not _bm_valido(m.get("bm_prob")) and bm_prob is not None:
                m["bm_prob"] = {LABELS[i]: round(bm_prob[i] * 100, 1) for i in range(3)}
            # Quote senza probabilita' non sono confrontabili: si azzerano
            # insieme, altrimenti la UI mostrerebbe una quota senza edge.
            if not _bm_valido(m.get("bm")) or not _bm_valido(m.get("bm_prob")):
                m["bm"] = None
                m["bm_prob"] = None
            if (m["bm"] is not None and m["ok"] is not None and m["esito"] is not None
                    and all(m["prob"].get(l) is not None for l in LABELS)):
                y_idx = {'H': 0, 'D': 1, 'A': 2}[m["esito"]]
                pb = np.array([m["bm_prob"][l] / 100.0 for l in LABELS])
                bm_rows.append((y_idx, pb))
                model_rows.append((y_idx, [m["prob"][l] / 100.0 for l in LABELS]))
                best_k = -1.0
                best_info = None
                for i, lab in enumerate(LABELS):
                    pk = m["prob"][lab] / 100.0
                    if pk > pb[i]:
                        kf = kelly_fraction(pk, m["bm"][lab])
                        if kf > 0 and kf > best_k:
                            best_k = kf
                            best_info = (lab, pk, m["bm"][lab], kf)
                if best_info is not None:
                    lab, pk, odd, kf = best_info
                    _bm_p = bm_prob[LABELS.index(lab)] if bm_prob is not None else 0.0
                    kelly_log.append({
                        "home": m["home"], "away": m["away"], "date": m["date"],
                        "giornata": g, "esito": m.get("esito"),
                        "pick": lab, "model_p": round(pk, 3), "odd": round(odd, 2),
                        "bm_p": round(_bm_p, 4),
                        "kelly": round(kf, 4), "win": (m["esito"] == lab),
                    })
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
                    if pk > pb:
                        kf = kelly_fraction(pk, bm_odds[i])
                        if kf > 0 and kf > best_k:
                            best_k = kf
                            best_info = (lab, pk, bm_odds[i], kf)
                if best_info is not None:
                    lab, pk, odd, kf = best_info
                    win = (ftr == lab)
                    kelly_log.append({
                        "home": m["home"], "away": m["away"], "date": m["date"],
                        "giornata": g, "esito": ftr,
                        "pick": lab, "model_p": round(pk, 3), "odd": round(odd, 2),
                        "bm_p": round(bm_prob[LABELS.index(lab)], 4) if bm_prob is not None else 0.0,
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
        # Ricalcola il log con bankroll progressivo, cosi' la pagina puo' mostrare
        # ogni puntata reale (stake in €, quota, esito, bankroll dopo).
        bets = []
        bb = 1000.0
        for i, bet in enumerate(kelly_log, 1):
            stake = bb * bet["kelly"]
            won = bet["win"]
            delta = stake * (bet["odd"] - 1) if won else -stake
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
            "fraction": KELLY_FRACTION,
            "scommesse_dettaglio": bets,
        }
        print(f"  Kelly (fraction={KELLY_FRACTION}): {len(kelly_log)} scommesse, "
              f"bankroll finale €{bankroll:.2f} (ROI {roi:.1f}%)")

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
            "freeze_until": freeze_until,
            "freeze_regola": "La giornata N si aggiorna solo fino ai risultati della N-1 (no hindsight)",
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
    print(f"Salvato: {OUT_JSON}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Genera le previsioni dal calendario")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    main(get_league(ap.parse_args().league))
