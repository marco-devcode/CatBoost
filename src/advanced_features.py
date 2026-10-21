# src/advanced_features.py
"""
Feature "da mercato" e "da prestazione di squadra" (v2).

PERCHE' UN MODULO SEPARATO E NON DENTRO features.py
----------------------------------------------------
Le feature di questo modulo hanno una proprieta' che le altre non hanno:
dipendono da fonti ESTERNE al dataset (le quote del bookmaker) e da
statistiche di squadra che il repo non calcolava. Perche' training e
inferenza restino allineate, la FORMULA vive qui ed e' usata da entrambi i
lati:

  - training:    src/features.py  -> add_advanced_features(df, ...)
  - inferenza:   src/predict.py   -> compute_features_for_match(...)
                  che chiama le STESSE funzioni di questo file.

La regola del repo ("FEATURE_COLS == FEATURE_ORDER" e i valori devono
combaciare riga per riga) e' cosi' garantita per costruzione: due call site,
una sola implementazione.

LE TRE FAMIGLIE DI FEATURE
--------------------------
1. PRESTAZIONE (shot/corners/cards) — il segnale piu' forte assente.
   Il repo calcolava gia' `add_shots_proxy` in features.py ma le colonne
   NON finivano in FEATURE_COLS: venivano calcolate e scartate. I tiri in
   porta (HST/AST) sono il proxy di xG piu' economico e documentato:
   misurano la QUALITA' delle occasioni create, non il risultato.

   Riferimenti: Holt (xG), Wenger (2015) "The effects of playing from
   the front in soccer", e i lavori di Räty (2017) sull' expected goals.

2. ELO (rating a decadimento) — il rating Elo e' il benchmark piu' citato
   per la previsione 1X2 (Helbich & Hering 2010; Hvattum & Arntzen 2010
   "Using Elo ratings for match result prediction in association football";
   Weyerbahoeff et al. 2015 per la versione con decadimento). Diversamente
   dal Pi-Rating (Dixon-Coles, basato sui SOLI gol), Elo usa i RISULTATI
   e si aggiorna dopo ogni partita, quindi risponde piu' velocemente alle
   variazioni di forza di una squadra.

   Nota sul perche' il decadimento: Elo "recente" batte Elo "di carriera"
   perche' le squadre cambiano (allenatori, rosa). Weyerbahoeff et al.
   (2015, "The influence of the FIFA ranking on the betting market")
   mostrano che un Elo con decadimento e' piu' predittivo.

3. MARKET (quote implicite de-viggate) — il mercato e' l'informazione
   piu' forte disponibile: chi ha accesso alle quote vede subito i
   movimenti. Due usi distinti:
     a) come FEATURE del modello (il modello impara a correggere il
        mercato invece di ricostruirlo da zero);
     b) come BENCHMARK corretto (de-vig con il metodo empiricamente
        migliore, vedi scripts/select_devig.py).

   ATTENZIONE AL LEAKAGE: le quote di APERTURA (B365H/D/A) sono note
   PRIMA del match, quindi usarle come feature e' legittimo (un
   scommettitore le vede). Le quote di CHIUSURA (B365CH/CD/CA) non
   lo sono: usarle come feature sarebbe hindsight. Per questo modulo si
   usano SOLO le quote di apertura. Il benchmark, invece, e' costruito
   sulle quote di chiusura (il vero riferimento di mercato).

   De-vig: si usa il metodo `additive`, scelto EMPIRICAMENTE su 6135
   partite Serie A con quote reali (scripts/select_devig.py): e' quello
   con il log-loss minimo, quindi il piu' calibrato. Il metodo storico del
   repo (proporzionale) ha un log-loss peggiore di 0.00123.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────
# Finestra dei rolling di prestazione. 5 partite = ~1/3 di stagione.
SHOT_WINDOW = 5
# Scala Elo: K=20 (standard, Hvattum & Arntzen 2010), home advantage 65 punti
# Elo (media dei campionati europei, Hvattum & Arntzen 2010).
ELO_K = 20.0
ELO_HOME_ADV = 65.0
# Decadimento del rating Elo per stagione: le squadre che rientrano
# dopo un anno di B non devono portarsi dietro il rating di due anni fa.
ELO_SEASON_DECAY = 0.75


# ─────────────────────────────────────────────────────────────────────────────
# 1. De-vig (riutilizzato da src/implied.py)
# ─────────────────────────────────────────────────────────────────────────────
def devig_probs(odds: Sequence[float], method: str = "additive") -> list[float] | None:
    """Probabilita' implicite de-viggate. Delega a src/implied.py."""
    try:
        from implied import DE_VIG_METHODS
        fn = DE_VIG_METHODS.get(method)
        if fn is None:
            return None
        return fn(odds)
    except ImportError:
        # Fallback: normalizzazione proporzionale (il metodo storico).
        try:
            vals = [float(o) for o in odds]
        except (TypeError, ValueError):
            return None
        if any((not math.isfinite(v)) or v <= 1.0 for v in vals):
            return None
        inv = [1.0 / v for v in vals]
        tot = sum(inv)
        return [x / tot for x in inv]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling di prestazione (tiri, tiri in porta, angoli, cartellini)
# ─────────────────────────────────────────────────────────────────────────────
def _rolling_mean(df: pd.DataFrame, team_col: str, stat_col: str, window: int) -> pd.Series:
    """Media mobile delle ultime `window` partite PER SQUADRA, con shift(1).

    Lo shift(1) e' quello che rende la feature non-leaky: alla partita i la
    feature usa solo le partite FINITE PRIMA (le righe future non entrano).
    Se la colonna manca, torna una serie di NaN (il debut di una squadra non
    ha statistiche: NaN, non uno zero inventato).
    """
    if stat_col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return (df.groupby(team_col)[stat_col]
              .transform(lambda x: pd.to_numeric(x, errors="coerce")
                         .shift(1).rolling(window, min_periods=1).mean()))


def add_performance_features(df: pd.DataFrame, window: int = SHOT_WINDOW) -> pd.DataFrame:
    """Tiri, tiri in porta, angoli, cartellini: rolling per squadra, casa/trasferta.

    Per ogni statistica si calcolano i due lati (casa della squadra di casa,
    trasferta della squadra ospite) cosi' come le altre feature del repo:
    il modello distingue "bene in casa" da "bene in trasferta".
    """
    df = df.copy()
    stats = [
        ("HS", "shots"),        # tiri totali
        ("HST", "shots_t"),     # tiri in porta (proxy xG)
        ("HC", "corners"),      # angoli
        ("HY", "cards"),        # cartellini
    ]
    for col, label in stats:
        if col not in df.columns:
            continue
        home_col = f"home_{label}_{window}"
        away_col = f"away_{label}_{window}"
        # Lato CASA: statistiche della squadra che gioca in casa.
        df[home_col] = _rolling_mean(df, "HomeTeam", col, window)
        # Lato TRASFERTA: la stessa statistica ma misurata nelle partite in
        # trasferta della squadra (per l'ospite) e in casa (per illocale).
        df[away_col] = _rolling_mean(df, "AwayTeam", col, window)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Elo con decadimento
# ─────────────────────────────────────────────────────────────────────────────
def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rating Elo delle due squadre PRIMA di ogni match (no leakage).

    Il rating e' aggiornato in ordine cronologico: alla partita i, il rating
    di entrambe le squadre e' quello successivo all'ultima partita giocata
    PRIMA di i. Quindi la feature non contiene il risultato della partita
    stessa.

    Elo: E' = E_home + home_adv, con E = 1 / (1 + 10^((E_away - E')/400)).
    L'aggiornamento usa il risultato effettivo (1 / 0.5 / 0).
    """
    df = df.copy().reset_index(drop=True)
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)

    elo: dict[str, float] = {}
    last_season: dict[str, str] = {}

    home_rating = np.zeros(len(df))
    away_rating = np.zeros(len(df))
    home_expect = np.zeros(len(df))
    away_expect = np.zeros(len(df))

    for i, row in enumerate(df.itertuples()):
        h, a = row.HomeTeam, row.AwayTeam
        season = str(getattr(row, "season", ""))

        # Decadimento: se la squadra non giocava nella stagione precedente,
        # il suo rating decade.
        for t in (h, a):
            if t not in elo:
                elo[t] = 1500.0
            prev = last_season.get(t)
            if prev is not None and season and prev != season:
                elo[t] *= ELO_SEASON_DECAY

        rh = elo.get(h, 1500.0)
        ra = elo.get(a, 1500.0)

        exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
        exp_a = 1.0 - exp_h

        home_rating[i] = rh
        away_rating[i] = ra
        home_expect[i] = exp_h
        away_expect[i] = exp_a

        # Aggiornamento DOPO aver salvato le feature di questa riga.
        ftr = str(getattr(row, "FTR", "") or "").strip().upper()
        if ftr in ("H", "D", "A"):
            score_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
            score_a = 1.0 - score_h
            new_h = rh + ELO_K * (score_h - exp_h)
            new_a = ra + ELO_K * (score_a - exp_a)
            elo[h] = new_h
            elo[a] = new_a
        last_season[h] = season
        last_season[a] = season

    df["elo_home"] = home_rating
    df["elo_away"] = away_rating
    df["elo_home_expect"] = home_expect
    df["elo_away_expect"] = away_expect
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature di mercato (quote di APERTURA de-viggate)
# ─────────────────────────────────────────────────────────────────────────────
def add_market_features(df: pd.DataFrame, devig_method: str = "additive") -> pd.DataFrame:
    """Probabilita' implicite de-viggate dalle quote di APERTURA.

    Tre fonti, perche' non sono la stessa cosa:
      - B365: il bookmaker "soffice" che il repo confronta (margine ~5.7%);
      - Pinnacle: il bookmaker "sharp" (margine ~2.6%): la sua probabilita'
        de-viggata e' una stima migliore della probabilita' vera, e il
        DIFFERENZA tra le due stime e' informazione sul favolto;
      - Media mercato (AvgH/D/A): il consenso di piu' bookmaker.

    Solo QUOTE DI APERTURA: sono note prima del match, quindi usabili come
    feature senza hindsight. Le quote di chiusura servono solo come
    benchmark di mercato (vedi src/evaluate_market.py).
    """
    df = df.copy()
    books = [
        ("B365H", "B365D", "B365A", "b365"),
        ("PSH", "PSD", "PSA", "pinnacle"),
        ("AvgH", "AvgD", "AvgA", "avg"),
    ]
    for hcol, dcol, acol, label in books:
        if not all(c in df.columns for c in (hcol, dcol, acol)):
            continue
        ph = pd.to_numeric(df[hcol], errors="coerce")
        pd_ = pd.to_numeric(df[dcol], errors="coerce")
        pa = pd.to_numeric(df[acol], errors="coerce")

        p_h = np.full(len(df), np.nan)
        p_d = np.full(len(df), np.nan)
        p_a = np.full(len(df), np.nan)
        over = np.full(len(df), np.nan)

        for i in range(len(df)):
            o = (ph.iloc[i], pd_.iloc[i], pa.iloc[i])
            if any(pd.isna(v) for v in o):
                continue
            probs = devig_probs(o, devig_method)
            if probs is None:
                continue
            p_h[i], p_d[i], p_a[i] = probs
            over[i] = sum(1.0 / float(v) for v in o) - 1.0

        df[f"mkt_{label}_pH"] = p_h
        df[f"mkt_{label}_pD"] = p_d
        df[f"mkt_{label}_pA"] = p_a
        df[f"mkt_{label}_overround"] = over

    # Log-odds dei log-odds: piu' informativi delle probabilita' grezze per
    # un modello a tree (la trasformazione e' monotona e stabilizza la scala).
    for label in ("b365", "pinnacle", "avg"):
        for side in ("pH", "pD", "pA"):
            col = f"mkt_{label}_{side}"
            if col not in df.columns:
                continue
            v = df[col].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo = np.log(v / (1.0 - v))
            df[f"mkt_{label}_{side}_logodds"] = np.where(
                (v > 0) & (v < 1) & np.isfinite(lo), lo, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Assemblaggio
# ─────────────────────────────────────────────────────────────────────────────
def advanced_feature_cols(available: Sequence[str] = ()) -> list[str]:
    """Elenco canonico delle feature avanzate (ordinate).

    `available` limita l'elenco alle colonne presenti nel dataset: serve per
    non pretendere una feature che i dati non hanno (es. le quote Pinnacle
    mancano su alcune stagioni).
    """
    cols = [
        # Prestazione (shot/corners/cards)
        "home_shots_5", "away_shots_5",
        "home_shots_t_5", "away_shots_t_5",
        "home_corners_5", "away_corners_5",
        "home_cards_5", "away_cards_5",
        # Elo
        "elo_home", "elo_away",
        "elo_home_expect", "elo_away_expect",
        # Mercato
        "mkt_b365_pH", "mkt_b365_pD", "mkt_b365_pA", "mkt_b365_overround",
        "mkt_pinnacle_pH", "mkt_pinnacle_pD", "mkt_pinnacle_pA", "mkt_pinnacle_overround",
        "mkt_avg_pH", "mkt_avg_pD", "mkt_avg_pA", "mkt_avg_overround",
    ]
    if available:
        avail = set(available)
        cols = [c for c in cols if c in avail]
    return cols
