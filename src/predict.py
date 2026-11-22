# src/predict.py
"""
Predizione di un singolo match Serie A usando il modello CatBoost allenato.

Calcola le feature (forma recente, Pi-Rating correnti, H2H, rest days,
flag neopromossa) dallo storico disponibile, poi restituisce le probabilita'
H/D/A.

Uso:
    python src/predict.py "Juventus" "Inter"
    python src/predict.py "Milan" "Napoli" --date 2026-09-20

Il flag neopromossa e' calcolato cosi': una squadra e' "neopromossa" nella
stagione corrente se e' presente nelle squadre della stagione corrente ma
ASSENTE dall'ULTIMA stagione COMPLETA nello storico. Il modello impara da solo
lo sconto di rating da applicare (pattern sistematico, non hardcoded).
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
try:
    from catboost import Pool
except ImportError:
    Pool = None

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

WINDOW = 5

# Soglia di saturazione dei giorni di riposo. Unica sorgente di verita': viene
# importata da leagues.py (che la espone a features.py e a predict.py) cosi'
# training e inferenza non possono divergere.
# Vedi la nota lunga su REST_CAP in features.add_rest_days.
from leagues import REST_CAP, get_league  # noqa: E402
from advanced_features import (  # noqa: E402
    ELO_INIT,
    ELO_HOME_ADV,
    ELO_SEASON_DECAY,
    _run_elo_loop,
    add_market_features,
    add_performance_features,
    compute_team_elo_state,
)

# Cache per evitare di ricalcolare lo storico ad ogni match in predict_fixtures.
_HISTORY_CACHE: dict = {}
_PURE_MODEL_CACHE: dict = {}


def _load_pure_model(league):
    if league is None:
        return None
    pure_path = Path(league.model_path).parent / f"{league.model_name}_pure.cbm"
    if not pure_path.exists():
        return None
    key = str(pure_path)
    cached = _PURE_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    m = CatBoostClassifier()
    m.load_model(key)
    _PURE_MODEL_CACHE[key] = m
    return m


def _extract_valid_triplet(feats: dict, prefix: str) -> list[float] | None:
    vals = [feats.get(f"{prefix}pH"), feats.get(f"{prefix}pD"), feats.get(f"{prefix}pA")]
    if all(v is not None and isinstance(v, (int, float)) and np.isfinite(float(v)) and float(v) > 0.0 for v in vals):
        return [float(v) for v in vals]
    return None


def predict_proba_with_prior(model, X_row: pd.DataFrame, feats: dict, league=None) -> np.ndarray:
    """Predice le probabilita' 1X2 con prior continuo sharp (Pinnacle + Avg -> B365 -> puro)."""
    model_feats = list(model.feature_names_) if hasattr(model, "feature_names_") else list(X_row.columns)
    if any(c.startswith("mkt_") for c in model_feats):
        return model.predict_proba(X_row[model_feats])[0]

    p_pin = _extract_valid_triplet(feats, "mkt_pinnacle_")
    p_avg = _extract_valid_triplet(feats, "mkt_avg_")
    p_b365 = _extract_valid_triplet(feats, "mkt_b365_")

    if p_pin is not None and p_avg is not None:
        p_prior = [0.65 * p_pin[i] + 0.35 * p_avg[i] for i in range(3)]
    elif p_pin is not None:
        p_prior = p_pin
    elif p_avg is not None and p_b365 is not None:
        p_prior = [0.50 * p_avg[i] + 0.50 * p_b365[i] for i in range(3)]
    else:
        p_prior = p_b365

    if p_prior is not None:
        base = np.log(np.clip([p_prior], 1e-6, 1.0))
        return model.predict_proba(Pool(X_row[model_feats], baseline=base))[0]

    pure_model = _load_pure_model(league)
    if pure_model is not None:
        pure_feats = list(pure_model.feature_names_) if hasattr(pure_model, "feature_names_") else model_feats
        return pure_model.predict_proba(X_row[pure_feats])[0]

    return model.predict_proba(X_row[model_feats])[0]


def load_history(league=None):
    """Carica il dataset pulito + Pi-Rating di una lega."""
    if league is None:
        league = get_league(None)
    return pd.read_csv(league.ratings_path, parse_dates=['Date'])


def _get_history_cache(df: pd.DataFrame) -> dict:
    """Precalcola una sola volta Pi-Rating, rolling prestazione ed Elo su `df`."""
    min_d = str(df['Date'].min()) if len(df) else ""
    max_d = str(df['Date'].max()) if len(df) else ""
    key = (id(df), len(df), min_d, max_d)
    cached = _HISTORY_CACHE.get(key)
    if cached is not None:
        return cached

    attack, defence, home_adv = ratings_from_history(df)
    perf = add_performance_features(df, window=WINDOW)
    elo, end_elo, end_last_season = _run_elo_loop(df)

    perf_idx = {
        (pd.Timestamp(r.Date), str(r.HomeTeam), str(r.AwayTeam)): pos
        for pos, r in enumerate(perf.itertuples(index=False))
    }
    elo_idx = {
        (pd.Timestamp(r.Date), str(r.HomeTeam), str(r.AwayTeam)): pos
        for pos, r in enumerate(elo.itertuples(index=False))
    }

    pi_arrays = {
        c: df[c].to_numpy(dtype=float)
        for c in ('pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A')
        if c in df.columns
    }

    cached = {
        "attack": attack,
        "defence": defence,
        "home_adv": home_adv,
        "pi_arrays": pi_arrays,
        "perf": perf,
        "perf_idx": perf_idx,
        "elo": elo,
        "elo_idx": elo_idx,
        "end_elo": end_elo,
        "end_last_season": end_last_season,
        "max_date": pd.Timestamp(df['Date'].max()) if len(df) else None,
    }
    if len(_HISTORY_CACHE) >= 8:
        _HISTORY_CACHE.clear()
    _HISTORY_CACHE[key] = cached
    return cached


# ── Ricostruzione di attack/d defence/home_adv dal dataset ──────────────
# Perche' serve (misurato, non ipotizzato):
#   nel TRAINING le colonne pi_* sono la FORMULA (ratings.compute_pi_ratings):
#       pi_home_H = attack[home] - defence[away] + home_adv
#       pi_away_H = attack[away] - defence[home] - home_adv
#       pi_home_A = defence[home]
#       pi_away_A = attack[away]
#   con attack/defence STIMATI UNA VOLTA SULLO STORICO (fit globale), quindi
#   costanti per squadra e indipendenti dalla data.
#
# L'inferenza faceva qualcosa di DIVERSO: leggeva pi_* dall'ULTIMA riga
# storica in cui la squadra e' comparsa. Ma pi_home_H/pi_away_H sono
# valori COPPIA (dipendono anche dall'avversaria di quella riga): riprenderli
# da un'altra partita significa attribuire a Sassuolo-Milan il rating
# calcolato per Sassuolo-Juventus. Per la coppia del 11/10 l'errore era
# +0.14 su pi_home_H e +0.18 su pi_away_H, abbastanza da capovolgere la
# stima (il modello dava 24.8% al Milan, il suo stesso Pi-Rating 50.6%).
#   50/50 righe 2627 e 6130/6130 righe storiche erano discordanti.
#
# Il fix e' ricostruire attack/defence/home_adv dagli stessi numeri del
# training e riapplicare la formula: l'inferenza riceve allora esattamente
# la feature su cui il modello e' stato addestrato.
def ratings_from_history(df):
    """Estrae gli ultimi (attack, defence, home_adv) aggiornati in ordine cronologico."""
    attack, defence, ha_rows = {}, {}, []
    for row in df.itertuples():
        h, a = row.HomeTeam, row.AwayTeam
        defence[h] = float(row.pi_home_A)
        attack[a] = float(row.pi_away_A)
        if h in attack and a in defence:
            ha_rows.append(float(row.pi_home_H) - attack[h] + defence[a])
    home_adv = float(np.mean(ha_rows[-50:])) if ha_rows else 0.0
    return attack, defence, home_adv


def pi_features_from_ratings(attack, defence, home_adv, home, away):
    """Le 4 feature Pi-Rating con la STESSA formula del training."""
    return {
        'pi_home_H': attack.get(home, 0.0) - defence.get(away, 0.0) + home_adv,
        'pi_away_H': attack.get(away, 0.0) - defence.get(home, 0.0) - home_adv,
        'pi_home_A': defence.get(home, 0.0),
        'pi_away_A': attack.get(away, 0.0),
    }


def _season_from_date(match_date) -> int:
    """Converte una data nel codice stagione integer (es. 2026-09-20 -> 2627)."""
    dt = pd.Timestamp(match_date)
    start = dt.year if dt.month >= 7 else dt.year - 1
    return int(f"{start % 100:02d}{(start + 1) % 100:02d}")


def is_promoted(history, team, match_date, current_season_teams=None):
    """True se `team` e' neopromossa nella stagione di match_date.

    Replica esattamente la logica di features.add_promotion_flags: una squadra
    e' neopromossa nella stagione `target_s` se e' assente dalla stagione
    immediatamente precedente presente nello storico (< target_s).
    """
    if len(history) == 0:
        return 0
    seasons = sorted(int(s) for s in history['season'].unique())
    if match_date is not None:
        target_s = _season_from_date(match_date)
        prev_seasons = [s for s in seasons if s < target_s]
        if not prev_seasons:
            return 0  # prima stagione storica: nessuna precedente -> 0
        prev_s = prev_seasons[-1]
    else:
        if current_season_teams is not None and len(seasons) >= 2:
            last_sub = history[history['season'] == seasons[-1]]
            last_set = set(last_sub['HomeTeam']) | set(last_sub['AwayTeam'])
            prev_s = seasons[-2] if last_set <= set(current_season_teams) else seasons[-1]
        else:
            prev_s = seasons[-1]

    past = history[history['season'] == prev_s]
    seen = len(past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)]) > 0
    return 0 if seen else 1


def _normalizza_odds(odds):
    """Accetta (h,d,a) o dict per bookmaker e restituisce il dict normalizzato."""
    if odds is None:
        return {}
    if isinstance(odds, dict):
        out = {}
        for k, v in odds.items():
            if v is None:
                continue
            try:
                vals = tuple(float(x) for x in v)
            except (TypeError, ValueError):
                continue
            if len(vals) == 3 and all(np.isfinite(x) and x > 1.0 for x in vals):
                out[str(k).lower()] = vals
        return out
    try:
        vals = tuple(float(x) for x in odds)
    except (TypeError, ValueError):
        return {}
    if len(vals) == 3 and all(np.isfinite(x) and x > 1.0 for x in vals):
        return {"b365": vals}
    return {}


def _g(odds_by_book, book, idx):
    v = odds_by_book.get(book)
    return float(v[idx]) if v else np.nan


def compute_features_for_match(df, home, away, match_date, h2h_cache=None,
                               current_season_teams=None, odds=None):
    """Calcola le feature di un match usando SOLO dati precedenti a match_date.

    `odds` = (h, d, a) del bookmaker per QUESTO match, se disponibili. Serve
    alle feature di mercato: in training arrivano dal CSV (quote di apertura),
    in inferenza da TheOddsAPI. Se assenti, le feature di mercato valgono NaN
    (il modello le tratta come mancanti, esattamente come i debut).
    """
    match_ts = pd.Timestamp(match_date) if match_date is not None else None
    past = df[df['Date'] < match_ts] if match_ts is not None else df

    feats = {}
    cache = _get_history_cache(df)
    mkey = (match_ts, str(home), str(away)) if match_ts is not None else None
    ppos = cache["perf_idx"].get(mkey) if mkey is not None else None

    # Pi-Rating: se la partita e' gia' nello storico usa i valori walk-forward
    # esatti di quella data; per le future usa i rating piu' recenti.
    if ppos is not None and len(cache["pi_arrays"]) == 4:
        for c, arr in cache["pi_arrays"].items():
            feats[c] = float(arr[ppos])
    else:
        feats.update(pi_features_from_ratings(
            cache["attack"], cache["defence"], cache["home_adv"], home, away
        ))

    # Forma ibrida: 60% ruolo specifico (casa/trasferta) + 40% forma globale (casa + trasferta)
    def team_form(team, is_home):
        tcol, scol, acol = ('HomeTeam', 'FTHG', 'FTAG') if is_home else ('AwayTeam', 'FTAG', 'FTHG')
        sub_v = past[past[tcol] == team].tail(WINDOW)
        if len(sub_v) == 0:
            return np.nan
        pts_v = np.where(sub_v[scol].values > sub_v[acol].values, 3.0,
                         np.where(sub_v[scol].values == sub_v[acol].values, 1.0, 0.0))
        venue_mean = float(np.mean(pts_v))

        sub_all = past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)].tail(WINDOW)
        pts_all = np.where(
            sub_all['HomeTeam'].values == team,
            np.where(sub_all['FTHG'].values > sub_all['FTAG'].values, 3.0,
                     np.where(sub_all['FTHG'].values == sub_all['FTAG'].values, 1.0, 0.0)),
            np.where(sub_all['FTAG'].values > sub_all['FTHG'].values, 3.0,
                     np.where(sub_all['FTAG'].values == sub_all['FTHG'].values, 1.0, 0.0)),
        )
        all_mean = float(np.mean(pts_all))
        return 0.60 * venue_mean + 0.40 * all_mean

    feats['home_form_5'] = team_form(home, True)
    feats['away_form_5'] = team_form(away, False)

    # Goal diff rolling
    def team_goaldiff(team, is_home):
        tcol, scol, acol = ('HomeTeam', 'FTHG', 'FTAG') if is_home else ('AwayTeam', 'FTAG', 'FTHG')
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0:
            return np.nan   # stessa semantica di team_form: vedi sopra
        return float(np.mean(sub[scol].values - sub[acol].values))

    feats['home_goal_diff_5'] = team_goaldiff(home, True)
    feats['away_goal_diff_5'] = team_goaldiff(away, False)

    # Rest days (approssimato: giorni dall'ultima partita di ciascuna squadra)
    # Saturati a REST_CAP: oltre la soglia il recupero e' completo e i giorni
    # in piu' non devono cambiare la stima. Deve corrispondere ESATTAMENTE a
    # features.add_rest_days, altrimenti training e inferenza vedono la stessa
    # feature con significato diverso.
    def rest_days(team):
        sub = past[(past['HomeTeam'] == team) | (past['AwayTeam'] == team)].sort_values('Date')
        if len(sub) == 0 or match_date is None:
            return float(REST_CAP)
        last = sub['Date'].max()
        try:
            return float(min((match_date - last).days, REST_CAP))
        except Exception:
            return float(REST_CAP)

    # rest_advantage NON e' una feature del modello: e' esattamente
    # home_rest_days - away_rest_days (verificato: coincide su tutte le righe),
    # quindi non aggiunge informazione. Il modello riceve solo i due livelli
    # assoluti, da cui ricava il vantaggio da solo.
    feats['home_rest_days'] = rest_days(home)
    feats['away_rest_days'] = rest_days(away)

    # Rolling goals (replica di features.add_rolling_goals): media gol fatti/subiti
    # negli ultimi 5 match CASA per la di casa, TRASFERTA per l'ospite.
    # Il match futuro non e' nel 'past', quindi tail(5).mean() equivale allo shift(1).
    def rolling_goals(team, is_home):
        if is_home:
            sub = past[past['HomeTeam'] == team].tail(WINDOW)
            scored_col, conceded_col = 'FTHG', 'FTAG'
        else:
            sub = past[past['AwayTeam'] == team].tail(WINDOW)
            scored_col, conceded_col = 'FTAG', 'FTHG'
        if len(sub) == 0:
            return np.nan, np.nan  # come in training (debutti -> NaN, gestito da CatBoost)
        return float(sub[scored_col].mean()), float(sub[conceded_col].mean())

    hs_, hc_ = rolling_goals(home, True)
    as_, ac_ = rolling_goals(away, False)
    feats['home_goals_scored_5'] = hs_
    feats['home_goals_conceded_5'] = hc_
    feats['away_goals_scored_5'] = as_
    feats['away_goals_conceded_5'] = ac_

    # H2H: win rate della squadra `home` negli ultimi WINDOW scontri diretti
    # con `away` (replica features.add_h2h). La chiave di cache include home,
    # away e data per evitare riutilizzi errati tra andata e ritorno.
    cache_key = (home, away, str(match_ts)) if match_ts is not None else (home, away, None)
    if h2h_cache is not None and cache_key in h2h_cache:
        feats['h2h_home_winrate'] = h2h_cache[cache_key]
    else:
        sub = past[((past['HomeTeam'] == home) & (past['AwayTeam'] == away)) |
                   ((past['HomeTeam'] == away) & (past['AwayTeam'] == home))].tail(WINDOW)
        if len(sub) == 0:
            feats['h2h_home_winrate'] = 0.4
        else:
            home_wins = (
                ((sub['HomeTeam'] == home) & (sub['FTR'] == 'H'))
                | ((sub['AwayTeam'] == home) & (sub['FTR'] == 'A'))
            )
            feats['h2h_home_winrate'] = float(home_wins.mean())
        if h2h_cache is not None:
            h2h_cache[cache_key] = feats['h2h_home_winrate']

    # Flag neopromossa: il modello impara lo sconto da solo. Squadre appena
    # promosse (presenti nella stagione corrente ma assenti dalla stagione
    # precedente dello storico) -> 1.
    feats['home_promoted'] = is_promoted(df, home, match_ts, current_season_teams)
    feats['away_promoted'] = is_promoted(df, away, match_ts, current_season_teams)

    # ── Mercato: quote del match de-viggate ─────────────────────────────
    # Stessa formula del training (advanced_features.add_market_features),
    # applicata a una riga con le quote di QUESTO match.
    # `odds` puo' essere:
    #   - una tupla (h, d, a): solo B365 (retrocompatibile);
    #   - un dict {"b365": (h,d,a), "pinnacle": (h,d,a), "avg": (h,d,a)}:
    #     tutte le fonti disponibili, cosi' l'inferenza reproduce ESATTAMENTE
    #     il training (che ha Pinnacle e Avg dal CSV storico).
    # Le fonti assenti restano NaN, che e' il valore che il training ha per
    # le partite senza quelle quote: il modello le tratta come mancanti.
    odds_by_book = _normalizza_odds(odds)
    if odds_by_book:
        try:
            row = pd.DataFrame([{
                "B365H": _g(odds_by_book, "b365", 0), "B365D": _g(odds_by_book, "b365", 1),
                "B365A": _g(odds_by_book, "b365", 2),
                "PSH": _g(odds_by_book, "pinnacle", 0), "PSD": _g(odds_by_book, "pinnacle", 1),
                "PSA": _g(odds_by_book, "pinnacle", 2),
                "AvgH": _g(odds_by_book, "avg", 0), "AvgD": _g(odds_by_book, "avg", 1),
                "AvgA": _g(odds_by_book, "avg", 2),
            }])
            mkt = add_market_features(row, devig_method="additive")
            for col in mkt.columns:
                if col.startswith("mkt_"):
                    feats[col] = float(mkt[col].iloc[0])
        except Exception:
            pass

    # Le feature di mercato mancanti devono essere NaN, non 0.0: nel training
    # la loro assenza e' NaN ("non lo so"), e 0.0 e' un valore FUORI dominio
    # che il modello non ha mai visto su quelle colonne.
    for col in ("mkt_b365_pH", "mkt_b365_pD", "mkt_b365_pA", "mkt_b365_overround",
                "mkt_pinnacle_pH", "mkt_pinnacle_pD", "mkt_pinnacle_pA",
                "mkt_pinnacle_overround",
                "mkt_avg_pH", "mkt_avg_pD", "mkt_avg_pA", "mkt_avg_overround"):
        if col not in feats:
            feats[col] = np.nan

    # ── Feature avanzate (prestazione, Elo, mercato) ────────────────────────
    # STESSE funzioni usate in training (src/features.py -> advanced_features):
    # una sola implementazione, quindi parita' per costruzione.
    #
    # Il trucco e' ricostruire il "passato" della squadra come se fosse una
    # riga del dataset: si aggiungono le colonne di prestazione ed Elo sul
    # dataframe storico, poi si prende la riga della partita da predire.
    feats.update(_advanced_features_for_match(df, home, away, match_ts, cache=cache))

    return feats


def _adjust_elo_for_season(team: str, elo_map: dict, last_season_map: dict,
                           target_season: str | None) -> float:
    val = elo_map.get(team, ELO_INIT)
    prev = last_season_map.get(team)
    if target_season and prev is not None and str(prev) != str(target_season):
        return ELO_INIT + ELO_SEASON_DECAY * (val - ELO_INIT)
    return val


def _advanced_features_for_match(df, home, away, match_date, odds=None, cache=None):
    """Feature avanzate per un singolo match, calcolate sulle stesse formule
    del training (advanced_features.add_*).
    """
    out = {}
    if cache is None:
        cache = _get_history_cache(df)

    match_ts = pd.Timestamp(match_date) if match_date is not None else None
    mkey = (match_ts, str(home), str(away)) if match_ts is not None else None

    # ── Prestazione (tiri, tiri in porta, angoli, cartellini) ───────────
    perf = cache["perf"]
    ppos = cache["perf_idx"].get(mkey) if mkey is not None else None
    for col in ("home_shots_5", "away_shots_5", "home_shots_t_5", "away_shots_t_5",
                "home_corners_5", "away_corners_5", "home_cards_5", "away_cards_5"):
        if col not in perf.columns:
            continue
        if ppos is not None:
            out[col] = float(perf[col].iloc[ppos])
        else:
            out[col] = _rolling_future(perf, df, home, away, match_ts, col)

    # ── Elo ──────────────────────────────────────────────────────────────
    elo = cache["elo"]
    epos = cache["elo_idx"].get(mkey) if mkey is not None else None
    if epos is not None:
        row = elo.iloc[epos]
        for col in ("elo_home", "elo_away", "elo_home_expect", "elo_away_expect"):
            if col in elo.columns:
                out[col] = float(row[col])
        return out

    # Match futuro o non presente in df: usa il vero rating post-partita fino a match_date.
    target_season = str(_season_from_date(match_ts)) if match_ts is not None else None
    max_date = cache["max_date"]
    if match_ts is None or (max_date is not None and match_ts > max_date):
        elo_map = cache["end_elo"]
        last_season_map = cache["end_last_season"]
        rh = _adjust_elo_for_season(home, elo_map, last_season_map, target_season)
        ra = _adjust_elo_for_season(away, elo_map, last_season_map, target_season)
    else:
        past = df[df["Date"] < match_ts]
        if len(past) == 0:
            rh = ELO_INIT
            ra = ELO_INIT
        else:
            elo_map, _ = compute_team_elo_state(past, target_season=target_season)
            rh = float(elo_map.get(home, ELO_INIT))
            ra = float(elo_map.get(away, ELO_INIT))

    exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
    out["elo_home"] = rh
    out["elo_away"] = ra
    out["elo_home_expect"] = exp_h
    out["elo_away_expect"] = 1.0 - exp_h

    return out


def _rolling_future(perf, df, home, away, match_date, col):
    """Rolling di prestazione per un match FUTURO: media delle ultime N
    partite della squadra in quel ruolo (casa per il locale, trasferta per
    l'ospite), replicando il shift(1) del training."""
    past = perf[perf["Date"] < match_date] if match_date is not None else perf
    if col.startswith("home_"):
        sub = past[past["HomeTeam"] == home].tail(5)
        stat = col.replace("home_", "").rsplit("_", 1)[0]
        mapping = {"shots": "HS", "shots_t": "HST", "corners": "HC", "cards": "HY"}
        src_col = mapping.get(stat)
    else:
        sub = past[past["AwayTeam"] == away].tail(5)
        stat = col.replace("away_", "").rsplit("_", 1)[0]
        mapping = {"shots": "AS", "shots_t": "AST", "corners": "AC", "cards": "AY"}
        src_col = mapping.get(stat)
    if src_col is None or src_col not in past.columns or len(sub) == 0:
        return np.nan
    vals = pd.to_numeric(sub[src_col], errors="coerce")
    return float(vals.mean())


# L'elenco delle feature e' costruito come in training (features.py ->
# build_feature_list): base + avanzate PRESENTI nel dataset. Cosi' i due
# lati non possono divergere per costruzione, e il check di allineamento in
# features.py fallisce in CI se qualcosa non torna.
#
# Nota: rest_advantage NON e' inclusa perche' e' esattamente
# home_rest_days - away_rest_days (ridondante deterministica).
def _build_feature_order(available_cols=None) -> list:
    from features import build_feature_list
    return build_feature_list(available_cols)


# Default: le feature base + le avanzate che il dataset storico contiene.
#
# Il dataset non e' sempre disponibile all'import (in alcuni ambienti il
# ratings.csv non c'e' ancora), quindi il default e' l'elenco BASE. Chi
# carica il modello (predict_fixtures, predict_match) usa
# model.feature_names_, che e' l'elenco esatto del training: e' la fonte
# autorevole, e non puo' divergere dal modello.
FEATURE_ORDER = _build_feature_order(None)


def apply_saved_calibration(league, probs):
    """Applica la calibrazione Dirichlet salvata dal training, se esiste.

    La calibrazione e' un array alpha + un peso w: le probabilita' grezze
    del modello vengono ricalcolate come
        p_k = softmax(alpha_k + w * log p_k)
    Se il file non c'e' (modello allenato senza calibrazione) le
    probabilita' restano quelle grezze.
    """
    import json as _json
    from pathlib import Path as _Path
    cal_path = _Path(league.model_path).parent / f"calibration_{league.key}.json"
    if not cal_path.exists():
        return probs
    try:
        with open(cal_path) as f:
            cal = _json.load(f)
        from experiments import apply_dirichlet_calibration
        return apply_dirichlet_calibration(np.asarray([probs]), cal["alpha"], cal["w"])[0]
    except Exception:
        return probs


def predict_match(home, away, match_date=None, league=None,
                  current_season_teams=None):
    if league is None:
        league = get_league(None)
    elif isinstance(league, str):
        league = get_league(league)
    model_path = league.model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello non trovato in {model_path}. Esegui prima src/model.py "
            f"--league {league.key} (o la GitHub Action) per allenarlo."
        )

    df = load_history(league)
    if match_date:
        match_date = pd.Timestamp(match_date)

    feats = compute_features_for_match(df, home, away, match_date,
                                        current_season_teams=current_season_teams)

    # Carica il modello PRIMA di leggere feature_names_.
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    feature_order = list(model.feature_names_) if hasattr(model, "feature_names_") else FEATURE_ORDER
    X = pd.DataFrame([{c: (feats.get(c, np.nan) if c.startswith("mkt_")
                           else feats.get(c, 0.0)) for c in feature_order}])

    probs = predict_proba_with_prior(model, X, feats, league=league)

    # Calibrazione Dirichlet (se presente): e' un post-processing delle
    # probabilita' che riduce l'overconfidence del modello. Va applicata
   # SEMPRE, altrimenti il sito pubblicherebbe probabilita' diverse da quelle
    # misurate (RPS 0.13044 grezze vs 0.12953 calibrate).
    probs = apply_saved_calibration(league, probs)

    labels = ['H', 'D', 'A']
    out = {labels[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
    return out, feats


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Predici un match")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--date", default=None,
                    help="Data match (YYYY-MM-DD) per il contesto temporale")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    args = ap.parse_args()

    league = get_league(args.league)
    try:
        out, feats = predict_match(args.home, args.away, args.date, league)
    except FileNotFoundError as e:
        print(f"ERRORE: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  {league.name}")
    print(f"  {args.home}  vs  {args.away}")
    if args.date:
        print(f"  (contesto: {args.date})")
    print(f"{'='*50}")
    for k in ['H', 'D', 'A']:
        bar = '█' * int(out[k] / 2.5)
        print(f"  {k}: {out[k]:5.1f}%  {bar}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
