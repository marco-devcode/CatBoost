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
    add_elo_features,
    add_market_features,
    add_performance_features,
    advanced_feature_cols,
)


def load_history(league):
    """Carica il dataset pulito + Pi-Rating di una lega."""
    return pd.read_csv(league.ratings_path, parse_dates=['Date'])


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
    """Estrae (attack, defence, home_adv) dal dataset con le colonne pi_*.

    Il dataset contiene gia' i termini della formula, quindi si legge al
    contrario senza dover rifare il fit MLE:
      defence[t] = pi_home_A  su una riga dove t e' in casa
      attack[t]  = pi_away_A  su una riga dove t e' e' in trasferta
      home_adv   = pi_home_H - attack[home] + defence[away]
    Verificato: ricostruisce il CSV con scarto max 1.2e-4 (arrotondamento
    del JSON a 4 decimali, non errore di calcolo).
    """
    attack, defence, ha_rows = {}, {}, []
    for row in df.itertuples():
        h, a = row.HomeTeam, row.AwayTeam
        if h not in defence:
            defence[h] = float(row.pi_home_A)
        if a not in attack:
            attack[a] = float(row.pi_away_A)
        if h in attack and a in defence:
            ha_rows.append(float(row.pi_home_H) - attack[h] + defence[a])
    home_adv = float(np.mean(ha_rows)) if ha_rows else 0.0
    return attack, defence, home_adv


def pi_features_from_ratings(attack, defence, home_adv, home, away):
    """Le 4 feature Pi-Rating con la STESSA formula del training."""
    return {
        'pi_home_H': attack.get(home, 0.0) - defence.get(away, 0.0) + home_adv,
        'pi_away_H': attack.get(away, 0.0) - defence.get(home, 0.0) - home_adv,
        'pi_home_A': defence.get(home, 0.0),
        'pi_away_A': attack.get(away, 0.0),
    }


def _last_complete_season_teams(history):
    """Insieme delle squadre nell'ULTIMA stagione presente nello storico."""
    if len(history) == 0:
        return set()
    last_season = sorted(history['season'].unique())[-1]
    sub = history[history['season'] == last_season]
    return (set(sub['HomeTeam']) | set(sub['AwayTeam']))


def is_promoted(history, team, match_date, current_season_teams=None):
    """True se `team` e' neopromossa rispetto alla stagione di match_date.

    Senza `current_season_teams`: logica storica pura (assente nella stagione
    immediatamente precedente a match_date).
    Con `current_season_teams` (insieme delle squadre della Serie A corrente,
    es. derivato dal calendario): neopromossa = presente in current ma ASSENTE
    dall'ultima stagione COMPLETA nello storico. Questo gestisce correttamente
    il caso inizio 26/27, dove lo storico si ferma al 25/26 e il Venezia appena
    promosso non ha Pi-Rating freschi.
    """
    last_teams = _last_complete_season_teams(history)
    if current_season_teams is not None:
        if team not in current_season_teams:
            return 0  # non e' in Serie A quest'anno -> non marcare
        return 0 if team in last_teams else 1
    # fallback: logica stagione-precedente
    if match_date is None:
        return 0
    prev_year = match_date.year - 1
    prev_season = int(f"{prev_year % 100:02d}{(prev_year + 1) % 100:02d}") \
        if match_date.month < 7 else \
        int(f"{(match_date.year - 1) % 100:02d}{match_date.year % 100:02d}")
    past = history[history['season'] == prev_season]
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
    past = df[df['Date'] < match_date].copy() if match_date else df.copy()

    if len(past) == 0:
        past = df.copy()

    feats = {}

    # Pi-Rating: STESSA formula del training (ratings.compute_pi_ratings).
    # Vedi la nota lunga su ratings_from_history: leggere pi_* dall'ultima
    # riga storica della squadra attribuiva alla partita corrente il rating
    # calcolato per una COPPIA DIVERSA (l'ultima avversaria incontrata).
    attack, defence, home_adv = ratings_from_history(df)
    feats.update(pi_features_from_ratings(attack, defence, home_adv, home, away))

    # Forma recente (ultimi WINDOW match della squadra, casa o trasferta)
    def team_form(team, is_home):
        tcol, scol, acol = ('HomeTeam', 'FTHG', 'FTAG') if is_home else ('AwayTeam', 'FTAG', 'FTHG')
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0:
            # NaN, non un prior numerico: features.add_rolling_form usa
            # shift(1).rolling(5, min_periods=1) sulle partite di CASA della
            # squadra, quindi alla sua prima presenza in quel ruolo la
            # feature manca. Qui l'inferenza restituiva 1.0, un valore che il
            # modello non ha MAI visto in quella posizione: 14 righe su 600
            # campionate differivano dal training (tutte debut di stagione).
            return np.nan
        pts = np.where(sub[scol].values > sub[acol].values, 3,
                      np.where(sub[scol].values == sub[acol].values, 1, 0))
        return float(np.mean(pts))

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

    # H2H: win rate CASA negli ultimi WINDOW scontri diretti.
    # Deve replicare features.add_h2h, che conta le partite CONCLUSE DA 'H'
    # (vittoria della squadra di casa di QUELLA partita), non le vittorie
    # della squadra nominata home in questo match. Le due definizioni
    # divergono: su 37 righe su 50 della 26/27 la versione qui sotto dava
    # un numero diverso dal training, quindi il modello vedeva una feature
    # h2h che non aveva mai visto addestrata.
    key = (min(home, away), max(home, away))
    if h2h_cache is not None and key in h2h_cache:
        feats['h2h_home_winrate'] = h2h_cache[key]
    else:
        sub = past[((past['HomeTeam'] == home) & (past['AwayTeam'] == away)) |
                   ((past['HomeTeam'] == away) & (past['AwayTeam'] == home))].tail(WINDOW)
        if len(sub) == 0:
            feats['h2h_home_winrate'] = 0.4
        else:
            feats['h2h_home_winrate'] = float((sub['FTR'] == 'H').mean())
        if h2h_cache is not None:
            h2h_cache[key] = feats['h2h_home_winrate']

    # Flag neopromossa: il modello impara lo sconto da solo. Squadre appena
    # promosse (presenti nella stagione corrente ma assenti dall'ultima stagione
    # storica completa, es. Venezia 26/27) -> 1.
    feats['home_promoted'] = is_promoted(df, home, match_date, current_season_teams)
    feats['away_promoted'] = is_promoted(df, away, match_date, current_season_teams)

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
    feats.update(_advanced_features_for_match(df, home, away, match_date, odds))

    return feats


def _advanced_features_for_match(df, home, away, match_date, odds=None):
    """Feature avanzate per un singolo match, calcolate sulle stesse formule
    del training (advanced_features.add_*).

    Per la prestazione e l'Elo si ricostruisce il dataset storico con le
    colonne avanzate e si prende il valore alla riga del match: i rolling e
    l'Elo sono gia' calcolati con shift(1), quindi il valore alla riga del
    match usa solo il passato.
    """
    out = {}

    # ── Prestazione (tiri, tiri in porta, angoli, cartellini) ───────────
    perf = add_performance_features(df, window=5)
    for col in ("home_shots_5", "away_shots_5", "home_shots_t_5", "away_shots_t_5",
                "home_corners_5", "away_corners_5", "home_cards_5", "away_cards_5"):
        if col not in perf.columns:
            continue
        # La riga del match: se il match e' gia' nel dataset (partita gia'
        # giocata), la sua riga ha gia' i rolling corretti; se e' futuro, si
        # calcola il rolling come se il match fosse l'ultimo.
        if match_date is not None:
            mask = (perf["Date"] == match_date) & (perf["HomeTeam"] == home) & (perf["AwayTeam"] == away)
            if mask.any():
                out[col] = float(perf.loc[mask, col].iloc[0])
                continue
        out[col] = _rolling_future(perf, df, home, away, match_date, col)

    # ── Elo ──────────────────────────────────────────────────────────────
    elo = add_elo_features(df)
    if match_date is not None:
        mask = (elo["Date"] == match_date) & (elo["HomeTeam"] == home) & (elo["AwayTeam"] == away)
        if mask.any():
            row = elo.loc[mask].iloc[0]
            for col in ("elo_home", "elo_away", "elo_home_expect", "elo_away_expect"):
                if col in elo.columns:
                    out[col] = float(row[col])
            return out
    # Match futuro: Elo calcolato sullo storico fino a match_date.
    past = df[df["Date"] < match_date].copy() if match_date is not None else df.copy()
    if len(past) == 0:
        past = df.copy()
    elo_past = add_elo_features(past)
    if len(elo_past):
        last = elo_past.iloc[-1]
        ratings = {}
        for t in set(elo_past["HomeTeam"]) | set(elo_past["AwayTeam"]):
            sub = elo_past[(elo_past["HomeTeam"] == t) | (elo_past["AwayTeam"] == t)]
            if len(sub):
                ratings[t] = sub.iloc[-1]
        rh = float(ratings.get(home, {"elo_home": 1500.0})["elo_home"]) if home in ratings else 1500.0
        ra = float(ratings.get(away, {"elo_away": 1500.0})["elo_away"]) if away in ratings else 1500.0
        from advanced_features import ELO_HOME_ADV
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
    if match_date is None:
        return np.nan
    past = perf[perf["Date"] < match_date]
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
        raise ValueError("predict_match richiede un league")
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

    # L'elenco viene dal MODELO (stessa fonte di predict_fixtures): se il
    # training e l'inferenza divergessero, il modello riceverebbe colonne
    # diverse da quelle su cui e' stato addestrato.
    feature_order = list(model.feature_names_) if hasattr(model, "feature_names_") else FEATURE_ORDER
    X = pd.DataFrame([{c: feats.get(c, 0.0) for c in feature_order}])

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    probs = model.predict_proba(X)[0]

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
