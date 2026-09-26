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
# importata da features.py cosi' training e inferenza non possono divergere.
# Vedi la nota lunga su REST_CAP in features.add_rest_days.
from features import REST_CAP  # noqa: E402


def load_history():
    """Carica il dataset pulito + Pi-Rating."""
    ratings = pd.read_csv(DATA_DIR / "serie_a_ratings.csv", parse_dates=['Date'])
    return ratings


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


def compute_features_for_match(df, home, away, match_date, h2h_cache=None,
                               current_season_teams=None):
    """Calcola le feature di un match usando SOLO dati precedenti a match_date."""
    past = df[df['Date'] < match_date].copy() if match_date else df.copy()

    if len(past) == 0:
        past = df.copy()

    feats = {}

    # Pi-Rating correnti: usa gli ultimi valori noti per le due squadre.
    # ATTENZIONE: l'ultima partita di una squadra puo' essere in casa O in
    # trasferta. I campi pi_*_H / pi_*_A nel CSV sono sempre riferiti alla
    # squadra DI CASA / squadra IN TRASFERTA di QUELLA riga. Quindi prendiamo
    # il rating della squadra stessa in base a dove ha giocato l'ultima volta,
    # altrimenti si leggerebbe il rating dell'avversaria (bug: Roma 1a giornata
    # 26/27 leggeva pi_home_H spurio da una trasferta -> quotata sotto).
    for team, role in [(home, 'home'), (away, 'away')]:
        sub = past[past['HomeTeam'] == team]
        sub2 = past[past['AwayTeam'] == team]
        last = pd.concat([sub, sub2]).sort_values('Date').tail(1)
        if len(last) == 0:
            feats[f'pi_{role}_H'] = 0.0
            feats[f'pi_{role}_A'] = 0.0
        else:
            lr = last.iloc[0]
            was_home = (lr['HomeTeam'] == team)
            if role == 'home':
                feats['pi_home_H'] = lr['pi_home_H'] if was_home else lr['pi_away_H']
                feats['pi_home_A'] = lr['pi_home_A'] if was_home else lr['pi_away_A']
            else:
                feats['pi_away_H'] = lr['pi_home_H'] if was_home else lr['pi_away_H']
                feats['pi_away_A'] = lr['pi_home_A'] if was_home else lr['pi_away_A']

    # Forma recente (ultimi WINDOW match della squadra, casa o trasferta)
    def team_form(team, is_home):
        tcol, scol, acol = ('HomeTeam', 'FTHG', 'FTAG') if is_home else ('AwayTeam', 'FTAG', 'FTHG')
        sub = past[past[tcol] == team].tail(WINDOW)
        if len(sub) == 0:
            return 1.0  # prior neutro
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
            return 0.0
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

    # H2H: win rate casa negli scontri diretti passati
    key = (min(home, away), max(home, away))
    if h2h_cache is not None and key in h2h_cache:
        feats['h2h_home_winrate'] = h2h_cache[key]
    else:
        sub = past[((past['HomeTeam'] == home) & (past['AwayTeam'] == away)) |
                   ((past['HomeTeam'] == away) & (past['AwayTeam'] == home))].tail(WINDOW)
        if len(sub) == 0:
            feats['h2h_home_winrate'] = 0.4
        else:
            hw = sum(1 for _, r in sub.iterrows()
                     if (r['HomeTeam'] == home and r['FTR'] == 'H') or
                        (r['AwayTeam'] == home and r['FTR'] == 'A'))
            feats['h2h_home_winrate'] = hw / len(sub)
        if h2h_cache is not None:
            h2h_cache[key] = feats['h2h_home_winrate']

    # Flag neopromossa: il modello impara lo sconto da solo. Squadre appena
    # promosse (presenti nella stagione corrente ma assenti dall'ultima stagione
    # storica completa, es. Venezia 26/27) -> 1.
    feats['home_promoted'] = is_promoted(df, home, match_date, current_season_teams)
    feats['away_promoted'] = is_promoted(df, away, match_date, current_season_teams)

    return feats


FEATURE_ORDER = [
    'pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A',
    'home_form_5', 'away_form_5',
    'home_goal_diff_5', 'away_goal_diff_5',
    # Riposo: i due livelli assoluti. rest_advantage NON e' inclusa perche' e'
    # la loro differenza (home - away) e quindi ridondante.
    'home_rest_days', 'away_rest_days',
    'h2h_home_winrate',
    'home_goals_scored_5', 'home_goals_conceded_5',
    'away_goals_scored_5', 'away_goals_conceded_5',
    'home_promoted', 'away_promoted',
]


def predict_match(home, away, match_date=None, model_path=None,
                  current_season_teams=None):
    if model_path is None:
        model_path = MODELS_DIR / "catboost_seriea_v1.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modello non trovato in {model_path}. Esegui prima src/model.py "
            f"(o la GitHub Action) per allenarlo."
        )

    df = load_history()
    if match_date:
        match_date = pd.Timestamp(match_date)

    feats = compute_features_for_match(df, home, away, match_date,
                                        current_season_teams=current_season_teams)

    feature_order = FEATURE_ORDER
    X = pd.DataFrame([{c: feats.get(c, 0.0) for c in feature_order}])

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    probs = model.predict_proba(X)[0]

    labels = ['H', 'D', 'A']
    out = {labels[i]: round(float(probs[i]) * 100, 1) for i in range(3)}
    return out, feats


def main():
    ap = argparse.ArgumentParser(description="Predici un match Serie A")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--date", default=None, help="Data match (YYYY-MM-DD) per il contesto temporale")
    args = ap.parse_args()

    try:
        out, feats = predict_match(args.home, args.away, args.date)
    except FileNotFoundError as e:
        print(f"ERRORE: {e}")
        sys.exit(1)

    print(f"\n{'='*50}")
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
