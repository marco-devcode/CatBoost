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


def compute_features_for_match(df, home, away, match_date, h2h_cache=None,
                               current_season_teams=None):
    """Calcola le feature di un match usando SOLO dati precedenti a match_date."""
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

    feature_order = FEATURE_ORDER
    X = pd.DataFrame([{c: feats.get(c, 0.0) for c in feature_order}])

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    probs = model.predict_proba(X)[0]

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
