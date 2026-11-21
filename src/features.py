# src/features.py
"""
Fase 5 — Feature Engineering per CatBoost (un campionato alla volta).

Aggiunge feature contestuali al di sopra dei 4 Pi-Rating baseline:
  1. Rolling Form (punti ultimi 5 match) — home/away separati
  2. Rolling Goals (gol fatti/subiti ultimi 5 match)
  3. Goal Difference Rolling
  4. Rest Days (giorni di riposo, saturati a REST_CAP)
  5. Head-to-Head storico
  6. Shots on Target proxy per xG
  7. Flag neopromossa (home_promoted / away_promoted)

Tutte le feature usano shift(1) per evitare data leakage.

FEATURE_COLS (17) e' la feature set CANONICA condivisa da TUTTE le leghe e
deve combaciare 1:1 con predict.FEATURE_ORDER: CatBoost fa match per NOME
colonna, quindi training e inferenza devono ricevere esattamente le stesse
feature, nello stesso ordine. Se divergono, l'inferenza passa feature diverse
da quelle su cui il modello e' stato addestrato -> output rumore, specialmente
sulle squadre con feature mancanti (neopromosse senza Pi-Rating freschi).

Uso:
    python src/features.py --league seriea
    python src/features.py --league eredivisie
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league, REST_CAP  # noqa: E402
from advanced_features import (  # noqa: E402
    add_elo_features,
    add_market_features,
    add_performance_features,
    advanced_feature_cols,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ═══════════════════════════════════════════════════════════════════════════════
# 5.1 Rolling Points Form (ultimi N match, home/away separati)
# ═══════════════════════════════════════════════════════════════════════════════
def add_rolling_form(df, window=5):
    """Calcola i punti medi negli ultimi N match per ogni squadra."""
    df = df.copy()

    # Punti per ogni partita (prospetta home)
    df['points_home'] = np.where(
        df['FTHG'] > df['FTAG'], 3,
        np.where(df['FTHG'] == df['FTAG'], 1, 0)
    )
    df['points_away'] = np.where(
        df['FTAG'] > df['FTHG'], 3,
        np.where(df['FTAG'] == df['FTHG'], 1, 0)
    )

    # Rolling per squadra di casa (performance in casa)
    form_home = (
        df.groupby('HomeTeam')['points_home']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Rolling per squadra ospite (performance in trasferta)
    form_away = (
        df.groupby('AwayTeam')['points_away']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df[f'home_form_{window}'] = form_home
    df[f'away_form_{window}'] = form_away

    df.drop(columns=['points_home', 'points_away'], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2 Rolling Goals Scored / Conceded
# ═══════════════════════════════════════════════════════════════════════════════
def add_rolling_goals(df, window=5):
    """Media gol fatti e subiti negli ultimi N match."""
    df = df.copy()

    df[f'home_goals_scored_{window}'] = (
        df.groupby('HomeTeam')['FTHG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df[f'home_goals_conceded_{window}'] = (
        df.groupby('HomeTeam')['FTAG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df[f'away_goals_scored_{window}'] = (
        df.groupby('AwayTeam')['FTAG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df[f'away_goals_conceded_{window}'] = (
        df.groupby('AwayTeam')['FTHG']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    df[f'home_goal_diff_{window}'] = (
        df[f'home_goals_scored_{window}'] - df[f'home_goals_conceded_{window}']
    )
    df[f'away_goal_diff_{window}'] = (
        df[f'away_goals_scored_{window}'] - df[f'away_goals_conceded_{window}']
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.3 Rest Days (giorni di riposo)
# ═══════════════════════════════════════════════════════════════════════════════
# REST_CAP: oltre questa soglia il recupero muscolare e' completo e i giorni
# in piu' non cambiano la prestazione. Il lettering empirico sul calcio (e la
# pratica comune) mette la soglia attorno ai 7 giorni. Sopra, la feature viene
# appiattita: 8 e 40 giorni di riposo danno lo stesso identico valore.
#
# Serve anche a tagliare gli outlier del dataset: lo storico contiene squadre
# che rientravano dopo anni di assenza (storico parziale, promozioni), che
# producevano rest_days di 400-3000 giorni. Non hanno nessun significato
# fisiologico e il modello non ha dati nella regione 20-3000 per costruire
# split affidabili.
REST_CAP = 7


def add_rest_days(df):
    """Calcola i giorni di riposo per ogni squadra prima di ogni partita.

    I giorni sono saturati a REST_CAP: sotto la soglia il recupero conta,
    sopra no. Vedi nota su REST_CAP.
    """
    df = df.copy()

    home_dates = df[['Date', 'HomeTeam']].rename(columns={'HomeTeam': 'team'})
    away_dates = df[['Date', 'AwayTeam']].rename(columns={'AwayTeam': 'team'})
    all_dates = pd.concat([home_dates, away_dates]).sort_values('Date').reset_index(drop=True)

    all_dates['last_match'] = all_dates.groupby('team')['Date'].shift(1)
    all_dates['rest_days'] = (all_dates['Date'] - all_dates['last_match']).dt.days

    home_rest = (
        all_dates[all_dates['Date'].isin(df['Date'])]
        .groupby(['Date', 'team'])['rest_days']
        .first()
        .reset_index()
        .rename(columns={'team': 'HomeTeam', 'rest_days': 'home_rest_days'})
    )
    away_rest = (
        all_dates[all_dates['Date'].isin(df['Date'])]
        .groupby(['Date', 'team'])['rest_days']
        .first()
        .reset_index()
        .rename(columns={'team': 'AwayTeam', 'rest_days': 'away_rest_days'})
    )

    df = df.merge(home_rest, on=['Date', 'HomeTeam'], how='left')
    df = df.merge(away_rest, on=['Date', 'AwayTeam'], how='left')

    df['home_rest_days'] = df['home_rest_days'].fillna(REST_CAP)
    df['away_rest_days'] = df['away_rest_days'].fillna(REST_CAP)

    # Satura: oltre REST_CAP il riposo e' comunque completo.
    df['home_rest_days'] = df['home_rest_days'].clip(upper=REST_CAP)
    df['away_rest_days'] = df['away_rest_days'].clip(upper=REST_CAP)

    # rest_advantage derivato DOPO la saturazione, cosi' resta coerente e non
    # reintroduce l'informazione che la saturazione ha appena rimosso.
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.4 Head-to-Head Storico
# ═══════════════════════════════════════════════════════════════════════════════
def add_h2h(df, window=5):
    """Win rate della squadra di casa negli ultimi N scontri diretti tra le due squadre."""
    df = df.copy()
    results = []
    h2h_cache = {}

    for row in df.itertuples():
        h, a = row.HomeTeam, row.AwayTeam
        key = (min(h, a), max(h, a))

        past_matches = h2h_cache.get(key, [])
        if not past_matches:
            results.append(0.4)  # prior neutro (leggero vantaggio casa)
        else:
            recent = past_matches[-window:]
            home_wins = sum(1 for winner in recent if winner == h)
            results.append(home_wins / len(recent))

        ftr = row.FTR
        if ftr == 'H':
            winner = h
        elif ftr == 'A':
            winner = a
        else:
            winner = None

        h2h_cache.setdefault(key, []).append(winner)

    df['h2h_home_winrate'] = results
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5.5 Shots on Target proxy (se disponibili nel CSV grezzo)
# ═══════════════════════════════════════════════════════════════════════════════
def add_shots_proxy(df, window=5):
    """Usa tiri in porta come proxy per xG (dati football-data.co.uk)."""
    df = df.copy()

    has_shots = 'HST' in df.columns and 'AST' in df.columns

    if has_shots:
        df[f'home_shots_target_{window}'] = (
            df.groupby('HomeTeam')['HST']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f'away_shots_target_{window}'] = (
            df.groupby('AwayTeam')['AST']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df['shots_diff'] = df[f'home_shots_target_{window}'] - df[f'away_shots_target_{window}']
    else:
        print("  [INFO] Colonne HST/AST non disponibili, skipping shots proxy")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Assemblaggio Feature Matrix
# ═══════════════════════════════════════════════════════════════════════════════
# Feature canoniche (17), ordinate come predict.FEATURE_ORDER per allineare
# training <-> inferenza. Tutte obbligatorie; i debutti (rolling goals NaN)
# sono gestiti nativamente da CatBoost (valore mancante permesso).
#
# rest_advantage NON e' una feature: e' esattamente home_rest_days -
# away_rest_days (verificato: coincide su TUTTE le righe, Serie A ed
# Eredivisie), quindi aggiunge zero informazione ed e' ridondante
# deterministica. Resta calcolata sul dataset solo per ispezione.
FEATURE_COLS = [
    # Pi-rating (baseline)
    'pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A',
    # Forma recente
    'home_form_5', 'away_form_5',
    'home_goal_diff_5', 'away_goal_diff_5',
    # Riposo: i due livelli assoluti, saturati a REST_CAP
    'home_rest_days', 'away_rest_days',
    # Head to head
    'h2h_home_winrate',
    # Rolling goals (ultimi 5, casa/trasferta)
    'home_goals_scored_5', 'home_goals_conceded_5',
    'away_goals_scored_5', 'away_goals_conceded_5',
    # Flag neopromossa (asimmetrici: il modello impara lo sconto da solo)
    'home_promoted', 'away_promoted',
]

# Le feature avanzate (prestazione, Elo, mercato) sono elencate da
# advanced_features.advanced_feature_cols e VENGONO AGGIUNTE a FEATURE_COLS
# quando il dataset le contiene davvero: le colonne di mercato (Pinnacle,
# media mercato) mancano su alcune stagioni, e una feature sempre a zero
# aggiungerebbe solo rumore.
#
# predict.py costruisce FEATURE_ORDER con la STESSA funzione sullo stesso
# dataset: quindi training e inferenza vedono esattamente le stesse feature,
# e il check di allineamento in fondo a questo file fallisce se diverge.
from advanced_features import advanced_feature_cols as _adv_cols  # noqa: E402


def build_feature_list(available_cols=None) -> list:
    """Elenco canonico delle feature: base + avanzate disponibili."""
    cols = list(FEATURE_COLS)
    if available_cols is None:
        return cols
    avail = set(available_cols)
    for c in _adv_cols(avail):
        if c not in cols:
            cols.append(c)
    return cols


def add_promotion_flags(df, current_season_teams=None):
    """Flag neopromozione storica + supporto squadre stagione corrente.

    Logica storica: per ogni stagione s successiva alla prima dello storico,
    una squadra presente in s ma ASSENTE dalla stagione immediatamente precedente
    (seasons[i - 1]) viene marcata come neopromossa=1 per tutta la stagione s.

    `current_season_teams` (opzionale) estende l'elenco squadre solo dell'ultima
    stagione nello storico: NON altera le stagioni storiche gia' giocate.
    """
    df = df.copy()
    seasons = sorted(df['season'].unique())
    season_teams = {
        s: set(df.loc[df['season'] == s, 'HomeTeam'])
        | set(df.loc[df['season'] == s, 'AwayTeam'])
        for s in seasons
    }
    promoted = {}
    for i, s in enumerate(seasons):
        prev = season_teams[seasons[i - 1]] if i > 0 else None
        teams_in_s = set(season_teams[s])
        if i == len(seasons) - 1 and current_season_teams is not None:
            teams_in_s |= set(current_season_teams)
        for t in teams_in_s:
            promoted[(s, t)] = 0 if (prev is None or t in prev) else 1
    df['home_promoted'] = [promoted[(r.season, r.HomeTeam)] for r in df.itertuples()]
    df['away_promoted'] = [promoted[(r.season, r.AwayTeam)] for r in df.itertuples()]
    n = int((df['home_promoted'] == 1).sum() + (df['away_promoted'] == 1).sum())
    print(f"  Flag neopromozione: {n} presenze neopromosse su {len(df)} partite")
    return df


def build_feature_matrix(df, cutoff='2025-08-01'):
    """Costruisce la matrice feature completa e fa il split temporale.

    Se il dataset e' parziale (solo stagione corrente, es. 2627 con 50 righe) il
    cutoff temporale lascia ZERO righe di training: CatBoost fallirebbe con
    "Labels variable is empty". In quel caso si ripiega su uno split
    posizionale (80/20 cronologico) che mantiene l'ordine temporale.
    """
    # L'elenco delle feature dipende dalle colonne PRESENTI nel dataset:
    # le feature di mercato (Pinnacle, media) mancano su alcune stagioni.
    feature_list = build_feature_list(df.columns)
    df_clean = df[feature_list + ['Date', 'FTR']].copy()

    # DROP: solo le colonne BASE e l'etichetta.
    #
    # Prima si scartava la riga se QUALSIASI feature era NaN. Con le feature
    # avanzate (Elo, prestazione, mercato) questo tagliava il dataset da
    # 6080 a 2463 righe: le quote Pinnacle mancano su alcune stagioni e i
    # NaN dei debut si propagano. Il training e' passato da 5642 a 2265
    # righe e il test da 438 a 198: un modello addestrato su cosi' pochi dati
    # e' rumore.
    #
    # CatBoost gestisce i NaN nativamente (le split trovano la direzione
    # "manca" come una feature a se'). Quindi: NaN = "non so", che e'
    # esattamente la verita' per una feature di mercato su una partita senza
    # quote Pinnacle. Si scarta SOLO la riga senza risultato.
    base_required = FEATURE_COLS + ['FTR']
    df_clean = df_clean.dropna(subset=base_required)

    df_train = df_clean[df_clean['Date'] < cutoff].copy()
    df_test  = df_clean[df_clean['Date'] >= cutoff].copy()

    if len(df_train) == 0:
        # Dataset parziale: niente storia pre-cutoff. Split posizionale 80/20
        # sempre in ordine cronologico, così nessun lookahead.
        n = len(df_clean)
        cut = max(int(n * 0.8), 1)
        df_train = df_clean.iloc[:cut].copy()
        df_test  = df_clean.iloc[cut:].copy()
        print(f"  [split adattivo] nessuna riga pre-{cutoff}: uso split posizionale "
              f"80/20 -> train={len(df_train)}, test={len(df_test)}")
    else:
        print(f"  [split temporale] cutoff={cutoff} -> train={len(df_train)}, test={len(df_test)}")

    feature_cols = [c for c in df_clean.columns if c not in ('Date', 'FTR')]

    X_train = df_train[feature_cols].reset_index(drop=True)
    y_train = df_train['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)
    X_test  = df_test[feature_cols].reset_index(drop=True)
    y_test  = df_test['FTR'].map({'H': 0, 'D': 1, 'A': 2}).reset_index(drop=True)

    return X_train, X_test, y_train, y_test, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main(league):
    print("=" * 55)
    print(f"FASE 5 — Feature Engineering ({league.name})")
    print("=" * 55)

    df = pd.read_csv(league.ratings_path, parse_dates=['Date'])
    print(f"Dataset: {len(df)} partite")

    print("\n5.1 — Rolling Form...")
    df = add_rolling_form(df, window=5)

    print("5.2 — Rolling Goals...")
    df = add_rolling_goals(df, window=5)

    print("5.3 — Rest Days...")
    df = add_rest_days(df)

    print("5.4 — Head-to-Head (puo' richiedere 1-2 min)...")
    df = add_h2h(df, window=5)

    print("5.4b — Flag neopromozione (stagione corrente)...")
    cur_season = sorted(df['season'].unique())[-1]
    cur_teams = (set(df.loc[df['season'] == cur_season, 'HomeTeam'])
                 | set(df.loc[df['season'] == cur_season, 'AwayTeam']))
    df = add_promotion_flags(df, current_season_teams=cur_teams)

    print("5.5 — Prestazione (tiri, tiri in porta, angoli, cartellini)...")
    # Le colonne di prestazione arrivano gia' nel ratings_path: clean_data.py
    # le conserva (insieme alle quote), quindi sono allineate riga per riga
    # per costruzione. Il vecchio add_shots_proxy le leggeva dal clean_path
    # con un allineamento POSIZIONALE fragile e il risultato finiva in
    # FEATURE_COLS solo se qualcuno lo avesse aggiunto: non lo era.
    df = add_performance_features(df, window=5)

    print("5.6 — Elo con decadimento...")
    df = add_elo_features(df)

    print("5.7 — Probabilita' implicite di mercato (quote di apertura, de-vig)...")
    df = add_market_features(df, devig_method="additive")

    df.to_csv(league.features_path, index=False)
    print(f"\nDataset con feature salvato in {league.features_path}")

    print(f"\n{'-'*50}")
    print("Feature generate:")
    all_feature_cols = [c for c in df.columns if c not in
                        ('Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'season',
                         'HST', 'AST', 'HS', 'AS')]
    for col in all_feature_cols:
        # Solo colonne numeriche: il CSV ha anche colonne testuali (es.
        # "Div", "Unnamed: 70") e su quelle .mean() solleva
        # "TypeError: Cannot perform reduction 'mean' with string dtype".
        # Il dump e' puramente diagnostico: non deve mai far fallire la
        # pipeline (i dati sono gia' salvati nella riga sopra).
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            print(f"  {col:<25} (non numerica)")
            continue
        print(f"  {col:<25} mean={s.mean():.3f}, std={s.std():.3f}")

    X_train, X_test, y_train, y_test, feature_cols = build_feature_matrix(df)

    print(f"\n{'-'*50}")
    print(f"Feature matrix ({league.name}):")
    print(f"  Feature ({len(feature_cols)}): {feature_cols}")
    print(f"  Train: {X_train.shape}")
    print(f"  Test : {X_test.shape}")

    # Split v2 per lega: due campionati non possono condividere i file.
    X_train.to_csv(league.split_path('X_train_v2'), index=False)
    X_test.to_csv(league.split_path('X_test_v2'), index=False)
    y_train.to_csv(league.split_path('y_train_v2'), index=False)
    y_test.to_csv(league.split_path('y_test_v2'), index=False)
    print(f"\nSplit v2 salvato in data/ ({league.data_prefix}_*)")

    # Verifica di allineamento training/inferenza: se l'elenco delle feature
    # usate dal training divergesse da quello che l'inferenza usera', il
    # modello riceverebbe colonne diverse da quelle su cui e' stato
    # addestrato. Fallo fallire qui, in CI, non sul sito.
    #
    # La fonte autorevole e' il MODELO: predict_fixtures e predict_match
    # costruiscono la riga di inferenza con model.feature_names_. Quindi il
    # check confronta la lista del training con quella del modello gia'
    # allenato (se esiste) e, in assenza del modello, con l'elenco che
    # predict.py costruisce sullo stesso dataset.
    model_feature_names = None
    try:
        from catboost import CatBoostClassifier as _CBC
        if league.model_path.exists():
            _m = _CBC()
            _m.load_model(str(league.model_path))
            model_feature_names = list(_m.feature_names_)
    except Exception as exc:
        print(f"  (modello non leggibile: {exc})")

    if model_feature_names is not None:
        from model import simplified_feature_set as _model_feature_set
        expected_model_cols = _model_feature_set(feature_cols)
        if model_feature_names != expected_model_cols and model_feature_names != feature_cols:
            print("\nERRORE: le feature del TRAINING divergono da quelle del MODELO!")
            print(f"  solo in training: {[c for c in expected_model_cols if c not in model_feature_names]}")
            print(f"  solo nel modello: {[c for c in model_feature_names if c not in expected_model_cols]}")
            raise SystemExit(1)
        print(f"Allineamento OK: {len(model_feature_names)} feature nel modello, "
              f"{len(feature_cols)} nel dataset (training == modello == inferenza)")
    else:
        # Nessun modello allenato (fresh runner): si confronta con l'elenco
        # che predict.py costruisce sugli stessi dati.
        try:
            import predict as _predict
            feature_order = _predict._build_feature_order(df.columns)
        except Exception as exc:
            print(f"  ATTENZIONE: impossibile verificare l'allineamento: {exc}")
            feature_order = None
        if feature_order is not None and feature_order != feature_cols:
            print("\nERRORE: FEATURE_COLS e predict.FEATURE_ORDER divergono!")
            print(f"  solo in features: {[c for c in feature_cols if c not in feature_order]}")
            print(f"  solo in predict: {[c for c in feature_order if c not in feature_cols]}")
            raise SystemExit(1)
        if feature_order is not None:
            print(f"Allineamento OK: {len(feature_cols)} feature, training == inferenza")

    print(f"\nFase 5 completata per {league.name} — pronto per il training")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Feature Engineering")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    main(get_league(ap.parse_args().league))
