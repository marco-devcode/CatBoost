# src/leagues.py
"""Config unica dei campionati supportati.

Un solo repo, un solo sito, piu' leghe. Ogni lega ha le sue sorgenti dati e i
suoi artefatti; i file di output sono SEPARATI per lega
(`predictions_<lega>.json`) perche' e' impossibile confondere i dati di due
campionati in UI e in storico.

Dove prima c'era un hardcode di lega (codice football-data, id TheSportsDB,
path `data/<lega>_*.csv`, nome del modello, JSON di output) ora c'e' una voce
in LEAGUES. Gli script leggono da qui e non hanno piu' `if league ==`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
WEBAPP_PUBLIC = ROOT / "webapp" / "public"

# Soglia di saturazione dei giorni di riposo: oltre questa il recupero e'
# completo. Unica sorgente di verita' — features.py la definisce e predict.py
# la importa, cosi' training e inferenza non possono divergere.
REST_CAP = 7


@dataclass(frozen=True)
class League:
    """Un campionato: identita', dati, modello, calendario, alias."""

    key: str                      # identificativo interno, es. "seriea"
    name: str                     # nome leggibile, es. "Serie A"
    country: str = ""

    # ── Dati storici ────────────────────────────────────────────────────────
    fd_code: str = ""             # codice football-data.co.uk (I1, N1, ...)
    seasons: tuple[str, ...] = ()  # formato YY1YY2, storico completo
    data_prefix: str = ""         # es. "serie_a" -> data/serie_a_raw.csv
    # Prefisso dei file di split (X_train_v2 ecc.). Vuoto = nomi storici senza
    # prefisso, usati dalla Serie A per non rinominare artefatti gia' committati.
    split_prefix: str = ""

    # ── Modello ─────────────────────────────────────────────────────────────
    model_name: str = ""          # es. "catboost_seriea_v1"

    # ── Calendario / quote ──────────────────────────────────────────────────
    season_fd: str = ""           # stagione corrente per football-data (2627)
    tsdb_league: str = ""         # id TheSportsDB (solo se la lega c'e' nel free tier)
    tsdb_rounds: int = 0
    teams: int = 0              # squadre per stagione (18 Eredivisie, 20 Serie A)
    matches_per_round: int = 10  # partite per giornata

    # Calendario openfootball: repo public domain, nessuna chiave. E' la fonte
    # delle partite FUTURE (football-data.co.uk pubblica solo le giocate).
    # None = non usare openfootball per questa lega.
    openfootball_key: str = ""    # es. "nl.1"
    openfootball_repo: str = "openfootball/football.json"
    openfootball_branch: str = "master"

    # Quote prepartita: TheOddsAPI. Copre solo l'orizzonte vicino (le quote dei
    # bookmaker esistono per le partite che si avvicinano, non per tutto il
    # calendario). sport_key=None = le quote restano solo sulle partite giocate,
    # cioe' quelle che football-data.co.uk distribuisce nel CSV stagionale.
    odds_sport_key: str = ""      # es. "soccer_netherlands_eredivisie"
    odds_bookmaker: str = ""      # bookmaker preferito (sharp), vuoto = media

    # ── JSON di output ──────────────────────────────────────────────────────
    json_name: str = ""           # es. "predictions_seriea.json"

    # ── Nomi squadra ────────────────────────────────────────────────────────
    # openfootball e TheOddsAPI usano nomi diversi da football-data.co.uk.
    # La chiave e' il nome esterno, il valore quello canonico dello storico.
    # Se una squadra non c'e' in aliases, il fallback e' norm_team().
    aliases: dict = field(default_factory=dict)

    # ── Derivati ────────────────────────────────────────────────────────────
    @property
    def matches_per_season(self) -> int:
        """Partite a stagione = turno di andata + ritorno."""
        return 2 * self.teams * (self.matches_per_round - 1)

    @property
    def raw_path(self) -> Path:
        return DATA_DIR / f"{self.data_prefix}_raw.csv"

    @property
    def clean_path(self) -> Path:
        return DATA_DIR / f"{self.data_prefix}_clean.csv"

    @property
    def ratings_path(self) -> Path:
        return DATA_DIR / f"{self.data_prefix}_ratings.csv"

    @property
    def features_path(self) -> Path:
        return DATA_DIR / f"{self.data_prefix}_features.csv"

    @property
    def ratings_snapshot_path(self) -> Path:
        return DATA_DIR / f"{self.data_prefix}_pi_ratings.json"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"{self.model_name}.cbm"

    @property
    def json_path(self) -> Path:
        return WEBAPP_PUBLIC / self.json_name

    @property
    def fd_url(self) -> str:
        return ("https://www.football-data.co.uk/mmz4281/"
                f"{{season}}/{self.fd_code}.csv")

    @property
    def openfootball_dir(self) -> str:
        """openfootball nomina le stagioni con l'anno pieno: 2627 -> 2026-27."""
        return f"20{self.season_fd[:2]}-{self.season_fd[2:]}"

    @property
    def openfootball_url(self) -> str | None:
        if not self.openfootball_key:
            return None
        return (f"https://raw.githubusercontent.com/{self.openfootball_repo}/"
                f"{self.openfootball_branch}/{self.openfootball_dir}/"
                f"{self.openfootball_key}.json")

    def split_path(self, kind: str) -> Path:
        """X_train_v2 / X_test_v2 / y_* — separati per lega.

        Il prefisso e' esplicito per lega (`split_prefix`). La Serie A resta
        SENZA prefisso perche' quei nomi sono gia' committati e li consuma la
        pipeline: rinominarli produrrebbe un diff enorme su file generati e
        rischierebbe i conflitti di rebase a ogni run. Ogni lega non-default
        invece riceve il proprio prefisso, cosi' due campionati non possono
        condividere uno split per costruzione, non per convenzione.
        """
        return DATA_DIR / f"{self.split_prefix}{kind}.csv"

    def report_path(self, name: str) -> Path:
        return MODELS_DIR / f"{name}_{self.key}.json"


SERIE_A = League(
    key="seriea",
    name="Serie A",
    country="Italy",
    fd_code="I1",
    seasons=('1011', '1112', '1213', '1314', '1415', '1516', '1617', '1718',
             '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526',
             '2627'),
    data_prefix="serie_a",
    model_name="catboost_seriea_v1",
    season_fd="2627",
    tsdb_league="4332",          # Serie A e' nel free tier di TheSportsDB
    tsdb_rounds=38,
    teams=20,
    matches_per_round=10,
    # openfootball: MISURATO necessario, non ipotizzato. Il commento precedente
    # diceva "TheSportsDB copre gia' le future" ma era un'assunzione: in CI il
    # calendario si fermava a 5 giornate (tutte giocate, 0 future) e la pagina
    # non mostrava nulla da giocare. TheSportsDB nel free tier non dà il
    # calendario completo della stagione corrente. openfootball dà 380 partite
    # su 38 giornate da 10, ed e' public domain senza chiave.
    openfootball_key="it.1",
    odds_sport_key="soccer_italy_serie_a",
    odds_bookmaker="pinnacle",
    json_name="predictions_seriea.json",
    # Serie A: alias TheSportsDB -> storico football-data (invariato dal preesistente)
    aliases={
        # openfootball (fonte del CALENDARIO, 380 partite su 38 giornate) usa
        # nomi diversi da football-data (fonte dello STORICO). Senza questi
        # alias la chiave del freeze non combacia: la stessa partita viene
        # trattata come nuova, ricalcolata, e il conteggio "corrette" cambia
        # da run a run. Misurato: Inter->"Internazionale Milano",
        # Parma->"Parma 1913" rompevano 10 partite su 50 (24/50 -> 21/50).
        "internazionale milano": "Inter", "fc internazionale milano": "Inter",
        "internazionale": "Inter", "inter milan": "Inter", "fc internazionale": "Inter",
        "parma 1913": "Parma", "parma calcio 1913": "Parma", "ac parma": "Parma",
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
    },
)

EREDIVISIE = League(
    key="eredivisie",
    name="Eredivisie",
    country="Netherlands",
    fd_code="N1",
    seasons=('1011', '1112', '1213', '1314', '1415', '1516', '1617', '1718',
             '1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526',
             '2627'),
    data_prefix="eredivisie",
    split_prefix="eredivisie_",   # obbligatorio: senza, i due campionati
                                   # condividerebbero X_train_v2.csv
    model_name="catboost_eredivisie_v1",
    season_fd="2627",
    tsdb_league="",              # Eredivisie NON e' nel free tier di TheSportsDB
    tsdb_rounds=0,
    teams=18,
    matches_per_round=9,         # 18 squadre -> 9 partite per giornata
    openfootball_key="nl.1",     # <- fonte delle partite future
    odds_sport_key="soccer_netherlands_eredivisie",
    odds_bookmaker="pinnacle",
    json_name="predictions_eredivisie.json",
    # Nomi openfootball + nomi TheOddsAPI -> storico football-data.
    # Verificato: i 62 risultati gia' giocati di openfootball coincidono al
    # 100% con football-data.co.uk (0 discrepanze su 62).
    aliases={
        # openfootball
        "ado den haag": "Den Haag",
        "afc ajax": "Ajax",
        "az": "AZ Alkmaar",
        "fc groningen": "Groningen",
        "fc twente '65": "Twente", "fc twente": "Twente",
        "fc utrecht": "Utrecht",
        "feyenoord rotterdam": "Feyenoord",
        "fortuna sittard": "For Sittard",
        "go ahead eagles": "Go Ahead Eagles",
        "nec": "Nijmegen",
        "pec zwolle": "Zwolle",
        "psv": "PSV Eindhoven",
        "sbv excelsior": "Excelsior",
        "sc cambuur-leeuwarden": "Cambuur", "sc cambuur": "Cambuur",
        "sc heerenveen": "Heerenveen",
        "sparta rotterdam": "Sparta Rotterdam",
        "telstar 1963": "Telstar", "sc telstar": "Telstar",
        "willem ii tilburg": "Willem II",
        # TheOddsAPI (nomi diversi da openfootball per le stesse squadre)
        "fc twente enschede": "Twente",
        "fc zwolle": "Zwolle",
        "fortuna sittard": "For Sittard",
        "nec nijmegen": "Nijmegen",
        "psv eindhoven": "PSV Eindhoven",
        "sc telstar": "Telstar",
        "az alkmaar": "AZ Alkmaar",
        "willem ii": "Willem II",
    },
)

LEAGUES = {lg.key: lg for lg in (SERIE_A, EREDIVISIE)}
DEFAULT_LEAGUE = SERIE_A.key


def _assert_no_artifact_collisions(leagues) -> None:
    """Nessuna coppia di leghe puo' scrivere sugli stessi file.

    Senza questo controllo due leghe con lo stesso `split_prefix` o lo stesso
    `data_prefix` si sovrascrivono a vicenda in silenzio: la pipeline continua
    a essere verde e i due modelli finiscono allenati sugli stessi dati.
    """
    for field in ("data_prefix", "split_prefix", "model_name", "json_name"):
        seen = {}
        for lg in leagues.values():
            val = getattr(lg, field)
            if not val:
                continue
            if val in seen:
                raise SystemExit(
                    f"CONFIG ERRATA: le leghe '{seen[val]}' e '{lg.key}' "
                    f"condividono {field}={val!r}. Ogni lega deve scrivere su "
                    f"file propri."
                )
            seen[val] = lg.key


_assert_no_artifact_collisions(LEAGUES)


def get_league(key: str | None = None) -> League:
    """Risolvi una lega da chiave. None -> default (Serie A)."""
    k = (key or os.environ.get("LEAGUE") or DEFAULT_LEAGUE).strip().lower()
    if k not in LEAGUES:
        raise SystemExit(
            f"Lega sconosciuta: {k!r}. Disponibili: {', '.join(sorted(LEAGUES))}"
        )
    return LEAGUES[k]
