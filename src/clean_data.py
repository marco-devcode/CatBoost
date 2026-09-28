# src/clean_data.py
"""
Pulizia e validazione del dataset grezzo di un campionato.
Ordinamento temporale STRETTAMENTE CRESCENTE — NON NEGOZIABILE per Pi-Rating.

Uso:
    python src/clean_data.py --league seriea
    python src/clean_data.py --league eredivisie
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league  # noqa: E402

# Forza UTF-8 in output (necessario su Windows con encoding cp1252)
sys.stdout.reconfigure(encoding="utf-8")

CRITICAL_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "season"]


# ── 0. Normalizza colonna stagione ───────────────────────────────────────────
# football-data.co.uk scrive "Season" (es. "2023-2024") o "season" (es. "2425").
# Unifichiamo in "season" formato YY1YY2.
def _norm_season(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s == "nan":
        return None
    # formato "2023-2024" -> "2324"
    if "-" in s:
        a, b = s.split("-", 1)
        return f"{a[-2:]}{b[-2:]}"
    # già "2425"
    return s


def _norm_team(v):
    """Normalizza i nomi squadra: strip degli spazi finali.

    Nello storico Eredivisie ci sono squadre con spazi finali ("Ajax ",
    "Vitesse ", "Utrecht "): sono lo stesso nome di quelle senza, ma il
    modello le tratterebbe come due squadre diverse con zero Pi-Rating.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    return str(v).strip()


def main(league):
    # ── 1. Lettura ────────────────────────────────────────────────────────────
    df = pd.read_csv(league.raw_path, on_bad_lines="skip")
    print(f"[{league.name}] Righe grezze lette: {len(df)}")

    # Se presente 'Season' (storico), unificala in 'season'
    if "Season" in df.columns and "season" not in df.columns:
        df["season"] = df["Season"].map(_norm_season)
    elif "Season" in df.columns and "season" in df.columns:
        # alcune righe hanno season vuoto ma Season valorizzato
        df["season"] = df.apply(
            lambda r: _norm_season(r["season"]) or _norm_season(r.get("Season")),
            axis=1)
    elif "season" in df.columns:
        df["season"] = df["season"].map(_norm_season)

    # ── 2. Selezione colonne ──────────────────────────────────────────────────
    df = df[CRITICAL_COLS].copy()

    # Nomi squadra normalizzati (vedi _norm_team)
    df["HomeTeam"] = df["HomeTeam"].map(_norm_team)
    df["AwayTeam"] = df["AwayTeam"].map(_norm_team)

    # ── 3. Conversione Date (dayfirst=True, errori -> NaT) ───────────────────
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # ── 4. Drop righe con NaN in colonne critiche ─────────────────────────────
    before_drop = len(df)
    df = df.dropna(subset=CRITICAL_COLS)
    print(f"[{league.name}] Righe rimosse per NaN: {before_drop - len(df)}")

    # ── 5. Normalizza FTR e filtra valori validi ──────────────────────────────
    df["FTR"] = df["FTR"].astype(str).str.strip().str.upper()
    df = df[df["FTR"].isin(["H", "D", "A"])]

    # ── 6. Cast tipi numerici ────────────────────────────────────────────────
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # ── 7. Ordinamento cronologico STRETTAMENTE CRESCENTE ────────────────────
    df = df.sort_values("Date", ascending=True, kind="mergesort").reset_index(drop=True)

    # ── 8. Salvataggio ───────────────────────────────────────────────────────
    df.to_csv(league.clean_path, index=False)

    # ── 9. Statistiche ───────────────────────────────────────────────────────
    print(f"\n{'-'*45}")
    print(f"  {league.name}")
    print(f"  Partite totali   : {len(df)}")
    print(f"  Stagioni         : {df['season'].nunique()}")
    print(f"  Squadre          : {len(set(df['HomeTeam']) | set(df['AwayTeam']))}")
    print(f"  Data minima      : {df['Date'].min().date()}")
    print(f"  Data massima     : {df['Date'].max().date()}")
    print(f"\n  Distribuzione FTR:")
    for esito, cnt in df["FTR"].value_counts().sort_index().items():
        print(f"    {esito}  ->  {cnt:>4}  ({cnt / len(df) * 100:.1f}%)")
    print(f"{'-'*45}")
    print(f"  Salvato in: {league.clean_path}")

    # ── 10. BLOCCO ASSERT DI VALIDAZIONE ─────────────────────────────────────
    assert df["Date"].is_monotonic_increasing, \
        "ERRORE: le date NON sono in ordine strettamente crescente!"

    assert set(df["FTR"].unique()).issubset({"H", "D", "A"}), \
        f"ERRORE: FTR contiene valori non validi: {set(df['FTR'].unique()) - {'H','D','A'}}"

    for col in CRITICAL_COLS:
        n_nan = df[col].isna().sum()
        assert n_nan == 0, f"ERRORE: colonna '{col}' contiene {n_nan} valori NaN!"

    # Sanity sullo storico: sotto poche centinaia di righe non si stimano
    # 17 feature e le previsioni diventano rumore. La soglia e' DERIVATA dal
    # dataset reale, non da un numero fisso: l'Eredivisie gioca 306 partite a
    # stagione contro le 380 della Serie A, quindi un range tarato sulla
    # Serie A darebbe un falso allarme (o peggio, un falso "va bene").
    # Riferimento: partite per stagione = 2 * squadre * (partite_per_gornata - 1)
    # (turno inverso), quindi le squadre si ricavano dai dati effettivi.
    n_teams = len(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    seasons_present = df["season"].nunique()
    per_season = len(df) / max(seasons_present, 1)
    expected = league.matches_per_season
    # La stagione corrente e' ancora in corso: si confronta con quello che
    # sarebbe a stagione piena, non con il totale.
    if per_season < expected * 0.6:
        print(f"\n  ATTENZIONE: {per_season:.0f} partite/stagione medi, "
              f"attese ~{expected} per {league.name}.")
        print(f"  Sotto questa soglia i pesi su 17 feature non sono stimabili "
              f"e le previsioni sono rumore. Stagioni presenti: {seasons_present}")
    else:
        print(f"  Storico: {len(df)} partite su {seasons_present} stagioni "
              f"({per_season:.0f}/stagione, {n_teams} squadre) — atteso "
              f"~{expected} a stagione piena per {league.name}")

    # Segnale forte di storico troncato: le ultime 3 stagioni dovrebbero
    # esserci tutte. Se mancano, quasi sempre e' un guard di download che ha
    # short-circuitato (capita due volte in questo repo).
    want = set(league.seasons[-3:])
    missing = want - set(df["season"].unique())
    if missing:
        print(f"  ATTENZIONE: stagioni recenti mancanti: {sorted(missing)}. "
              f"Controlla il download prima di allenare.")

    print("\nDataset pronto per la fase Pi-Rating")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pulizia dataset grezzo")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    args = ap.parse_args()
    main(get_league(args.league))
