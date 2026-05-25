# src/clean_data.py
"""
Pulizia e validazione del dataset Serie A grezzo.
Ordinamento temporale STRETTAMENTE CRESCENTE — NON NEGOZIABILE per Pi-Rating.
"""
import sys
import pandas as pd
from pathlib import Path

# Forza UTF-8 in output (necessario su Windows con encoding cp1252)
sys.stdout.reconfigure(encoding='utf-8')

# --- Percorsi robusti ---
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH   = ROOT / "data" / "serie_a_raw.csv"
CLEAN_PATH = ROOT / "data" / "serie_a_clean.csv"

CRITICAL_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'season']

# ── 1. Lettura ────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH, on_bad_lines='skip')
print(f"Righe grezze lette: {len(df)}")

# ── 2. Selezione colonne ──────────────────────────────────────────────────────
df = df[CRITICAL_COLS].copy()

# ── 3. Conversione Date (dayfirst=True, errori → NaT) ────────────────────────
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# ── 4. Drop righe con NaN in colonne critiche ─────────────────────────────────
before_drop = len(df)
df = df.dropna(subset=CRITICAL_COLS)
print(f"Righe rimosse per NaN: {before_drop - len(df)}")

# ── 5. Normalizza FTR e filtra valori validi ──────────────────────────────────
df['FTR'] = df['FTR'].astype(str).str.strip().str.upper()
df = df[df['FTR'].isin(['H', 'D', 'A'])]

# ── 6. Cast tipi numerici ─────────────────────────────────────────────────────
df['FTHG'] = df['FTHG'].astype(int)
df['FTAG'] = df['FTAG'].astype(int)

# ── 7. Ordinamento cronologico STRETTAMENTE CRESCENTE ─────────────────────────
df = df.sort_values('Date', ascending=True, kind='mergesort').reset_index(drop=True)

# ── 8. Salvataggio ────────────────────────────────────────────────────────────
df.to_csv(CLEAN_PATH, index=False)

# ── 9. Statistiche ────────────────────────────────────────────────────────────
print(f"\n{'─'*45}")
print(f"  Partite totali   : {len(df)}")
print(f"  Data minima      : {df['Date'].min().date()}")
print(f"  Data massima     : {df['Date'].max().date()}")
print(f"\n  Distribuzione FTR:")
for esito, cnt in df['FTR'].value_counts().sort_index().items():
    pct = cnt / len(df) * 100
    print(f"    {esito}  →  {cnt:>4}  ({pct:.1f}%)")
print(f"{'─'*45}")
print(f"  Salvato in: {CLEAN_PATH}")

# ── 10. BLOCCO ASSERT DI VALIDAZIONE ─────────────────────────────────────────
assert df['Date'].is_monotonic_increasing, \
    "❌ ERRORE: le date NON sono in ordine strettamente crescente!"

assert set(df['FTR'].unique()).issubset({'H', 'D', 'A'}), \
    f"❌ ERRORE: FTR contiene valori non validi: {set(df['FTR'].unique()) - {'H','D','A'}}"

for col in CRITICAL_COLS:
    n_nan = df[col].isna().sum()
    assert n_nan == 0, \
        f"❌ ERRORE: colonna '{col}' contiene {n_nan} valori NaN!"

assert 4800 <= len(df) <= 5500, \
    f"❌ ATTENZIONE: numero partite fuori range atteso ({len(df)} — atteso 4800-5200)"

print("\n✓ Dataset pronto per Fase 3")