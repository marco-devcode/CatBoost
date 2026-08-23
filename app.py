# app.py
"""Dashboard Streamlit — Predizione match Serie A (CatBoost + Pi-Rating).

Esegui in locale:  streamlit run app.py
Oppure deploy gratis su Streamlit Community Cloud (collega questo repo GitHub).
"""
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import predict_match, load_history

st.set_page_config(page_title="Predizione Serie A", page_icon="⚽", layout="centered")

st.title("⚽ Predizione Serie A")
st.caption("CatBoost + Pi-Rating (Dixon-Coles) — modello allenato su storico 2010/11 → oggi")

# ── Carica lista squadre dallo storico ──────────────────────────────────
@st.cache_data
def get_teams():
    try:
        df = load_history()
        teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        return teams
    except Exception:
        return []

teams = get_teams()

col1, col2 = st.columns(2)
with col1:
    home = st.selectbox("Squadra di casa", teams, index=0 if teams else None)
with col2:
    default_away = 1 if len(teams) > 1 else 0
    away = st.selectbox("Squadra in trasferta", teams, index=default_away if teams else None)

match_date = st.date_input("Data del match (per contesto storico)", value=None)

if st.button("🔮 Predici risultato", use_container_width=True):
    if not home or not away:
        st.warning("Seleziona entrambe le squadre.")
    elif home == away:
        st.warning("Seleziona due squadre diverse.")
    else:
        with st.spinner("Calcolo probabilità..."):
            try:
                date_str = match_date.strftime("%Y-%m-%d") if match_date else None
                out, feats = predict_match(home, away, date_str)
            except FileNotFoundError as e:
                st.error(f"Modello non ancora allenato. Esegui la GitHub Action o `python src/model.py`.\n\n{e}")
                st.stop()
            except Exception as e:
                st.error(f"Errore nella predizione: {e}")
                st.stop()

        st.subheader(f"{home}  vs  {away}")
        if match_date:
            st.caption(f"Contesto al {match_date}")

        for label, color, emoji in [("H", "#2ecc71", "🏠"), ("D", "#f1c40f", "🤝"), ("A", "#3498db", "✈️")]:
            pct = out[label]
            st.markdown(f"**{emoji} {label} — {pct:.1f}%**")
            st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}%")

        st.divider()
        with st.expander("Feature usate (debug)"):
            st.json({k: round(float(v), 3) for k, v in feats.items()})

st.divider()
st.caption("Le percentuali sono stime statistiche, non certezze. Usa il modello solo a scopo informativo.")
