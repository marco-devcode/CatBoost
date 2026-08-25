"use client";

import { useEffect, useMemo, useState } from "react";
import { MatchPrediction, PredictionsData } from "@/lib/types";

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("it-IT", { weekday: "short", day: "numeric", month: "short" });
}

function MatchCard({ m }: { m: MatchPrediction }) {
  const played = m.ok !== undefined;
  const maxKey = (["H", "D", "A"] as const).reduce((a, b) => (m.prob[a] >= m.prob[b] ? a : b));
  const hasBm = m.bm && m.bm_prob;
  return (
    <div className="match-card">
      <div className="match-teams">
        <span className="team-home">{m.home}</span>
        <span className="vs">
          {played ? (
            <>
              <span className="result-badge">{m.risultato}</span>{" "}
              <span className={m.ok ? "ok" : "ko"}>{m.ok ? "✅" : "❌"}</span>
            </>
          ) : (
            "vs"
          )}
        </span>
        <span className="team-away">{m.away}</span>
      </div>
      <div className="match-date">{formatDate(m.date)}</div>
      <div className="bars">
        {(["H", "D", "A"] as const).map((k) => (
          <div key={k} className="bar-row">
            <span className={`bar-label ${k}`}>{k === "H" ? "1" : k === "D" ? "X" : "2"}</span>
            <div className="bar-track">
              <div
                className={`bar-fill ${k}`}
                style={{ width: `${Math.min(m.prob[k], 100)}%`, opacity: maxKey === k ? 1 : 0.55 }}
              />
            </div>
            <span className="bar-pct">{m.prob[k].toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {hasBm && (
        <div className="bm-row">
          <span className="bm-title">Bookmaker</span>
          <span className="bm-odds">
            {(["H", "D", "A"] as const).map((k) => (
              <span key={k} className="bm-cell">
                <b>{k === "H" ? "1" : k === "D" ? "X" : "2"}</b> {m.bm![k].toFixed(2)}
                <i className="bm-p">({m.bm_prob![k].toFixed(0)}%)</i>
              </span>
            ))}
          </span>
        </div>
      )}

      {!played && m.kelly && (
        <div className="kelly-hint">
          💡 Kelly {m.kelly.stake_pct.toFixed(1)}% su{" "}
          <b>{m.kelly.pick === "H" ? "1" : m.kelly.pick === "D" ? "X" : "2"}</b>{" "}
          @ {m.kelly.odd.toFixed(2)} (modello {(m.kelly.model_p * 100).toFixed(0)}% vs quota)
        </div>
      )}
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [roundOverride, setRoundOverride] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/predictions.json");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setData(await res.json());
      } catch (e) {
        setError(e instanceof Error ? e.message : "Errore caricamento dati");
      }
    })();
  }, []);

  const roundKeys = useMemo(
    () => (data ? Object.keys(data.giornate).map(Number).sort((a, b) => a - b) : []),
    [data]
  );

  const defaultRound =
    data?.meta.prossima_giornata && roundKeys.includes(data.meta.prossima_giornata)
      ? data.meta.prossima_giornata
      : roundKeys[0];
  const current = roundOverride ?? defaultRound;
  const matches = current && data?.giornate[String(current)] ? data.giornate[String(current)] : [];

  const meta = data?.meta;
  const hasBmData = meta?.bookmaker_rps !== undefined && meta?.model_rps !== undefined;
  const kelly = meta?.kelly;

  if (error)
    return (
      <div className="container">
        <p className="error">⚠️ Errore nel caricamento dei dati: {error}</p>
      </div>
    );

  if (!data)
    return (
      <div className="container">
        <p className="loading">Caricamento…</p>
      </div>
    );

  return (
    <div className="container">
      <header className="header">
        <h1>⚽ Serie A — Predizioni</h1>
        <div className="update-line">
          <span className="pulse" />
          Aggiornato al{" "}
          {new Date(data.meta.last_update).toLocaleDateString("it-IT", {
            day: "numeric",
            month: "long",
            year: "numeric",
          })}
          {" · "}
          Stagione {data.meta.stagione.slice(0, 2)}/{data.meta.stagione.slice(2)}
        </div>
      </header>

      {data.stats.total > 0 && (
        <div className="stats-bar">
          <div className="stat-card">
            <div className="stat-value">
              {data.stats.corrette}/{data.stats.total}
            </div>
            <div className="stat-label">
              Predizioni corrette ({((data.stats.corrette / data.stats.total) * 100).toFixed(0)}%)
            </div>
          </div>
        </div>
      )}

      {hasBmData && (
        <div className="compare-bar">
          <div className="compare-title">Confronto Bookmaker (RPS ↓)</div>
          <div className="compare-grid">
            <div className="compare-cell">
              <span className="compare-name">CatBoost</span>
              <span className="compare-val">{meta!.model_rps!.toFixed(4)}</span>
            </div>
            <div className="compare-cell">
              <span className="compare-name">Bet365</span>
              <span className="compare-val">{meta!.bookmaker_rps!.toFixed(4)}</span>
            </div>
            <div className="compare-cell">
              <span className="compare-name">Esito</span>
              <span className={`compare-val ${meta!.beats_bookmaker_rps ? "good" : "bad"}`}>
                {meta!.beats_bookmaker_rps ? "✅ battiamo" : "⚠️ sotto"}
              </span>
            </div>
          </div>
        </div>
      )}

      {kelly && (
        <div className="kelly-bar">
          <div className="kelly-title">
            💰 Kelly Simulation {kelly.fraction * 100}% · bankroll €1000 (test set)
          </div>
          <div className="kelly-grid">
            <div className="kelly-cell">
              <span className="kelly-num">€{kelly.bankroll_finale.toFixed(0)}</span>
              <span className="kelly-lab">Bankroll finale</span>
            </div>
            <div className="kelly-cell">
              <span className={`kelly-num ${kelly.roi_pct >= 0 ? "good" : "bad"}`}>
                {kelly.roi_pct >= 0 ? "+" : ""}
                {kelly.roi_pct.toFixed(1)}%
              </span>
              <span className="kelly-lab">ROI</span>
            </div>
            <div className="kelly-cell">
              <span className="kelly-num">{kelly.scommesse}</span>
              <span className="kelly-lab">Scommesse</span>
            </div>
          </div>
          <div className="kelly-note">
            Simulazione retrospettiva su partite giocate con edge (pₘ &gt; p_bookmaker).
            Non è un consiglio di scommessa.
          </div>
        </div>
      )}

      <nav className="round-nav">
        <button
          className="round-btn"
          disabled={current === roundKeys[0]}
          onClick={() => setRoundOverride(Math.max(roundKeys[0], current - 1))}
          aria-label="Giornata precedente"
        >
          ‹
        </button>
        <select
          className="round-select"
          value={current ?? ""}
          onChange={(e) => setRoundOverride(Number(e.target.value))}
        >
          {roundKeys.map((r) => (
            <option key={r} value={r}>
              Giornata {r}
            </option>
          ))}
        </select>
        <button
          className="round-btn"
          disabled={current === roundKeys[roundKeys.length - 1]}
          onClick={() => setRoundOverride(Math.min(roundKeys[roundKeys.length - 1], current + 1))}
          aria-label="Giornata successiva"
        >
          ›
        </button>
      </nav>

      <main className="matches">
        {matches.map((m, i) => (
          <MatchCard key={`${m.home}-${m.away}-${i}`} m={m} />
        ))}
        {matches.length === 0 && (
          <p className="loading">Nessuna partita per questa giornata.</p>
        )}
      </main>

      <footer className="footer">
        Modello: CatBoost + Pi-Rating Dixon-Coles (decay 2 anni)
        <br />
        Storico di training: Serie A 2010/11 → oggi · Dati: football-data.co.uk
        <br />
        Le percentuali sono stime statistiche, non consigli.
      </footer>
    </div>
  );
}
