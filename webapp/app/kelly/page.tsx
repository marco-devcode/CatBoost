"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { KellyBet, PredictionsData } from "@/lib/types";

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("it-IT", {
    day: "numeric",
    month: "short",
    year: "2-digit",
  });
}

function BetCard({ b }: { b: KellyBet }) {
  return (
    <div className={`bet-card ${b.win ? "win" : "lose"}`}>
      <div className="bet-head">
        <span className="bet-n">#{b.n}</span>
        {b.giornata && <span className="bet-gg">GG{b.giornata}</span>}
        <span className={`bet-res ${b.win ? "win" : "lose"}`}>
          {b.win ? "✅ Vinta" : "❌ Persa"}
        </span>
      </div>

      <div className="bet-match">
        {b.home} <span className="bet-vs">vs</span> {b.away}
      </div>
      <div className="bet-date">{fmtDate(b.date)}</div>

      <div className="bet-pick">
        <span className="bet-pick-label">Puntata</span>
        <b>{b.pick_label}</b>
        <span className="bet-pick-odd">@ {b.odd.toFixed(2)}</span>
      </div>

      <div className="bet-grid">
        <div className="bet-cell">
          <span className="bet-num">{b.model_p.toFixed(1)}%</span>
          <span className="bet-lab">Prob. modello</span>
        </div>
        <div className="bet-cell">
          <span className="bet-num">{b.bm_p > 0 ? `${b.bm_p.toFixed(1)}%` : "—"}</span>
          <span className="bet-lab">Prob. bookmaker</span>
        </div>
        <div className="bet-cell">
          <span className="bet-num">{b.kelly_pct.toFixed(2)}%</span>
          <span className="bet-lab">Frazione Kelly</span>
        </div>
        <div className="bet-cell">
          <span className="bet-num">€{b.stake.toFixed(2)}</span>
          <span className="bet-lab">Puntata €</span>
        </div>
      </div>

      <div className={`bet-foot ${b.win ? "win" : "lose"}`}>
        {b.win ? (
          <>
            <span>Payout €{b.payout.toFixed(2)}</span>
            <span className="bet-delta good">+€{b.delta.toFixed(2)}</span>
          </>
        ) : (
          <>
            <span>Payout €0.00</span>
            <span className="bet-delta bad">−€{Math.abs(b.delta).toFixed(2)}</span>
          </>
        )}
        <span className="bet-bankroll">
          Bankroll €{b.bankroll_dopo.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export default function KellyPage() {
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);

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

  if (error)
    return (
      <div className="container">
        <p className="error">⚠️ Errore nel caricamento dei dati: {error}</p>
        <div className="kelly-back">
          <Link href="/">← Torna alle predizioni</Link>
        </div>
      </div>
    );

  if (!data)
    return (
      <div className="container">
        <p className="loading">Caricamento…</p>
      </div>
    );

  const kelly = data.meta.kelly;
  const bets = kelly?.scommesse_dettaglio ?? [];
  const winRate = kelly && kelly.scommesse > 0
    ? ((kelly.vinte ?? bets.filter((b) => b.win).length) / kelly.scommesse) * 100
    : 0;

  if (!kelly || bets.length === 0)
    return (
      <div className="container">
        <header className="header">
          <h1>💰 Simulazione Kelly</h1>
        </header>
        <p className="loading">
          Nessuna puntata registrata. La simulazione compare quando ci sono
          partite giocate con edge del modello (pₘ &gt; p_bookmaker).
        </p>
        <div className="kelly-back">
          <Link href="/">← Torna alle predizioni</Link>
        </div>
      </div>
    );

  return (
    <div className="container">
      <header className="header">
        <h1>💰 Simulazione Kelly — puntate effettive</h1>
        <div className="update-line">
          <span className="pulse" />
          Aggiornato al{" "}
          {new Date(data.meta.last_update).toLocaleDateString("it-IT", {
            day: "numeric",
            month: "long",
            year: "numeric",
          })}
        </div>
      </header>

      <div className="kelly-bar">
        <div className="kelly-title">
          Bankroll iniziale €{kelly.bankroll_iniziale.toFixed(0)} · frazione Kelly{" "}
          {(kelly.fraction * 100).toFixed(0)}%
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
            <span className="kelly-lab">Puntate</span>
          </div>
          <div className="kelly-cell">
            <span className="kelly-num">
              {kelly.vinte ?? bets.filter((b) => b.win).length}/
              {kelly.scommesse}
            </span>
            <span className="kelly-lab">Vinte ({winRate.toFixed(0)}%)</span>
          </div>
        </div>
        <div className="kelly-note">
          Simulazione retrospettiva su partite già giocate dove il modello
          aveva probabilità maggiore del bookmaker. Ogni puntata è pari al{" "}
          {kelly.fraction * 100}% della frazione di Kelly piena calcolata su
          quella quota. Non è un consiglio di scommessa.
        </div>
      </div>

      <h2 className="kelly-h2">Tutte le puntate, in ordine</h2>

      <div className="bets-list">
        {bets.map((b) => (
          <BetCard key={b.n} b={b} />
        ))}
      </div>

      <div className="kelly-back">
        <Link href="/">← Torna alle predizioni</Link>
      </div>
    </div>
  );
}
