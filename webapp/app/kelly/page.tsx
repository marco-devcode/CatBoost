"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { KellyBet, PredictionsData } from "@/lib/types";
import { LEAGUES } from "@/lib/leagues";

/** c_dynamic come percentuale con la virgola italiana, es. "50%". */
function cPctLabel(c: number | null | undefined): string {
  if (c == null) return "—";
  return `${(c * 100).toFixed(0).replace(".", ",")}%`;
}

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
  const [leagueKey, setLeagueKey] = useState<string>(LEAGUES[0].key);
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const league = LEAGUES.find((l) => l.key === leagueKey) ?? LEAGUES[0];
    (async () => {
      setData(null);
      setError(null);
      try {
        const res = await fetch(league.json);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Errore caricamento dati");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [leagueKey]);

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
  // cPct legge dal JSON: la pagina non ripete i numeri della regola a mano.
  const cPct = (i: number) => cPctLabel(kelly?.c_dynamic?.[i]);
  const bets = kelly?.scommesse_dettaglio ?? [];
  const winRate = kelly && kelly.scommesse > 0
    ? ((kelly.vinte ?? bets.filter((b) => b.win).length) / kelly.scommesse) * 100
    : 0;

  // Esempio: la puntata con l'edge piu' ampio, per mostrare il caso peggiore
  // (frazione di Kelly piu' aggressiva) e rendere la spiegazione concreta.
  const example =
    bets.length > 0
      ? bets.reduce((best, b) =>
          b.model_p - b.bm_p > best.model_p - best.bm_p ? b : best
        )
      : null;
  const fullKellyPct = example
    ? (((example.model_p / 100) * example.odd - 1) / (example.odd - 1)) * 100
    : 0;
  const betBankrollBefore = example
    ? example.n === 1
      ? kelly!.bankroll_iniziale
      : bets[example.n - 2].bankroll_dopo
    : 0;

  if (!kelly || bets.length === 0)
    return (
      <div className="container">
        <header className="header">
          <h1>💰 Simulazione Kelly — {data.meta.league_name ?? "Serie A"}</h1>
          <nav className="league-switch" aria-label="Campionato">
            {LEAGUES.map((l) => (
              <button
                key={l.key}
                type="button"
                className={l.key === leagueKey ? "league-btn active" : "league-btn"}
                onClick={() => setLeagueKey(l.key)}
                aria-current={l.key === leagueKey}
              >
                {l.name}
              </button>
            ))}
          </nav>
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
        <h1>
          💰 Simulazione Kelly — puntate effettive ·{" "}
          {data.meta.league_name ?? "Serie A"}
        </h1>
        <nav className="league-switch" aria-label="Campionato">
          {LEAGUES.map((l) => (
            <button
              key={l.key}
              type="button"
              className={l.key === leagueKey ? "league-btn active" : "league-btn"}
              onClick={() => setLeagueKey(l.key)}
              aria-current={l.key === leagueKey}
            >
              {l.name}
            </button>
          ))}
        </nav>
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
          Bankroll iniziale €{kelly.bankroll_iniziale.toFixed(0)} · Kelly
          dinamico {cPct(0)}–{cPct(3)}
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
          aveva probabilità maggiore del bookmaker. Ogni puntata è un{" "}
          <b>{cPct(0)}–{cPct(3)}</b> della frazione di Kelly piena — la quota
          decide quale, perché una quota alta moltiplica la varianza del
          ritorno — con un tetto massimo del {kelly.max_caps_pct[0]}% fino a
          quota {kelly.cap_odds_threshold.toFixed(2)} e del{" "}
          {kelly.max_caps_pct[1]}% oltre. Non è un consiglio di scommessa.
        </div>
      </div>

      <h2 className="kelly-h2">Come funziona questa simulazione</h2>

      <div className="kelly-explain">
        <p>
          <b>Cosa sia il Kelly criterion.</b> Kelly (1956) risponde a una domanda:
          se hai una stima di probabilità <i>p</i> e una quota <i>o</i> che
          paga <i>o</i> volte la puntata in caso di vittoria, che percentuale del
          bankroll conviene giocare? La formula della frazione piena è
        </p>
        <p className="kelly-formula">
          f* = (p·o − 1) / (o − 1)
        </p>
        <p>
          Il termine <i>p·o</i> è il ritorno atteso moltiplicato per la quota; −1
          toglie la puntata che hai impegnato. Se <i>p·o ≤ 1</i> il risultato è
          ≤ 0 e la formula dice di non giocare: la quota non contiene abbastanza
          margine per compensare l&apos;errore della stima.
        </p>

        <p>
          <b>Perché una frazione, e perché variabile.</b> Kelly pieno è
          corretto solo se la probabilità <i>p</i> è esatta. In realtà <i>p</i> è
          una stima di un modello, quindi sbaglia. Il Kelly pieno cresce
          esponenzialmente con l&apos;errore: se <i>p</i> è sovrastimata, la
          frazione ottima calcolata è troppo grande e il bankroll crolla molto
          prima che tu possa accorgertene. Su questa stagione (
          {kelly.vinte ?? 0} vinte su {kelly.scommesse}, ROI{" "}
          {kelly.roi_pct >= 0 ? "+" : ""}
          {kelly.roi_pct.toFixed(1)}%) la differenza non è teorica: è il
          risultato reale.
          <br />
          <br />
          Praticando il <b>Kelly frazionato</b> (in letteratura{" "}
          <i>Fractional Kelly</i>) si sacrifica una parte del rendimento atteso
          in cambio di una volatilità contenuta. La parte non è fissa, perché
          il rischio di una scommessa non è uguale a tutte: il payoff di una
          giocata a quota alta è più grande, quindi lo stesso errore di{" "}
          <i>p</i> fa oscillare il bankroll molto di più. Per questo il
          moltiplicatore <i>c_dynamic</i>{" "}
          <b>scende con la quota</b>: {cPct(0)} fino a quota{" "}
          {kelly.odds_thresholds[0].toFixed(2)}, {cPct(1)} fino a{" "}
          {kelly.odds_thresholds[1].toFixed(2)}, {cPct(2)} fino a{" "}
          {kelly.odds_thresholds[2].toFixed(2)}, {cPct(3)} oltre. Sopra ci mette
          un <b>hard cap</b> del {kelly.max_caps_pct[0]}% (fino a quota{" "}
          {kelly.cap_odds_threshold.toFixed(2)}) e del{" "}
          {kelly.max_caps_pct[1]}% oltre, che taglia le code in cui la stima di{" "}
          <i>p</i> è ottimista. In questa simulazione{" "}
          {(kelly.capped ?? kelly.n_troncate_al_cap ?? 0) > 0 ? (
            <>
              {kelly.capped ?? kelly.n_troncate_al_cap} puntate su{" "}
              {kelly.scommesse} sono state troncate dal tetto.
            </>
          ) : (
            <>nessuna puntata ha raggiunto il tetto.</>
          )}
        </p>

        <p>
          <b>Come viene scelta la puntata.</b> La pipeline simula partite
          <i>già giocate</i> e su ognuna sceglie l&apos;esito su cui il modello ha
          probabilità maggiore di quella del bookmaker (edge = p_modello −
          p_bookmaker, entrambe private dell&apos;overround). Su quella scommessa
          calcola la frazione di Kelly piena, la moltiplica per il{" "}
          <i>c_dynamic</i> di quella quota e la tronca al MAX_CAP. La puntata in
          euro è quella frazione applicata al bankroll <i>di quel momento</i>,
          quindi le scommesse successive sono più grandi o più piccole a
          seconda di come sta andando il bankroll — è il compounding. Ogni
          passaggio è ricalcolato a ogni singola puntata, non una volta per
          simulazione.
        </p>

        <p>
          <b>Esempio reale da questa pagina.</b>{" "}
          {example ? (
            <>
              Partita #{example.n}{" "}
              {example.home}–{example.away}, pick {example.pick_label} a quota{" "}
              {example.odd.toFixed(2)}, il modello dava{" "}
              {example.model_p.toFixed(1).replace(".", ",")}% contro il{" "}
              {example.bm_p.toFixed(1).replace(".", ",")}% del bookmaker (edge{" "}
              {(example.model_p - example.bm_p).toFixed(1).replace(".", ",")} punti).
              Con p = {(example.model_p / 100).toFixed(3).replace(".", ",")} e
              o = {example.odd.toFixed(2).replace(".", ",")} la frazione piena è{" "}
              {fullKellyPct.toFixed(2).replace(".", ",")}%. Su quella quota il
              moltiplicatore è {cPctLabel(example.c_dynamic)}
              {example.capped ? (
                <>
                  , quindi {fullKellyPct.toFixed(2).replace(".", ",")}% ×{" "}
                  {cPctLabel(example.c_dynamic)} supererebbe il tetto del{" "}
                  {example.max_cap_pct?.toFixed(2).replace(".", ",")}% e viene
                  troncato: la puntata effettiva è{" "}
                </>
              ) : (
                <>, quindi {fullKellyPct.toFixed(2).replace(".", ",")}% ×{" "}
                  {cPctLabel(example.c_dynamic)} dà </>
              )}
              <b>{example.kelly_pct.toFixed(2).replace(".", ",")}%</b>, cioè €
              {example.stake.toFixed(2).replace(".", ",")} su un bankroll di €
              {betBankrollBefore.toFixed(2).replace(".", ",")}. La scommessa{" "}
              {example.win ? (
                <>
                  ha vinto e ha reso €{example.payout.toFixed(2).replace(".", ",")},
                  portando il bankroll a €
                  {example.bankroll_dopo.toFixed(2).replace(".", ",")}.
                </>
              ) : (
                <>
                  è persa: il bankroll è sceso a €
                  {example.bankroll_dopo.toFixed(2).replace(".", ",")}, e la
                  puntata si rifà sulle {" "}
                  {example.n === bets.length ? "€1000 iniziali" : `€${betBankrollBefore.toFixed(2).replace(".", ",")}`}{" "}
                  di allora.
                </>
              )}
            </>
          ) : (
            <>Non ci sono puntate con edge sufficiente per mostrare un esempio.</>
          )}
        </p>

        <p>
          <b>Cosa NON misura.</b> La simulazione non dice quanto il modello
          guadagnerebbe in media a lungo periodo: mostra cosa sarebbe successo
          su {kelly.scommesse} partite specifiche, con queste quote, in questo
          ordine. L&apos;ordine conta (il compounding amplifica sia le vincite
          sia le perdite), il campione è piccolo, e le quote contengono già il
          margine del banco. Per valutare la qualità del modello in astratto
          conviene guardare l&apos;RPS, che misura la calibrazione e non dipende
          dalla sequenza.{" "}
          <b>Nessuna di queste cifre è un consiglio di scommessa.</b>
        </p>
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
