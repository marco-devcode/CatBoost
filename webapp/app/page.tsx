"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { MatchPrediction, PredictionsData } from "@/lib/types";
import { LEAGUES } from "@/lib/leagues";

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("it-IT", { weekday: "short", day: "numeric", month: "short" });
}

// Etichette leggibili delle 18 feature (ordine FEATURE_ORDER)
const FEATURE_LABELS: Record<string, string> = {
  pi_home_H: "Pi-Rating casa (home)",
  pi_away_H: "Pi-Rating casa (away)",
  pi_home_A: "Pi-Rating trasf. (home)",
  pi_away_A: "Pi-Rating trasf. (away)",
  home_form_5: "Forma casa (ult. 5)",
  away_form_5: "Forma trasf. (ult. 5)",
  home_pts5: "Punti casa (ult. 5)",
  away_pts5: "Punti trasf. (ult. 5)",
  home_goal_diff5: "Diff. reti casa (ult. 5)",
  away_goal_diff5: "Diff. reti trasf. (ult. 5)",
  h2h_home: "H2H casa (storico scontri)",
  h2h_away: "H2H trasf. (storico scontri)",
  home_rest: "Giorni di riposo casa",
  away_rest: "Giorni di riposo trasf.",
  home_promoted: "Neopromossa casa",
  away_promoted: "Neopromossa trasf.",
  home_goals5: "Gol casa (ult. 5)",
  away_goals5: "Gol trasf. (ult. 5)",
};

function fmtVal(v: number | null | undefined): string {
  // Le feature mancanti arrivano come null (JSON non puo' rappresentare NaN).
  if (v === null || v === undefined || Number.isNaN(v)) return "n/d";
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(3);
}

// ── Interpretazione del confronto Bookmaker via RPS ──────────────────
//
// RPS = Ranked Probability Score. Si calcola sulle 3 probabilita' 1X2 messe
// in ordine cumulato (casa, pareggio, trasferta) e misura quanto sbaglia chi
// mette i propri pesi sui risultati probabili. Piu' il punteggio e' basso, meglio
// e'. Riferimento: ~0.20 e' il valore "neutro" della letteratura per il calcio,
// la skill media di un bookmaker professionista sta intorno a 0.19-0.20.
function rpsInterpretation(
  modelRps: number,
  bmRps: number,
  beats: boolean,
  nMatches: number
): string {
  const gap = modelRps - bmRps; // positivo = il modello e' peggio
  const pct = bmRps > 0 ? Math.abs(gap / bmRps) * 100 : 0;

  if (Math.abs(gap) < 0.005) {
    return `Il modello e il bookmaker sono sostanzialmente equivalenti (scarto ${gap >= 0 ? "+" : ""}${gap.toFixed(4)}, ${pct.toFixed(1)}%): il modello non aggiunge informazione utile sul mercato.`;
  }

  if (beats) {
    return `Il modello e migliore del bookmaker di ${pct.toFixed(1)}% su RPS (${gap.toFixed(4)}). In pratica: quando il modello e il mercato dicono cose diverse, il modello ha ragione piu' spesso del bookmaker — quindi su queste partite c'e' un edge scommettibile. Attenzione pero': RPS premia la calibrazione, non il profitto. Un modello puo' battere il bookmaker su RPS e perdere soldi, perche' le quote contengono gia' il margine del banco e il RPS non lo considera. Su ${nMatches} partite il margine di errore resta ampio.`;
  }

  return `Il modello e peggio del bookmaker di ${pct.toFixed(1)}% su RPS (+${gap.toFixed(4)}). In pratica: sulle partite in cui il modello si discosta dal mercato, il modello si sbaglia piu' spesso del bookmaker. Con una sola stagione (${nMatches} partite) il campione e troppo piccolo per distinguere abilita' reale dal rumore — la differenza e ${pct.toFixed(1)}%, non abbastanza per concludere che il modello abbia edge.`;
}

function MatchFeatures({ m }: { m: MatchPrediction }) {
  const [open, setOpen] = useState(false);
  if (!m.features) return null;
  return (
    <div className="feat-wrap">
      <button className="feat-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} Variabili usate ({Object.keys(m.features).length})
      </button>
      {open && (
        <div className="feat-grid">
          {Object.entries(m.features).map(([k, v]) => (
            <div key={k} className="feat-item">
              <span className="feat-name">{FEATURE_LABELS[k] ?? k}</span>
              <span className="feat-val">{fmtVal(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MatchCard({ m }: { m: MatchPrediction }) {
  const played = m.ok !== undefined;
  const maxKey = (["H", "D", "A"] as const).reduce((a, b) => (m.prob[a] >= m.prob[b] ? a : b));
  // Truthiness NON basta: `bm = {"H": null, "D": null, "A": null}` e' truthy
  // (un dict esiste) ma i valori sono null, e `bm_prob[k].toFixed(0)` fa
  // TypeError. E' successo davvero: 63 partite Eredivisie hanno
  // bm/bm_prob pieni di null, perche' lo snapshot riporta il dict vuoto
  // invece di assente. Qui si richiedono numeri veri su tutti e tre gli esiti.
  const hasBm =
    !!m.bm && !!m.bm_prob && (["H", "D", "A"] as const).every((k) => Number.isFinite(m.bm![k]) && Number.isFinite(m.bm_prob![k]));
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
      {!hasBm && (
        <div className="bm-row bm-none">
          <span className="bm-title">Bookmaker</span>
          <span className="bm-na">Non disponibile (calendario futuro)</span>
        </div>
      )}

      {!played && m.kelly && (
        <div className="kelly-hint">
          💡 Kelly {m.kelly.stake_pct.toFixed(1)}% su{" "}
          <b>{m.kelly.pick === "H" ? "1" : m.kelly.pick === "D" ? "X" : "2"}</b>{" "}
          @ {m.kelly.odd.toFixed(2)} (modello {(m.kelly.model_p * 100).toFixed(0)}% vs quota)
        </div>
      )}

      <MatchFeatures m={m} />
    </div>
  );
}

export default function Home() {
  const [leagueKey, setLeagueKey] = useState<string>(LEAGUES[0].key);
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [roundOverride, setRoundOverride] = useState<number | null>(null);
  const [showVars, setShowVars] = useState(false);

  // Cambiando lega si ricarica il SUO JSON e si azzera la giornata scelta:
  // tenere la giornata di un campionato sull'altro mostrerebbe partite
  // sbagliate (stessi numeri, squadre diverse).
  useEffect(() => {
    let cancelled = false;
    const league = LEAGUES.find((l) => l.key === leagueKey) ?? LEAGUES[0];
    (async () => {
      setData(null);
      setError(null);
      setRoundOverride(null);
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
  // `!== undefined` NON basta: in JS `null !== undefined` e' vero, quindi
  // una lega senza quote storiche (bookmaker_rps = null, come l'Eredivisie)
  // passava il guard e poi .toFixed(4) faceva TypeError su null, con la
  // pagina interamente bianca. Number.isFinite accetta solo numeri veri:
  // scarta sia undefined sia null sia NaN.
  const hasBmData = Number.isFinite(meta?.bookmaker_rps) && Number.isFinite(meta?.model_rps);
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
        <h1>⚽ {meta?.league_name ?? "Serie A"} — Predizioni</h1>
        {/* Selettore campionato: i JSON sono separati, quindi qui si sceglie
            quale caricare. Nessun dato dei due campionati si mescola. */}
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
          <p className="compare-note">{rpsInterpretation(meta!.model_rps!, meta!.bookmaker_rps!, meta!.beats_bookmaker_rps!, data.stats.total)}</p>
        </div>
      )}

      {kelly && (
        <div className="kelly-bar">
          <Link href="/kelly" className="kelly-title kelly-title-link">
            💰 Kelly Simulation {kelly.fraction * 100}% · bankroll €1000 (test set)
            <span className="kelly-cta">Vedi le puntate →</span>
          </Link>
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

      <button className="vars-toggle" onClick={() => setShowVars((s) => !s)}>
        {showVars ? "▾" : "▸"} Variabili di prediction (come nasce la stima)
      </button>
      {showVars && (
        <div className="vars-panel">
          <p className="vars-intro">
            Il modello (CatBoost + Pi-Rating Dixon-Coles, decay 2 anni) riceve <b>18 variabili</b> per
            ogni match. Ecco cosa usiamo e cosa rappresentano. Clicca "Variabili usate" su una
            partita per vedere i valori reali di quella sfida.
          </p>
          <div className="vars-grid">
            {Object.entries(FEATURE_LABELS).map(([k, label]) => (
              <div key={k} className="vars-item">
                <code className="vars-key">{k}</code>
                <span className="vars-desc">{label}</span>
              </div>
            ))}
          </div>
          <p className="vars-note">
            Nota di trasparenza: all&apos;inizio della stagione forma/H2H/riposo sono neutri, quindi
            la stima dipende quasi interamente dai <b>Pi-Rating</b> (forza stimata sui risultati, non
            sulla posizione in classifica). Se un Pi-Rating è anomalo, la quota ne risente — ed è
            visibile qui.
          </p>
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
        Storico di training: {meta?.league_name ?? "Serie A"} 2010/11 → oggi · Dati:{" "}
        football-data.co.uk · Calendario: {meta?.fonte ?? "n/d"} · Quote: TheOddsAPI
        <br />
        Le percentuali sono stime statistiche, non consigli.
      </footer>
    </div>
  );
}
