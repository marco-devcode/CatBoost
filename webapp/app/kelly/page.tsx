"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { KellyBet, PredictionsData } from "@/lib/types";
import { LEAGUES } from "@/lib/leagues";

function cPctLabel(c: number | null | undefined): string {
  if (c == null) return "n/d";
  return `${(c * 100).toFixed(0).replace(".", ",")}%`;
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("it-IT", {
    day: "numeric",
    month: "short",
    year: "2-digit",
  });
}

function EquityChart({
  bets,
  initialBankroll,
}: {
  bets: KellyBet[];
  initialBankroll: number;
}) {
  const series = useMemo(
    () => [initialBankroll, ...bets.map((b) => b.bankroll_dopo)],
    [bets, initialBankroll]
  );

  const rawMin = Math.min(...series);
  const rawMax = Math.max(...series);
  const step = rawMax - rawMin > 600 ? 200 : 100;
  const yMin = Math.max(0, Math.floor((rawMin - 40) / step) * step);
  const yMax = Math.ceil((rawMax + 40) / step) * step;
  const ySpan = Math.max(1, yMax - yMin);

  const w = 640;
  const h = 190;
  const padL = 14;
  const padR = 64;
  const padT = 16;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const xAt = (i: number) =>
    padL + (series.length > 1 ? (i / (series.length - 1)) * plotW : plotW / 2);
  const yAt = (v: number) => padT + ((yMax - v) / ySpan) * plotH;

  const points = series.map((v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(" ");

  const yTicks: number[] = [];
  for (let v = yMin; v <= yMax + 0.01; v += step) {
    yTicks.push(Math.round(v));
  }

  const xCount = Math.min(5, Math.max(2, series.length));
  const xIndices = Array.from({ length: xCount }, (_, k) =>
    Math.round((k / (xCount - 1)) * (series.length - 1))
  );

  const baselineY = yAt(initialBankroll);
  const lastX = xAt(series.length - 1);
  const lastY = yAt(series[series.length - 1]);

  return (
    <svg
      className="chart-svg"
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label={`Curva del bankroll da €${initialBankroll.toFixed(0)} a €${series[series.length - 1].toFixed(2)} su ${bets.length} puntate`}
    >
      {yTicks.map((tick) => {
        const y = yAt(tick);
        return (
          <g key={tick}>
            <line
              x1={padL}
              y1={y}
              x2={w - padR}
              y2={y}
              stroke="#191b1f"
              strokeWidth="1"
            />
            <text
              x={w - padR + 8}
              y={y + 4}
              fill="#8a8e96"
              fontSize="12"
              textAnchor="start"
            >
              €{tick}
            </text>
          </g>
        );
      })}

      <line
        x1={padL}
        y1={baselineY}
        x2={w - padR}
        y2={baselineY}
        stroke="#525660"
        strokeWidth="1"
        strokeDasharray="4 4"
      />

      <polyline
        fill="none"
        stroke="#4a8ad6"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
        points={points}
      />

      <circle cx={padL} cy={baselineY} r="3" fill="#8a8e96" />
      <circle cx={lastX} cy={lastY} r="4" fill="#4a8ad6" />

      {xIndices.map((idx, pos) => (
        <text
          key={`${idx}-${pos}`}
          x={xAt(idx)}
          y={h - 6}
          fill="#8a8e96"
          fontSize="12"
          textAnchor={pos === 0 ? "start" : pos === xIndices.length - 1 ? "end" : "middle"}
        >
          {idx === 0 ? "Inizio" : `#${idx}`}
        </text>
      ))}
    </svg>
  );
}

function BetCard({ b }: { b: KellyBet }) {
  return (
    <article className={`bet-card ${b.win ? "win" : "lose"}`}>
      <div className="bet-head">
        <div className="bet-meta">
          <span>#{b.n}</span>
          {b.giornata && <span>Giornata {b.giornata}</span>}
        </div>
        <span className={`pill ${b.win ? "pos" : "neg"}`}>
          {b.win ? "Vinta" : "Persa"}
        </span>
      </div>

      <h3 className="bet-match">
        {b.home} vs {b.away}
      </h3>
      <div className="bet-date">{fmtDate(b.date)}</div>

      <div className="bet-pick">
        <span>
          Puntata su <strong>{b.pick_label}</strong>
        </span>
        <span className="bet-pick-odd num">@ {b.odd.toFixed(2)}</span>
      </div>

      <div className="bet-grid">
        <div className="bet-cell">
          <span className="bet-num">{b.model_p.toFixed(1)}%</span>
          <span className="bet-lab">Prob. modello</span>
        </div>
        <div className="bet-cell">
          <span className="bet-num">{b.bm_p > 0 ? `${b.bm_p.toFixed(1)}%` : "n/d"}</span>
          <span className="bet-lab">Prob. mercato</span>
        </div>
        <div className="bet-cell">
          <span className="bet-num">{b.kelly_pct.toFixed(2)}%</span>
          <span className="bet-lab">Frazione Kelly</span>
        </div>
        <div className="bet-cell">
          <span className="bet-num">€{b.stake.toFixed(2)}</span>
          <span className="bet-lab">Importo</span>
        </div>
      </div>

      <div className="bet-foot">
        {b.win ? (
          <>
            <span className="num">Payout €{b.payout.toFixed(2)}</span>
            <span className="bet-delta pos">+€{b.delta.toFixed(2)}</span>
          </>
        ) : (
          <>
            <span className="num">Payout €0.00</span>
            <span className="bet-delta neg">-€{Math.abs(b.delta).toFixed(2)}</span>
          </>
        )}
        <span className="bet-bankroll">
          Bankroll €{b.bankroll_dopo.toFixed(2)}
        </span>
      </div>
    </article>
  );
}

export default function KellyPage() {
  const [leagueKey, setLeagueKey] = useState<string>(LEAGUES[0].key);
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [outcomeFilter, setOutcomeFilter] = useState<"all" | "win" | "lose">("all");
  const [showAllBets, setShowAllBets] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const league = LEAGUES.find((l) => l.key === leagueKey) ?? LEAGUES[0];
    (async () => {
      setData(null);
      setError(null);
      setOutcomeFilter("all");
      setShowAllBets(false);
      try {
        const res = await fetch(league.json);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Errore caricamento dati");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [leagueKey]);

  if (error) {
    return (
      <main>
        <div className="title-row">
          <div className="title-block">
            <h1>Simulazione Kelly</h1>
            <p className="title-sub">Impossibile caricare i dati della simulazione.</p>
          </div>
        </div>
        <div className="state-panel error">
          <p>Errore nel caricamento dei dati: {error}</p>
          <p style={{ marginTop: 12 }}>
            <Link href="/" className="inline-link">
              Torna alle predizioni
            </Link>
          </p>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main>
        <div className="title-row">
          <div className="title-block">
            <h1>Simulazione Kelly</h1>
            <p className="title-sub">Caricamento serie storica delle puntate.</p>
          </div>
        </div>
        <div className="state-panel">
          <div className="skeleton-line" />
          <p>Caricamento dati…</p>
        </div>
      </main>
    );
  }

  const kelly = data.meta.kelly;
  const cPct = (i: number) => cPctLabel(kelly?.c_dynamic?.[i]);
  const bets = kelly?.scommesse_dettaglio ?? [];
  const winCount = kelly ? kelly.vinte ?? bets.filter((b) => b.win).length : 0;
  const loseCount = kelly ? kelly.scommesse - winCount : 0;
  const winRate =
    kelly && kelly.scommesse > 0 ? (winCount / kelly.scommesse) * 100 : 0;

  const filteredBets =
    outcomeFilter === "all"
      ? bets
      : bets.filter((b) => (outcomeFilter === "win" ? b.win : !b.win));
  const visibleBets = showAllBets ? filteredBets : filteredBets.slice(0, 6);
  const equitySeries = [kelly?.bankroll_iniziale ?? 1000, ...bets.map((b) => b.bankroll_dopo)];
  const minBankroll = Math.min(...equitySeries);
  const maxBankroll = Math.max(...equitySeries);

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

  if (!kelly || bets.length === 0) {
    return (
      <main>
        <div className="title-row">
          <div className="title-block">
            <h1>Simulazione Kelly: {data.meta.league_name ?? "Serie A"}</h1>
            <p className="title-sub">
              Nessuna puntata con edge attivo registrata per questa stagione.
            </p>
          </div>
        </div>
        <div className="filters">
          <div className="segmented" role="group" aria-label="Campionato">
            {LEAGUES.map((l) => (
              <button
                key={l.key}
                type="button"
                className="seg-btn"
                aria-pressed={l.key === leagueKey}
                onClick={() => setLeagueKey(l.key)}
              >
                {l.name}
              </button>
            ))}
          </div>
          <Link href="/" className="btn">
            Torna alle predizioni
          </Link>
        </div>
        <div className="state-panel">
          Nessuna puntata registrata. La simulazione compare quando esistono partite
          giocate con probabilita del modello superiore a quella del bookmaker.
        </div>
      </main>
    );
  }

  const updatedStr = new Date(data.meta.last_update).toLocaleDateString("it-IT", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  return (
    <main>
      <div className="title-row">
        <div className="title-block">
          <h1>Simulazione Kelly: {data.meta.league_name ?? "Serie A"}</h1>
          <p className="title-sub">
            Capitale iniziale €{kelly.bankroll_iniziale.toFixed(0)} diventato €
            {kelly.bankroll_finale.toFixed(2)} ({kelly.roi_pct >= 0 ? "+" : ""}
            {kelly.roi_pct.toFixed(1)}%) dopo {kelly.scommesse} puntate reali con edge.
          </p>
        </div>
        <span className="stamp">Aggiornato al {updatedStr}</span>
      </div>

      <div className="filters">
        <div className="segmented" role="group" aria-label="Campionato">
          {LEAGUES.map((l) => (
            <button
              key={l.key}
              type="button"
              className="seg-btn"
              aria-pressed={l.key === leagueKey}
              onClick={() => setLeagueKey(l.key)}
            >
              {l.name}
            </button>
          ))}
        </div>

        <Link href="/" className="btn">
          Torna alle predizioni
        </Link>
      </div>

      <section className="kpis" aria-label="Riepilogo portafoglio Kelly">
        <div className="kpi">
          <div className="kpi-label">Bankroll finale</div>
          <div className="kpi-value">€{kelly.bankroll_finale.toFixed(2)}</div>
          <div className="kpi-delta">
            Base iniziale €{kelly.bankroll_iniziale.toFixed(0)}
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">ROI complessivo</div>
          <div className={`kpi-value ${kelly.roi_pct >= 0 ? "pos" : "neg"}`}>
            {kelly.roi_pct >= 0 ? "+" : ""}
            {kelly.roi_pct.toFixed(1)}%
          </div>
          <div className="kpi-delta">
            Saldo{" "}
            <span className={kelly.profitto >= 0 ? "pos" : "neg"}>
              {kelly.profitto >= 0 ? "+" : "-"}€{Math.abs(kelly.profitto).toFixed(2)}
            </span>
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">Puntate eseguite</div>
          <div className="kpi-value">{kelly.scommesse}</div>
          <div className="kpi-delta">
            {kelly.capped ?? kelly.n_troncate_al_cap ?? 0} limitate dal tetto massimo
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">Puntate vinte</div>
          <div className="kpi-value">
            {winCount}/{kelly.scommesse}
          </div>
          <div className="kpi-delta">Frequenza di successo {winRate.toFixed(1)}%</div>
        </div>
      </section>

      <div className="grid top">
        <section className="panel col-8" aria-labelledby="equity-heading">
          <div className="panel-head">
            <h2 id="equity-heading">Andamento del bankroll</h2>
            <div className="chart-legend">
              <span className="legend-item">
                <span className="legend-swatch accent" />
                Capitale netto (€)
              </span>
              <span className="legend-item">
                <span className="legend-swatch dashed" />
                Pareggio (€{kelly.bankroll_iniziale.toFixed(0)})
              </span>
            </div>
          </div>
          <EquityChart
            bets={bets}
            initialBankroll={kelly.bankroll_iniziale}
          />
          <p className="rps-note">
            Escursione del portafoglio su {bets.length} puntate: minimo €
            {minBankroll.toFixed(2)}, massimo €{maxBankroll.toFixed(2)}, finale €
            {kelly.bankroll_finale.toFixed(2)}.
          </p>
        </section>

        <section className="panel col-4" aria-labelledby="sizing-heading">
          <div className="panel-head">
            <h2 id="sizing-heading">Parametri Kelly dinamico</h2>
            <span className="panel-meta">Sizing per quota</span>
          </div>
          <div
            tabIndex={0}
            role="region"
            aria-labelledby="sizing-heading"
            className="table-wrap"
          >
            <table>
              <thead>
                <tr>
                  <th scope="col">Fascia quota</th>
                  <th scope="col" className="n">
                    Moltiplicatore
                  </th>
                  <th scope="col" className="n">
                    Tetto
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Fino a {kelly.odds_thresholds[0].toFixed(2)}</td>
                  <td className="n">{cPct(0)}</td>
                  <td className="n">{kelly.max_caps_pct[0].toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>
                    {kelly.odds_thresholds[0].toFixed(2)} -{" "}
                    {kelly.odds_thresholds[1].toFixed(2)}
                  </td>
                  <td className="n">{cPct(1)}</td>
                  <td className="n">{kelly.max_caps_pct[0].toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>
                    {kelly.odds_thresholds[1].toFixed(2)} -{" "}
                    {kelly.odds_thresholds[2].toFixed(2)}
                  </td>
                  <td className="n">{cPct(2)}</td>
                  <td className="n">{kelly.max_caps_pct[1].toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>Oltre {kelly.odds_thresholds[2].toFixed(2)}</td>
                  <td className="n">{cPct(3)}</td>
                  <td className="n">{kelly.max_caps_pct[1].toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="rps-note">
            Simulazione retrospettiva su partite giocate con edge positivo (p modello
            maggiore di p bookmaker). Non costituisce consiglio di scommessa.
          </p>
        </section>
      </div>

      <section className="panel" aria-labelledby="explain-heading">
        <div className="panel-head">
          <h2 id="explain-heading">Come funziona questa simulazione</h2>
          <span className="panel-meta">Regola applicata per singola puntata</span>
        </div>

        <div className="explain-grid">
          <div className="explain-col">
            <p>
              <strong>Cosa sia il Kelly criterion.</strong> Kelly (1956) risponde a una
              domanda: se hai una stima di probabilita <i>p</i> e una quota <i>o</i>{" "}
              che paga <i>o</i> volte la puntata in caso di vittoria, che percentuale
              del bankroll conviene giocare? La formula della frazione piena e
            </p>
            <div className="kelly-formula-box">
              <code>f* = (p·o - 1) / (o - 1)</code>
            </div>
            <p>
              Il termine <i>p·o</i> e il ritorno atteso moltiplicato per la quota; -1
              toglie la puntata impegnata. Se <i>p·o &lt;= 1</i> il risultato e zero o
              negativo e la formula non gioca: la quota non contiene abbastanza margine
              per compensare l&apos;errore di stima.
            </p>
            <p>
              <strong>Perche una frazione variabile.</strong> Il Kelly pieno e
              corretto solo se <i>p</i> e esatta. In realta <i>p</i> e prodotta da un
              modello statistico e contiene errore: su quote alte lo stesso scarto di{" "}
              <i>p</i> amplifica la varianza del bankroll. Per questo il moltiplicatore{" "}
              <i>c_dynamic</i> scende al salire della quota: {cPct(0)} fino a{" "}
              {kelly.odds_thresholds[0].toFixed(2)}, {cPct(1)} fino a{" "}
              {kelly.odds_thresholds[1].toFixed(2)}, {cPct(2)} fino a{" "}
              {kelly.odds_thresholds[2].toFixed(2)} e {cPct(3)} oltre, con hard cap del{" "}
              {kelly.max_caps_pct[0]}% (fino a {kelly.cap_odds_threshold.toFixed(2)}) e
              del {kelly.max_caps_pct[1]}% oltre. In questa stagione{" "}
              {(kelly.capped ?? kelly.n_troncate_al_cap ?? 0) > 0
                ? `${kelly.capped ?? kelly.n_troncate_al_cap} puntate su ${kelly.scommesse} sono state troncate dal tetto.`
                : "nessuna puntata ha raggiunto il tetto."}
            </p>
          </div>

          <div className="explain-col">
            <p>
              <strong>Come viene scelta la puntata.</strong> La pipeline simula partite
              gia giocate e seleziona l&apos;esito dove il modello ha probabilita
              maggiore del bookmaker (con probabilita de-viggate). Calcola il Kelly
              pieno, lo moltiplica per il <i>c_dynamic</i> della fascia di quota e lo
              limita al tetto massimo. La puntata in euro applica questa frazione al
              bankroll disponibile in quel preciso turno (compounding).
            </p>
            <p>
              <strong>Esempio reale da questa pagina.</strong>{" "}
              {example ? (
                <>
                  Partita #{example.n} ({example.home} - {example.away}), esito{" "}
                  {example.pick_label} a quota {example.odd.toFixed(2)}: il modello
                  stimava {example.model_p.toFixed(1).replace(".", ",")}% contro il{" "}
                  {example.bm_p.toFixed(1).replace(".", ",")}% del bookmaker (edge{" "}
                  {(example.model_p - example.bm_p).toFixed(1).replace(".", ",")} punti).
                  La frazione piena e {fullKellyPct.toFixed(2).replace(".", ",")}%,
                  moltiplicata per {cPctLabel(example.c_dynamic)}
                  {example.capped
                    ? ` supera il tetto del ${example.max_cap_pct?.toFixed(2).replace(".", ",")}% e viene troncata a `
                    : " produce "}
                  <strong>{example.kelly_pct.toFixed(2).replace(".", ",")}%</strong>,
                  pari a €{example.stake.toFixed(2).replace(".", ",")} su €
                  {betBankrollBefore.toFixed(2).replace(".", ",")}. La giocata{" "}
                  {example.win
                    ? `ha vinto e ha reso €${example.payout.toFixed(2).replace(".", ",")}, portando il bankroll a €${example.bankroll_dopo.toFixed(2).replace(".", ",")}.`
                    : `e stata persa e il bankroll e passato a €${example.bankroll_dopo.toFixed(2).replace(".", ",")}.`}
                </>
              ) : (
                "Non ci sono puntate con edge sufficiente per mostrare un esempio."
              )}
            </p>
            <p>
              <strong>Cosa non misura.</strong> La simulazione mostra il percorso
              storico su {kelly.scommesse} gare specifiche in quest&apos;ordine
              cronologico: il compounding amplifica sia serie positive sia drawdown, e
              le quote contengono gia il margine del banco. Per valutare la
              calibrazione in astratto conviene guardare l&apos;RPS, che non dipende
              dalla sequenza.
            </p>
          </div>
        </div>
      </section>

      <section className="panel" aria-labelledby="bets-heading">
        <div className="panel-head">
          <h2 id="bets-heading">Registro delle puntate ({filteredBets.length})</h2>
          <div className="filter-group">
            <div className="segmented" role="group" aria-label="Filtra esito puntate">
              <button
                type="button"
                className="seg-btn"
                aria-pressed={outcomeFilter === "all"}
                onClick={() => {
                  setOutcomeFilter("all");
                  setShowAllBets(false);
                }}
              >
                Tutte ({bets.length})
              </button>
              <button
                type="button"
                className="seg-btn"
                aria-pressed={outcomeFilter === "win"}
                onClick={() => {
                  setOutcomeFilter("win");
                  setShowAllBets(false);
                }}
              >
                Vinte ({winCount})
              </button>
              <button
                type="button"
                className="seg-btn"
                aria-pressed={outcomeFilter === "lose"}
                onClick={() => {
                  setOutcomeFilter("lose");
                  setShowAllBets(false);
                }}
              >
                Perse ({loseCount})
              </button>
            </div>
            {filteredBets.length > 6 && (
              <button
                type="button"
                className="btn"
                onClick={() => setShowAllBets((prev) => !prev)}
              >
                {showAllBets
                  ? "Mostra prime 6"
                  : `Mostra tutte (${filteredBets.length})`}
              </button>
            )}
          </div>
        </div>

        <div className="cards-grid">
          {visibleBets.map((b) => (
            <BetCard key={b.n} b={b} />
          ))}
        </div>
      </section>
    </main>
  );
}
