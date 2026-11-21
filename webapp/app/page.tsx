"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { MatchPrediction, PredictionsData } from "@/lib/types";
import { LEAGUES } from "@/lib/leagues";

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("it-IT", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

function formatShortDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("it-IT", {
    day: "numeric",
    month: "short",
  });
}

// Etichette leggibili delle 32 feature (piu' alias storici per retrocompatibilita')
const FEATURE_LABELS: Record<string, string> = {
  pi_home_H: "Pi-Rating attacco casa",
  pi_away_H: "Pi-Rating attacco trasf.",
  pi_home_A: "Pi-Rating difesa casa",
  pi_away_A: "Pi-Rating difesa trasf.",
  home_form_5: "Forma casa (ult. 5)",
  away_form_5: "Forma trasf. (ult. 5)",
  home_goal_diff_5: "Diff. reti casa (ult. 5)",
  away_goal_diff_5: "Diff. reti trasf. (ult. 5)",
  home_rest_days: "Giorni di riposo casa",
  away_rest_days: "Giorni di riposo trasf.",
  h2h_home_winrate: "Vittorie casa H2H",
  home_goals_scored_5: "Gol fatti casa (ult. 5)",
  home_goals_conceded_5: "Gol subiti casa (ult. 5)",
  away_goals_scored_5: "Gol fatti trasf. (ult. 5)",
  away_goals_conceded_5: "Gol subiti trasf. (ult. 5)",
  home_promoted: "Neopromossa casa",
  away_promoted: "Neopromossa trasf.",
  home_shots_5: "Tiri totali casa (ult. 5)",
  away_shots_5: "Tiri totali trasf. (ult. 5)",
  home_shots_t_5: "Tiri in porta casa (ult. 5)",
  away_shots_t_5: "Tiri in porta trasf. (ult. 5)",
  home_corners_5: "Angoli casa (ult. 5)",
  away_corners_5: "Angoli trasf. (ult. 5)",
  home_cards_5: "Cartellini casa (ult. 5)",
  away_cards_5: "Cartellini trasf. (ult. 5)",
  elo_home: "Rating Elo casa",
  elo_away: "Rating Elo trasf.",
  elo_home_expect: "Attesa Elo casa",
  elo_away_expect: "Attesa Elo trasf.",
  mkt_b365_pH: "Prob. Bet365 1 (de-vig)",
  mkt_b365_pD: "Prob. Bet365 X (de-vig)",
  mkt_b365_pA: "Prob. Bet365 2 (de-vig)",
  // Alias storici
  rest_advantage: "Vantaggio riposo",
  home_pts5: "Punti casa (ult. 5)",
  away_pts5: "Punti trasf. (ult. 5)",
  home_goal_diff5: "Diff. reti casa (ult. 5)",
  away_goal_diff5: "Diff. reti trasf. (ult. 5)",
  h2h_home: "H2H casa (storico)",
  h2h_away: "H2H trasf. (storico)",
  home_rest: "Giorni di riposo casa",
  away_rest: "Giorni di riposo trasf.",
  home_goals5: "Gol casa (ult. 5)",
  away_goals5: "Gol trasf. (ult. 5)",
};

const CORE_FEATURE_KEYS = [
  "pi_home_H",
  "pi_away_H",
  "pi_home_A",
  "pi_away_A",
  "home_form_5",
  "away_form_5",
  "home_goal_diff_5",
  "away_goal_diff_5",
  "home_rest_days",
  "away_rest_days",
  "h2h_home_winrate",
  "home_goals_scored_5",
  "home_goals_conceded_5",
  "away_goals_scored_5",
  "away_goals_conceded_5",
  "home_promoted",
  "away_promoted",
  "home_shots_5",
  "away_shots_5",
  "home_shots_t_5",
  "away_shots_t_5",
  "home_corners_5",
  "away_corners_5",
  "home_cards_5",
  "away_cards_5",
  "elo_home",
  "elo_away",
  "elo_home_expect",
  "elo_away_expect",
];

function fmtVal(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/d";
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(3);
}

function pickLabel(pick: "H" | "D" | "A"): string {
  return pick === "H" ? "1" : pick === "D" ? "X" : "2";
}

function outcomeLongLabel(pick: "H" | "D" | "A"): string {
  return pick === "H" ? "Casa (1)" : pick === "D" ? "Pareggio (X)" : "Trasf. (2)";
}

function rpsInterpretation(
  modelRps: number,
  bmRps: number,
  beats: boolean,
  nMatches: number
): string {
  const gap = modelRps - bmRps;
  const pct = bmRps > 0 ? Math.abs(gap / bmRps) * 100 : 0;

  if (Math.abs(gap) < 0.005) {
    return `Il modello e il bookmaker sono sostanzialmente equivalenti (scarto ${gap >= 0 ? "+" : ""}${gap.toFixed(4)}, ${pct.toFixed(1)}%): il modello non aggiunge informazione utile sul mercato.`;
  }

  if (beats) {
    return `Il modello supera il bookmaker del ${pct.toFixed(1)}% su RPS (${gap.toFixed(4)}). Quando le due stime divergono il modello e' piu' calibrato, ma l'RPS non include il margine del banco e su ${nMatches} partite la varianza resta ampia.`;
  }

  return `Il modello e' sotto il bookmaker del ${pct.toFixed(1)}% su RPS (+${gap.toFixed(4)}). Quando si discosta dal mercato commette piu' errori, e su ${nMatches} partite il campione e' troppo ridotto per isolare segnale reale dal rumore.`;
}

function hasValidBookmaker(m: MatchPrediction): boolean {
  // Truthiness non basta: dict con valori null causerebbe errori su toFixed
  return (
    !!m.bm &&
    !!m.bm_prob &&
    (["H", "D", "A"] as const).every(
      (k) => Number.isFinite(m.bm![k]) && Number.isFinite(m.bm_prob![k])
    )
  );
}

function RpsComparisonChart({
  modelRps,
  bmRps,
}: {
  modelRps: number;
  bmRps: number;
}) {
  const maxAxis = 0.25;
  const trackLeft = 104;
  const trackRight = 482;
  const trackWidth = trackRight - trackLeft;
  const xFor = (v: number) => trackLeft + Math.min(Math.max(v, 0) / maxAxis, 1) * trackWidth;

  const rows = [
    { label: "CatBoost", value: modelRps, fill: "#4a8ad6", y: 18 },
    { label: "Bet365", value: bmRps, fill: "#8a8e96", y: 46 },
    { label: "Soglia neutra", value: 0.2, fill: "#525660", y: 74 },
  ];
  const ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25];

  return (
    <svg
      className="chart-svg"
      viewBox="0 0 560 112"
      role="img"
      aria-label={`Confronto RPS: CatBoost ${modelRps.toFixed(4)}, Bet365 ${bmRps.toFixed(4)}, soglia neutra 0.2000`}
    >
      {ticks.map((t) => {
        const x = xFor(t);
        return (
          <g key={t}>
            <line
              x1={x}
              y1={10}
              x2={x}
              y2={92}
              stroke="#191b1f"
              strokeWidth="1"
            />
            <text
              x={x}
              y={106}
              fill="#8a8e96"
              fontSize="11"
              textAnchor="middle"
            >
              {t.toFixed(2)}
            </text>
          </g>
        );
      })}
      {rows.map((r) => {
        const w = Math.max(2, xFor(r.value) - trackLeft);
        return (
          <g key={r.label}>
            <text
              x={trackLeft - 10}
              y={r.y + 11}
              fill="#a4a8b0"
              fontSize="12"
              textAnchor="end"
            >
              {r.label}
            </text>
            <rect
              x={trackLeft}
              y={r.y}
              width={trackWidth}
              height={14}
              rx={3}
              fill="#1b1c1f"
            />
            <rect
              x={trackLeft}
              y={r.y}
              width={w}
              height={14}
              rx={3}
              fill={r.fill}
            />
            <text
              x={trackRight + 10}
              y={r.y + 11}
              fill="#f2f2f3"
              fontSize="12"
              fontWeight="600"
              textAnchor="start"
            >
              {r.value.toFixed(4)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function MatchFeatures({ m }: { m: MatchPrediction }) {
  if (!m.features) return null;
  const entries = Object.entries(m.features);
  return (
    <details className="disclosure">
      <summary>Variabili usate ({entries.length})</summary>
      <div className="disclosure-body">
        <div className="feat-grid">
          {entries.map(([k, v]) => (
            <div key={k} className="feat-item">
              <span className="feat-name">{FEATURE_LABELS[k] ?? k}</span>
              <span className="feat-val">{fmtVal(v)}</span>
            </div>
          ))}
        </div>
      </div>
    </details>
  );
}

function MatchCard({ m }: { m: MatchPrediction }) {
  const played = m.ok !== undefined;
  const maxKey = (["H", "D", "A"] as const).reduce((a, b) =>
    m.prob[a] >= m.prob[b] ? a : b
  );
  const hasBm = hasValidBookmaker(m);
  const hasKellyHint =
    !played &&
    !!m.kelly &&
    Number.isFinite(m.kelly.stake_pct) &&
    Number.isFinite(m.kelly.odd) &&
    Number.isFinite(m.kelly.model_p);

  return (
    <article className="match-card">
      <div className="match-top">
        <div>
          <h3 className="match-teams">
            {m.home}
            <span className="match-vs">vs</span>
            {m.away}
          </h3>
          <div className="match-date">{formatDate(m.date)}</div>
        </div>
        <div className="match-outcome">
          {played ? (
            <>
              <span className="match-score">{m.risultato}</span>
              <span className={`pill ${m.ok ? "pos" : "neg"}`}>
                {m.ok ? "Corretta" : "Errata"}
              </span>
            </>
          ) : (
            <span className="pill">Scelta {pickLabel(m.pred)}</span>
          )}
        </div>
      </div>

      <div className="bars">
        {(["H", "D", "A"] as const).map((k) => (
          <div key={k} className="bar-row">
            <span className={`bar-label ${maxKey === k ? "active" : ""}`}>
              {outcomeLongLabel(k)}
            </span>
            <div className="bar-track">
              <div
                className={`bar-fill ${k}`}
                style={{
                  width: `${Math.min(m.prob[k], 100)}%`,
                  opacity: maxKey === k ? 1 : 0.45,
                }}
              />
            </div>
            <span className="bar-pct">{m.prob[k].toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {hasBm ? (
        <div className="bm-row">
          <span className="bm-title">Quote Bet365</span>
          <span className="bm-odds">
            {(["H", "D", "A"] as const).map((k) => (
              <span key={k} className="bm-cell">
                <strong>{pickLabel(k)}</strong> {m.bm![k].toFixed(2)}
                <span className="bm-p">({m.bm_prob![k].toFixed(0)}%)</span>
              </span>
            ))}
          </span>
        </div>
      ) : (
        <div className="bm-row">
          <span className="bm-title">Quote Bet365</span>
          <span className="bm-na">Non disponibili (calendario futuro)</span>
        </div>
      )}

      {hasKellyHint ? (
        <div className="kelly-hint">
          Kelly <strong>{m.kelly!.stake_pct.toFixed(1)}%</strong> su{" "}
          <strong>{pickLabel(m.kelly!.pick)}</strong> @ {m.kelly!.odd.toFixed(2)} (modello{" "}
          {(m.kelly!.model_p * 100).toFixed(0)}% contro quota)
        </div>
      ) : !played && hasBm ? (
        <div className="kelly-hint muted">
          Nessuna puntata Kelly: nessun esito supera la quota di mercato
        </div>
      ) : !played ? (
        <div className="kelly-hint muted">
          Calcolo Kelly disponibile alla pubblicazione delle quote Bet365
        </div>
      ) : (
        <div className="kelly-hint muted">
          Esito registrato: {m.risultato} ({m.ok ? "pronostico confermato" : "pronostico smentito"})
        </div>
      )}

      <MatchFeatures m={m} />
    </article>
  );
}

export default function Home() {
  const [leagueKey, setLeagueKey] = useState<string>(LEAGUES[0].key);
  const [data, setData] = useState<PredictionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [roundOverride, setRoundOverride] = useState<number | null>(null);

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
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Errore caricamento dati");
        }
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
  const matches =
    current && data?.giornate[String(current)] ? data.giornate[String(current)] : [];

  const meta = data?.meta;
  const hasBmData =
    Number.isFinite(meta?.bookmaker_rps) && Number.isFinite(meta?.model_rps);
  const kelly = meta?.kelly;

  if (error) {
    return (
      <main>
        <div className="title-row">
          <div className="title-block">
            <h1>Predizioni CatBoost</h1>
            <p className="title-sub">Impossibile leggere il file dati della lega selezionata.</p>
          </div>
        </div>
        <div className="state-panel error">
          Errore nel caricamento dei dati: {error}
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main>
        <div className="title-row">
          <div className="title-block">
            <h1>Predizioni CatBoost</h1>
            <p className="title-sub">Caricamento probabilita 1X2 e confronto bookmaker in corso.</p>
          </div>
        </div>
        <div className="state-panel">
          <div className="skeleton-line" />
          <p>Caricamento dati…</p>
        </div>
      </main>
    );
  }

  const accPct =
    data.stats.total > 0 ? ((data.stats.corrette / data.stats.total) * 100).toFixed(1) : "0.0";
  const rpsGap = hasBmData ? meta!.model_rps! - meta!.bookmaker_rps! : 0;
  const updatedStr = new Date(data.meta.last_update).toLocaleDateString("it-IT", {
    day: "numeric",
    month: "long",
    year: "numeric",
  });
  const seasonLabel = `${data.meta.stagione.slice(0, 2)}/${data.meta.stagione.slice(2)}`;

  return (
    <main>
      <div className="title-row">
        <div className="title-block">
          <h1>Predizioni {meta?.league_name ?? "Serie A"}</h1>
          <p className="title-sub">
            Stagione {seasonLabel}:{" "}
            {hasBmData
              ? `RPS modello ${meta!.model_rps!.toFixed(4)} contro ${meta!.bookmaker_rps!.toFixed(4)} di Bet365 su ${data.stats.total} partite valutate`
              : `${data.stats.total} partite valutate`}
            .
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

        <div className="filter-group">
          <button
            type="button"
            className="btn btn-icon"
            disabled={current === roundKeys[0]}
            onClick={() => setRoundOverride(Math.max(roundKeys[0], current - 1))}
            aria-label="Giornata precedente"
          >
            ‹
          </button>
          <label htmlFor="round-select" className="sr-only">
            Seleziona giornata
          </label>
          <select
            id="round-select"
            className="select-control"
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
            type="button"
            className="btn btn-icon"
            disabled={current === roundKeys[roundKeys.length - 1]}
            onClick={() =>
              setRoundOverride(Math.min(roundKeys[roundKeys.length - 1], current + 1))
            }
            aria-label="Giornata successiva"
          >
            ›
          </button>
          {defaultRound && current !== defaultRound && (
            <button
              type="button"
              className="btn"
              onClick={() => setRoundOverride(defaultRound)}
            >
              Prossima (GG {defaultRound})
            </button>
          )}
        </div>
      </div>

      <section className="kpis" aria-label="Indicatori del modello">
        <div className="kpi">
          <div className="kpi-label">Esiti indovinati</div>
          <div className="kpi-value">
            {data.stats.total > 0 ? `${data.stats.corrette}/${data.stats.total}` : "n/d"}
          </div>
          <div className="kpi-delta">
            {data.stats.total > 0 ? `${accPct}% sulle partite giocate` : "Nessuna gara giocata"}
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">RPS CatBoost</div>
          <div className="kpi-value">
            {hasBmData ? meta!.model_rps!.toFixed(4) : "n/d"}
          </div>
          <div className="kpi-delta">Errore probabilistico (piu basso e meglio)</div>
        </div>

        <div className="kpi">
          <div className="kpi-label">RPS Bet365</div>
          <div className="kpi-value">
            {hasBmData ? meta!.bookmaker_rps!.toFixed(4) : "n/d"}
          </div>
          <div className="kpi-delta">
            {hasBmData ? (
              <span className={meta!.beats_bookmaker_rps ? "pos" : "warn"}>
                {meta!.beats_bookmaker_rps
                  ? `Modello avanti di ${Math.abs(rpsGap).toFixed(4)}`
                  : `Mercato avanti di ${Math.abs(rpsGap).toFixed(4)}`}
              </span>
            ) : (
              "Quote storiche assenti"
            )}
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">Simulazione Kelly</div>
          <div
            className={`kpi-value ${
              kelly ? (kelly.roi_pct >= 0 ? "pos" : "neg") : ""
            }`}
          >
            {kelly ? `€${kelly.bankroll_finale.toFixed(0)}` : "n/d"}
          </div>
          <div className="kpi-delta">
            {kelly ? (
              <>
                ROI {kelly.roi_pct >= 0 ? "+" : ""}
                {kelly.roi_pct.toFixed(1)}% su {kelly.scommesse} puntate ·{" "}
                <Link href="/kelly" className="inline-link">
                  Portafoglio
                </Link>
              </>
            ) : (
              "Nessuna simulazione attiva"
            )}
          </div>
        </div>
      </section>

      <div className="grid top">
        <section className="panel col-7" aria-labelledby="rps-heading">
          <div className="panel-head">
            <h2 id="rps-heading">Calibrazione RPS</h2>
            {hasBmData ? (
              <span className={`pill ${meta!.beats_bookmaker_rps ? "pos" : "warn"}`}>
                {meta!.beats_bookmaker_rps ? "Modello avanti" : "Sotto mercato"}
              </span>
            ) : (
              <span className="pill">Quote storiche assenti</span>
            )}
          </div>
          {hasBmData ? (
            <>
              <RpsComparisonChart
                modelRps={meta!.model_rps!}
                bmRps={meta!.bookmaker_rps!}
              />
              <p className="rps-note">
                {rpsInterpretation(
                  meta!.model_rps!,
                  meta!.bookmaker_rps!,
                  meta!.beats_bookmaker_rps!,
                  data.stats.total
                )}
              </p>
            </>
          ) : (
            <p className="vars-intro">
              Nessun benchmark RPS disponibile per il campionato selezionato: le quote
              storiche di confronto non sono presenti in questo dataset.
            </p>
          )}
        </section>

        <section className="panel col-5" aria-labelledby="vars-heading">
          <div className="panel-head">
            <h2 id="vars-heading">Variabili di input</h2>
            <span className="panel-meta">32 feature per match</span>
          </div>
          <p className="vars-intro">
            CatBoost e Pi-Rating Dixon-Coles (decay 2 anni) stimano le probabilita 1X2
            da forza storica, forma recente, tiri, Elo e quote di apertura Bet365.
          </p>
          <p className="vars-note" style={{ marginTop: 0, marginBottom: 10 }}>
            Nota di trasparenza: nelle prime giornate della stagione forma, scontri
            diretti e medie mobili hanno pochi dati, quindi la stima dipende in gran
            parte dai Pi-Rating e dal rating Elo.
          </p>
          <details className="disclosure">
            <summary>Apri dizionario delle 32 variabili</summary>
            <div className="disclosure-body">
              <div className="vars-grid">
                {CORE_FEATURE_KEYS.map((k) => (
                  <div key={k} className="vars-item">
                    <code>{k}</code>
                    <span className="vars-desc">{FEATURE_LABELS[k]}</span>
                  </div>
                ))}
              </div>
            </div>
          </details>
        </section>
      </div>

      <section className="panel" aria-labelledby="synoptic-heading">
        <div className="panel-head">
          <h2 id="synoptic-heading">Quadro sinottico Giornata {current}</h2>
          <span className="panel-meta">{matches.length} partite</span>
        </div>

        {matches.length === 0 ? (
          <p className="panel-meta">Nessuna partita registrata per questa giornata.</p>
        ) : (
          <div
            tabIndex={0}
            role="region"
            aria-labelledby="synoptic-heading"
            className="table-wrap"
          >
            <table className="synoptic-table">
              <thead>
                <tr>
                  <th scope="col">Partita</th>
                  <th scope="col" className="n">
                    1
                  </th>
                  <th scope="col" className="n">
                    X
                  </th>
                  <th scope="col" className="n">
                    2
                  </th>
                  <th scope="col" className="n">
                    Esito
                  </th>
                  <th scope="col" className="n">
                    Quota
                  </th>
                </tr>
              </thead>
              <tbody>
                {matches.map((m, idx) => {
                  const played = m.ok !== undefined;
                  const bm = hasValidBookmaker(m) ? m.bm![m.pred] : null;
                  return (
                    <tr key={`${m.home}-${m.away}-${idx}`}>
                      <td>
                        <span className="cell-match">
                          {m.home} vs {m.away}
                        </span>
                        <span className="cell-sub">{formatShortDate(m.date)}</span>
                      </td>
                      <td className="n">{m.prob.H.toFixed(0)}%</td>
                      <td className="n">{m.prob.D.toFixed(0)}%</td>
                      <td className="n">{m.prob.A.toFixed(0)}%</td>
                      <td className="n">
                        {played ? (
                          <span className={`pill ${m.ok ? "pos" : "neg"}`}>
                            {m.risultato} ({pickLabel(m.pred)})
                          </span>
                        ) : (
                          <span className="pill">{pickLabel(m.pred)}</span>
                        )}
                      </td>
                      <td className="n">{bm !== null ? bm.toFixed(2) : "n/d"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="panel-subhead">
          <h3>Analisi partita per partita</h3>
          <span className="section-sub">
            Giornata {current} · {matches.length} sfide
          </span>
        </div>

        <div className="cards-grid">
          {matches.map((m, i) => (
            <MatchCard key={`${m.home}-${m.away}-${i}`} m={m} />
          ))}
        </div>
      </section>
    </main>
  );
}
