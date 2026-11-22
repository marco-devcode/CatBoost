"use client";

import Link from "next/link";

// Dati reali del modello allenato (verificati contro models/final_report_seriea.json e src/)
const DATASET = {
  partite: 6190,
  stagioni: 17,
  squadre: 41,
  dal: "28 agosto 2010",
  al: "8 novembre 2026",
  esiti: { H: 2682, D: 1598, A: 1910 },
};

const PI_RATING = {
  decay: 2,
  optimizer: "L-BFGS-B",
  maxiter: 500,
  modello: "Poisson con correzione Dixon-Coles (tau)",
};

const CATBOOST = {
  loss: "MultiClass (Residual Logit-Offset)",
  iterazioni: 60,
  learningRate: 0.02,
  depth: 3,
  l2LeafReg: 30,
  seed: 42,
  selezione: "Prior continuo log(p_b365) + alberi regolarizzati sui soli 29 fondamentali",
};

const METRICHE = {
  accuracy: 0.5635,
  accuracyBookmaker: 0.5594,
  f1Macro: 0.4392,
  f1MacroBookmaker: 0.4265,
  rpsModello: 0.1295,
  rpsBookmaker: 0.1296,
  rpsNeutro: 0.2,
  testMatches: 488,
};

const REST_CAP = 7;

const FEATURES = [
  {
    gruppo: "Pi-Rating (Dixon-Coles)",
    count: 4,
    spiegazione:
      "Quattro numeri per squadra, stimati con massima verosimiglianza su un modello Poisson. Non sono posizioni in classifica: sono forze stimate dai gol fatti e subiti, con peso dimezzato ogni due anni.",
    campi: [
      {
        nome: "pi_home_H",
        etichetta: "Forza d'attacco (squadra di casa)",
        dettaglio:
          "exp(attacco - difesa avversaria + vantaggio casalingo). Componente che stima quanti gol segnera la squadra di casa.",
      },
      {
        nome: "pi_away_H",
        etichetta: "Forza d'attacco (squadra in trasferta)",
        dettaglio:
          "Stesso calcolo ma con il vantaggio casalingo in segno negativo: in trasferta le squadre rendono statisticamente meno.",
      },
      {
        nome: "pi_home_A",
        etichetta: "Solidita difensiva (squadra di casa)",
        dettaglio: "Misura della tenuta difensiva della squadra di casa.",
      },
      {
        nome: "pi_away_A",
        etichetta: "Solidita difensiva (squadra in trasferta)",
        dettaglio: "Solidita difensiva della squadra ospite.",
      },
    ],
  },
  {
    gruppo: "Forma recente (ultime 5)",
    count: 4,
    spiegazione:
      "Forma ibrida sui punti (60% ultime 5 partite nel ruolo casa/trasferta + 40% ultime 5 partite complessive) piu differenza reti nel ruolo, sempre con shift causale.",
    campi: [
      {
        nome: "home_form_5",
        etichetta: "Forma ibrida casa (ult. 5)",
        dettaglio:
          "60% media punti nelle ultime 5 gare in casa + 40% media punti nelle ultime 5 gare assolute.",
      },
      {
        nome: "away_form_5",
        etichetta: "Forma ibrida trasferta (ult. 5)",
        dettaglio:
          "60% media punti nelle ultime 5 trasferte + 40% media punti nelle ultime 5 gare assolute.",
      },
      {
        nome: "home_goal_diff_5",
        etichetta: "Differenza reti in casa (ult. 5)",
        dettaglio:
          "Media di (gol fatti - gol subiti) nelle ultime 5 gare in casa.",
      },
      {
        nome: "away_goal_diff_5",
        etichetta: "Differenza reti in trasferta (ult. 5)",
        dettaglio: "Stesso calcolo sulle ultime 5 trasferte.",
      },
    ],
  },
  {
    gruppo: "Gol rolling (ultime 5)",
    count: 4,
    spiegazione:
      "Media di gol fatti e subiti separata tra casa e trasferta, sempre calcolata esclusivamente su gare concluse in precedenza.",
    campi: [
      {
        nome: "home_goals_scored_5",
        etichetta: "Gol fatti in casa (ult. 5)",
        dettaglio: "Media dei gol segnati nelle ultime 5 gare in casa.",
      },
      {
        nome: "home_goals_conceded_5",
        etichetta: "Gol subiti in casa (ult. 5)",
        dettaglio: "Media dei gol subiti nelle ultime 5 gare in casa.",
      },
      {
        nome: "away_goals_scored_5",
        etichetta: "Gol fatti in trasferta (ult. 5)",
        dettaglio: "Media dei gol segnati nelle ultime 5 trasferte.",
      },
      {
        nome: "away_goals_conceded_5",
        etichetta: "Gol subiti in trasferta (ult. 5)",
        dettaglio: "Media dei gol subiti nelle ultime 5 trasferte.",
      },
    ],
  },
  {
    gruppo: "Riposo",
    count: 2,
    spiegazione: `Giorni trascorsi dall'ultima partita, saturati a ${REST_CAP}. Oltre ${REST_CAP} giorni il recupero completo rende identici valori come 8 o 40 giorni, evitando outlier fino a 3000 giorni al rientro dopo anni di assenza.`,
    campi: [
      {
        nome: "home_rest_days",
        etichetta: `Giorni di riposo casa (max ${REST_CAP})`,
        dettaglio: `Giorni dall'ultima gara della squadra di casa, limitati a ${REST_CAP}.`,
      },
      {
        nome: "away_rest_days",
        etichetta: `Giorni di riposo trasferta (max ${REST_CAP})`,
        dettaglio: "Stesso calcolo per la squadra ospite.",
      },
    ],
  },
  {
    gruppo: "Scontri diretti",
    count: 1,
    spiegazione:
      "Percentuale effettiva di vittorie della squadra ospitante negli ultimi 5 scontri diretti (in casa o in trasferta) tra le due squadre. In assenza di precedenti assume il valore neutro 0.4.",
    campi: [
      {
        nome: "h2h_home_winrate",
        etichetta: "Win rate della squadra di casa negli H2H",
        dettaglio:
          "Quota di vittorie della squadra attualmente in casa negli ultimi 5 incroci diretti, oppure 0.4 se assenti.",
      },
    ],
  },
  {
    gruppo: "Stagione corrente",
    count: 2,
    spiegazione:
      "Indicatore per le squadre neopromosse (assenti nella massima serie nella stagione immediatamente precedente). Il modello apprende dai dati storici quanto scontare il salto di categoria.",
    campi: [
      {
        nome: "home_promoted",
        etichetta: "Neopromossa (casa)",
        dettaglio: "Vale 1 se la squadra di casa e appena salita nella lega.",
      },
      {
        nome: "away_promoted",
        etichetta: "Neopromossa (trasferta)",
        dettaglio: "Stesso flag per la squadra ospite.",
      },
    ],
  },
  {
    gruppo: "Prestazione (ultime 5)",
    count: 8,
    spiegazione:
      "Statistiche di produzione offensiva e pressione (tiri totali, tiri in porta come proxy di xG, calci d'angolo e cartellini) sulle ultime 5 partite casa/trasferta per squadra (HS/HST/HC/HY in casa, AS/AST/AC/AY in trasferta), sempre con shift temporale.",
    campi: [
      {
        nome: "home_shots_5 / away_shots_5",
        etichetta: "Tiri totali casa e trasferta (2 feature)",
        dettaglio: "Media dei tiri totali effettuati nelle ultime 5 partite.",
      },
      {
        nome: "home_shots_t_5 / away_shots_t_5",
        etichetta: "Tiri in porta casa e trasferta (2 feature)",
        dettaglio: "Proxy economico di expected goals per la qualita delle occasioni.",
      },
      {
        nome: "home_corners_5 / away_corners_5",
        etichetta: "Angoli casa e trasferta (2 feature)",
        dettaglio: "Proxy di pressione territoriale e possesso offensivo.",
      },
      {
        nome: "home_cards_5 / away_cards_5",
        etichetta: "Cartellini casa e trasferta (2 feature)",
        dettaglio: "Indicatore disciplinare correlato ai falli commessi.",
      },
    ],
  },
  {
    gruppo: "Elo con decadimento",
    count: 4,
    spiegazione:
      "Rating Elo (K=20 pesato per scarto reti, vantaggio casa 65 punti, base 1500) aggiornato partita dopo partita, con mean-reversion del 25% verso 1500 tra una stagione e l'altra.",
    campi: [
      {
        nome: "elo_home / elo_away",
        etichetta: "Rating Elo casa e trasferta (2 feature)",
        dettaglio: "Stato del punteggio Elo prima del fischio d'inizio.",
      },
      {
        nome: "elo_home_expect / elo_away_expect",
        etichetta: "Probabilita attesa Elo (2 feature)",
        dettaglio: "Aspettativa Elo con fattore campo inclusa come input diretto.",
      },
    ],
  },
];

const STEPS = [
  {
    n: 1,
    titolo: "Scaricare i dati",
    testo:
      "Ogni lunedi e giovedi una GitHub Action scarica le 17 stagioni da football-data.co.uk dalla 2010/11 a oggi. Se la fonte non risponde, la pipeline si ferma senza pubblicare dati parziali.",
  },
  {
    n: 2,
    titolo: "Pulire e validare",
    testo:
      "Vengono tenute solo le gare con risultato valido (H/D/A), convertite le date e verificato l'ordine cronologico crescente per evitare fughe di informazione dal futuro.",
  },
  {
    n: 3,
    titolo: "Pi-Rating Walk-Forward",
    testo:
      "Stima Dixon-Coles (con fattore tau e decay ogni 2 anni) fino al cutoff 2025-08-01, poi ri-stima walk-forward mensile causale su partite gia concluse con warm-start.",
  },
  {
    n: 4,
    titolo: "Costruire le 29 feature",
    testo:
      "Ogni sfida genera 29 feature puramente calcistiche negli alberi (Pi-Rating, forma ibrida 60/40, gol, riposo, H2H, neopromosse, tiri propri casa/trasferta, angoli, cartellini ed Elo pesato sui gol).",
  },
  {
    n: 5,
    titolo: "Prior Sharp + Residual Offset",
    testo:
      "Le quote non entrano negli alberi: in valutazione/inferenza si usa il consenso sharp (65% Pinnacle + 35% media mercato, fallback Bet365/modello puro) come baseline continua su cui CatBoost applica il residuo fondamentale.",
  },
  {
    n: 6,
    titolo: "Filtri EV e Portafoglio Kelly",
    testo:
      "Sul test set successivo al 2025-08-01 (488 gare), il filtro seleziona solo vere value bet contro Bet365 con EV > 0,5% e quota <= 4,50, dimensionate con Kelly dinamico.",
  },
];

export default function ModelPage() {
  const totalFeatures = FEATURES.reduce((sum, g) => sum + g.count, 0);
  const pctH = ((DATASET.esiti.H / DATASET.partite) * 100).toFixed(1);
  const pctD = ((DATASET.esiti.D / DATASET.partite) * 100).toFixed(1);
  const pctA = ((DATASET.esiti.A / DATASET.partite) * 100).toFixed(1);

  return (
    <main>
      <div className="title-row">
        <div className="title-block">
          <h1>Come funziona il modello</h1>
          <p className="title-sub">
            CatBoost MultiClass + Pi-Rating Dixon-Coles su{" "}
            {DATASET.partite.toLocaleString("it-IT")} partite storiche ({DATASET.stagioni}{" "}
            stagioni).
          </p>
        </div>
        <Link href="/" className="btn">
          Torna alle predizioni
        </Link>
      </div>

      <section className="kpis" aria-label="Metriche del modello addestrato">
        <div className="kpi">
          <div className="kpi-label">Partite storiche</div>
          <div className="kpi-value">
            {DATASET.partite.toLocaleString("it-IT")}
          </div>
          <div className="kpi-delta">
            Dal {DATASET.dal} al {DATASET.al}
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">Accuratezza test set</div>
          <div className="kpi-value">
            {(METRICHE.accuracy * 100).toFixed(1).replace(".", ",")}%
          </div>
          <div className="kpi-delta">
            Contro {(METRICHE.accuracyBookmaker * 100).toFixed(1).replace(".", ",")}% di
            Bet365 ({METRICHE.testMatches} gare)
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">RPS CatBoost</div>
          <div className="kpi-value">
            {METRICHE.rpsModello.toFixed(4).replace(".", ",")}
          </div>
          <div className="kpi-delta">
            Contro {METRICHE.rpsBookmaker.toFixed(4).replace(".", ",")} di Bet365
          </div>
        </div>

        <div className="kpi">
          <div className="kpi-label">Variabili negli alberi</div>
          <div className="kpi-value">{totalFeatures}</div>
          <div className="kpi-delta">
            Solo fondamentali calcistici (+ prior mercato)
          </div>
        </div>
      </section>

      <div className="doc-shell">
        <nav className="doc-toc" aria-label="Indice metodologia">
          <div className="doc-toc-title">In questa pagina</div>
          <ul className="doc-toc-list">
            <li>
              <a href="#panoramica">Panoramica</a>
            </li>
            <li>
              <a href="#pipeline">Pipeline (6 passi)</a>
            </li>
            <li>
              <a href="#dati">I dati</a>
            </li>
            <li>
              <a href="#pirating">Pi-Rating</a>
            </li>
            <li>
              <a href="#catboost">CatBoost</a>
            </li>
            <li>
              <a href="#feature">Le {totalFeatures} feature</a>
            </li>
            <li>
              <a href="#metriche">Quanto e buono</a>
            </li>
            <li>
              <a href="#limiti">Limiti noti</a>
            </li>
          </ul>
        </nav>

        <div className="panel doc-prose">
          <section id="panoramica" className="doc-section">
            <h2>Panoramica in breve</h2>
            <p>
              Il sistema usa un&apos;architettura{" "}
              <strong>Residual Logit-Offset</strong> a due livelli: le 29
              variabili calcistiche (Pi-Rating Dixon-Coles, Elo, forma, gol,
              tiri, angoli, cartellini, riposo, H2H e neopromosse) entrano
              negli alberi di <strong>CatBoost</strong>, mentre i log-odds
              de-viggati di apertura (<code>log(p_b365)</code>) fungono da
              prior continuo esterno (<code>baseline</code>).
            </p>
            <p>
              Tenere le quote fuori dagli alberi evita che CatBoost quantizzi il
              bookmaker a gradini generando falsi segnali Kelly: gli alberi
              regolarizzati stimano solo la correzione residua quando i
              fondamentali divergono dalla linea di apertura (e sulle gare senza
              quote interviene un modello puro a 29 feature).
            </p>
            <div className="doc-callout">
              I Pi-Rating vengono stimati solo su dati pre-cutoff (&lt;
              2025-08-01) e tutte le stime di partita usano esclusivamente dati
              precedenti al fischio d&apos;inizio, cosi da evitare leakage
              temporale.
            </div>
          </section>

          <section id="pipeline" className="doc-section">
            <h2>La pipeline in 6 passaggi</h2>
            <div className="pipeline-grid">
              {STEPS.map((s) => (
                <article key={s.n} className="pipeline-step">
                  <div className="pipeline-step-head">
                    <span className="step-index">{s.n}</span>
                    <h3>{s.titolo}</h3>
                  </div>
                  <p>{s.testo}</p>
                </article>
              ))}
            </div>
          </section>

          <section id="dati" className="doc-section">
            <h2>I dati</h2>
            <p>
              Fonte storica principale: <strong>football-data.co.uk</strong>. Il dataset
              di riferimento comprende {DATASET.partite.toLocaleString("it-IT")} partite
              su {DATASET.stagioni} stagioni e {DATASET.squadre} squadre distinte dal{" "}
              {DATASET.dal} al {DATASET.al}. Se la fonte non risponde, la pipeline
              interrompe l&apos;esecuzione.
            </p>
            <h3>Distribuzione storica degli esiti</h3>
            <div
              className="dist-bar"
              role="img"
              aria-label={`Distribuzione esiti: Casa ${pctH}%, Pareggio ${pctD}%, Trasferta ${pctA}%`}
            >
              <div className="dist-h" style={{ width: `${pctH}%` }} />
              <div className="dist-d" style={{ width: `${pctD}%` }} />
              <div className="dist-a" style={{ width: `${pctA}%` }} />
            </div>
            <div className="dist-legend num">
              <span>
                <strong>Casa (1):</strong> {pctH}% ({DATASET.esiti.H})
              </span>
              <span>
                <strong>Pareggio (X):</strong> {pctD}% ({DATASET.esiti.D})
              </span>
              <span>
                <strong>Trasferta (2):</strong> {pctA}% ({DATASET.esiti.A})
              </span>
            </div>
          </section>

          <section id="pirating" className="doc-section">
            <h2>Il Pi-Rating (Dixon-Coles)</h2>
            <p>
              Ogni squadra riceve due parametri stimati dai gol: <strong>attacco</strong>{" "}
              (capacita realizzativa) e <strong>difesa</strong> (tenuta). Il modello
              assume una distribuzione di Poisson sui gol attesi:
            </p>
            <div className="kelly-formula-box">
              <code>lambda_casa = exp(attacco_casa - difesa_ospite + home_adv)</code>
            </div>
            <p>
              I parametri si ottengono per massima verosimiglianza con ottimizzatore{" "}
              <code>{PI_RATING.optimizer}</code> fino a {PI_RATING.maxiter} iterazioni.
            </p>
            <h3>Correzione Dixon-Coles e decay temporale</h3>
            <p>
              Il Poisson puro sottostima lo 0-0 e sovrastima l&apos;1-1: il fattore di
              Dixon-Coles corregge la dipendenza sui punteggi bassi. Ogni gara pesa{" "}
              <code>0.5^(anni/2)</code> con mezza vita di {PI_RATING.decay} anni,
              mantenendo i rating reattivi al cambio di ciclo delle squadre.
            </p>
          </section>

          <section id="catboost" className="doc-section">
            <h2>Il modello CatBoost</h2>
            <p>
              CatBoost lavora in modalita{" "}
              <strong>Residual Logit-Offset</strong> su alberi simmetrici: le
              quote di apertura non entrano mai come nodi di split, ma come{" "}
              <code>baseline</code> in log-odds nel <code>Pool</code>, cosi
              l&apos;apprendimento si concentra sul residuo informativo dei{" "}
              {totalFeatures} fondamentali calcistici.
            </p>
            <ul className="doc-list">
              <li>
                <strong>Funzione di perdita:</strong> {CATBOOST.loss} (tre classi: H, D,
                A).
              </li>
              <li>
                <strong>Regolarizzazione residua:</strong> depth{" "}
                {CATBOOST.depth}, {CATBOOST.iterazioni} iterazioni, learning rate{" "}
                {CATBOOST.learningRate} e <code>l2_leaf_reg={CATBOOST.l2LeafReg}</code>{" "}
                ({CATBOOST.selezione}).
              </li>
              <li>
                <strong>Fallback senza quote:</strong> modello CatBoost puro
                allenato sulle stesse {totalFeatures} variabili calcistiche
                senza prior di mercato.
              </li>
              <li>
                <strong>Seme fisso:</strong> {CATBOOST.seed}, per riproducibilita
                completa del training.
              </li>
            </ul>
          </section>

          <section id="feature" className="doc-section">
            <h2>Le {totalFeatures} feature, gruppo per gruppo</h2>
            <p>
              Il modello riceve esattamente {totalFeatures} variabili per ogni sfida,
              abbinate per nome tra addestramento e inferenza.
            </p>
            {FEATURES.map((g) => (
              <div key={g.gruppo} className="doc-group">
                <h3>
                  {g.gruppo} ({g.count})
                </h3>
                <p>{g.spiegazione}</p>
                <div
                  tabIndex={0}
                  role="region"
                  aria-label={`Variabili ${g.gruppo}`}
                  className="table-wrap"
                >
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Chiave</th>
                        <th scope="col">Descrizione</th>
                        <th scope="col">Dettaglio</th>
                      </tr>
                    </thead>
                    <tbody>
                      {g.campi.map((c) => (
                        <tr key={c.nome}>
                          <td>
                            <code>{c.nome}</code>
                          </td>
                          <td>{c.etichetta}</td>
                          <td>{c.dettaglio}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </section>

          <section id="metriche" className="doc-section">
            <h2>Quanto e buono, davvero</h2>
            <p>
              Usiamo due metriche complementari sul test set temporale (
              {METRICHE.testMatches} partite successive al 2025-08-01). L&apos;accuracy
              misura la percentuale di esiti indovinati ({(METRICHE.accuracy * 100).toFixed(1)}%
              modello contro {(METRICHE.accuracyBookmaker * 100).toFixed(1)}% Bet365, F1
              macro {METRICHE.f1Macro.toFixed(3)} contro{" "}
              {METRICHE.f1MacroBookmaker.toFixed(3)}), mentre il{" "}
              <strong>Ranked Probability Score (RPS)</strong> valuta l&apos;intera
              distribuzione di probabilita (piu basso e meglio).
            </p>
            <div className="doc-callout warn">
              Sul test set da {METRICHE.testMatches} partite il modello registra
              RPS <strong>{METRICHE.rpsModello.toFixed(4)}</strong> contro{" "}
              <strong>{METRICHE.rpsBookmaker.toFixed(4)}</strong> di Bet365
              (Accuracy{" "}
              <strong>
                {(METRICHE.accuracy * 100).toFixed(2)}%
              </strong>{" "}
              vs{" "}
              <strong>
                {(METRICHE.accuracyBookmaker * 100).toFixed(2)}%
              </strong>
              ). Combinando il prior sharp Pinnacle+Avg con il residuo
              fondamentale e i filtri di qualità (EV &gt; 0,5%, quota &le;
              4,50), il test set genera Kelly ROI +3,2% su 72 giocate
              (Flat ROI +18,7% su apertura senza filtro quota) — resta aperta la{" "}
              <Link href="/kelly" className="inline-link">
                pagina Portafoglio Kelly
              </Link>{" "}
              per verificare i dettagli partita per partita.
            </div>
          </section>

          <section id="limiti" className="doc-section">
            <h2>Limiti noti</h2>
            <ul className="doc-list">
              <li>
                <strong>Campione stagionale limitato:</strong> una singola stagione
                pubblicata contiene poche decine di gare giocate per lega; servono
                centinaia di sfide per separare abilita reale e varianza.
              </li>
              <li>
                <strong>Sostanziale parita con il bookmaker:</strong> l&apos;RPS sul test
                set e vicino a quello di Bet365 de-viggato; qualsiasi scarto positivo o
                negativo su piccoli campioni va interpretato con cautela.
              </li>
              <li>
                <strong>Il modello non anticipa il mercato:</strong> misurando i
                movimenti tra apertura e chiusura, il modello apprende a correggere le
                quote di apertura ma non dispone di informazioni private.
              </li>
              <li>
                <strong>Quote future non disponibili:</strong> sulle giornate lontane nel
                calendario le quote non esistono ancora e risultano non disponibili.
              </li>
              <li>
                <strong>Inizio stagione neutro:</strong> nelle prime giornate forma e H2H
                hanno pochi dati e le previsioni si appoggiano soprattutto al Pi-Rating.
              </li>
              <li>
                <strong>Assenza di notizie extra-campo:</strong> infortuni, squalifiche e
                cambi di allenatore non entrano nel calcolo numerico.
              </li>
            </ul>
          </section>
        </div>
      </div>
    </main>
  );
}
