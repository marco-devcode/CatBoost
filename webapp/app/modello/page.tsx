"use client";

import Link from "next/link";

// ─────────────────────────────────────────────────────────────────────────────
// Dati reali del modello. Hardcodati qui perche' descrivono il modello
// allenato, non lo stato di una partita: cambiano solo al prossimo training.
// Verificare contro models/metrics_v1.json e src/ prima di modificarli.
// ─────────────────────────────────────────────────────────────────────────────
const DATASET = {
  partite: 6130,
  stagioni: 17,
  squadre: 41,
  dal: "28 agosto 2010",
  al: "20 settembre 2026",
  esiti: { H: 2653, D: 1588, A: 1889 },
};

const PI_RATING = {
  decay: 2,
  optimizer: "L-BFGS-B",
  maxiter: 500,
  modello: "Poisson con correzione Dixon-Coles",
};

const CATBOOST = {
  loss: "MultiClass",
  iterazioni: 500,
  learningRate: 0.05,
  depth: 6,
  seed: 42,
  foldCount: 5,
  grid: [
    { iterazioni: 1000, learningRate: 0.03, depth: 6 },
    { iterazioni: 1000, learningRate: 0.03, depth: 8 },
    { iterazioni: 1000, learningRate: 0.05, depth: 6 },
    { iterazioni: 1000, learningRate: 0.05, depth: 8 },
    { iterazioni: 1500, learningRate: 0.02, depth: 6 },
  ],
};

// Misure sul test set temporale (cutoff 2025-08-01, 438 partite), con il
// benchmark bookmaker CALIBRATO: allineato per chiave e de-viggato con il
// metodo empiricamente migliore (additive). I valori precedenti erano falsati
// dal benchmark posizionale (vedi BATTERE-BOOKMAKER.md).
const METRICHE = {
  accuracy: 0.5616,
  accuracyBookmaker: 0.5434,
  f1Macro: 0.4892,
  f1MacroBookmaker: 0.4168,
  rpsModello: 0.13047,
  rpsBookmaker: 0.13011,
  rpsNeutro: 0.20,
};

const REST_CAP = 7;
const WINDOW = 5;

const FEATURES = [
  {
    gruppo: "Pi-Rating (Dixon-Coles)",
    spiegazione:
      "Quattro numeri per squadra, stimati con massima verosimiglianza su un modello Poisson. Non sono posizioni in classifica: sono forze stimate dai gol fatti e subiti, con i due anni più vecchi pesati per metà a ogni anno.",
    campi: [
      {
        nome: "pi_home_H",
        etichetta: "Forza d'attacco (squadra di casa)",
        dettaglio:
          "exp(attacco − difesa avversaria + vantaggio casalingo). Componente che stima quanti gol segnerà la squadra di casa.",
      },
      {
        nome: "pi_away_H",
        etichetta: "Forza d'attacco (squadra in trasferta)",
        dettaglio:
          "Stesso calcolo ma con il vantaggio casalingo in segno negativo: in trasferta le squadre rendono statisticamente peggio.",
      },
      {
        nome: "pi_home_A",
        etichetta: "Solidità difensiva (squadra di casa)",
        dettaglio: " quanto è solida la difesa della squadra di casa. Valore alto = subisce meno gol.",
      },
      {
        nome: "pi_away_A",
        etichetta: "Solidità difensiva (squadra in trasferta)",
        dettaglio: "solidità difensiva dell'ospite.",
      },
    ],
  },
  {
    gruppo: "Forma recente (ultimi 5)",
    spiegazione:
      "Media dei punti nelle ultime 5 partite, calcolata separatamente per casa e trasferta. Usa shift di una partita: la media include solo le partite già giocate PRIMA di quella che stiamo prevedendo, mai la partita stessa.",
    campi: [
      {
        nome: "home_form_5",
        etichetta: "Punti medi in casa (ult. 5)",
        dettaglio:
          "3 punti per la vittoria, 1 per il pareggio, 0 per la sconfitta, mediati sulle ultime 5 partite giocate in casa. Se la squadra ha meno di 5 partite in casa si usa la media di quante ne ha.",
      },
      {
        nome: "away_form_5",
        etichetta: "Punti medi in trasferta (ult. 5)",
        dettaglio: "stesso calcolo sulle ultime 5 trasferte.",
      },
      {
        nome: "home_goal_diff_5",
        etichetta: "Differenza reti in casa (ult. 5)",
        dettaglio:
          "media di (gol fatti − gol subiti) nelle ultime 5 partite in casa. Coglie il momento della squadra meglio della media punti.",
      },
      {
        nome: "away_goal_diff_5",
        etichetta: "Differenza reti in trasferta (ult. 5)",
        dettaglio: "stesso calcolo sulle ultime 5 trasferte.",
      },
    ],
  },
  {
    gruppo: "Gol rolling (ultimi 5)",
    spiegazione:
      "Media di gol fatti e gol subiti. Separata per casa/trasferta perché le squadre non rendono uguale nei due contesti. Come la forma, usa solo partite già giocate.",
    campi: [
      {
        nome: "home_goals_scored_5",
        etichetta: "Gol fatti in casa (ult. 5)",
        dettaglio: "media dei gol segnati nelle ultime 5 partite in casa.",
      },
      {
        nome: "home_goals_conceded_5",
        etichetta: "Gol subiti in casa (ult. 5)",
        dettaglio: "media dei gol subiti nelle ultime 5 partite in casa.",
      },
      {
        nome: "away_goals_scored_5",
        etichetta: "Gol fatti in trasferta (ult. 5)",
        dettaglio: "media dei gol segnati nelle ultime 5 trasferte.",
      },
      {
        nome: "away_goals_conceded_5",
        etichetta: "Gol subiti in trasferta (ult. 5)",
        dettaglio: "media dei gol subiti nelle ultime 5 trasferte.",
      },
    ],
  },
  {
    gruppo: "Riposo",
    spiegazione: `Giorni dall'ultima partita, saturati a ${REST_CAP}. Oltre ${REST_CAP} giorni il recupero muscolare è completo, quindi 8 giorni e 40 giorni danno lo stesso identico valore. Sotto la soglia la differenza è intera e conta: 2 giorni contro 4 resta 2 giorni di svantaggio. Prima della saturazione i valori arrivavano a 3000 giorni (squadre che rientravano dopo anni di assenza), fuori da qualunque distribuzione sensata.`,
    campi: [
      {
        nome: "home_rest_days",
        etichetta: `Giorni di riposo casa (max ${REST_CAP})`,
        dettaglio: `giorni dall'ultima partita della squadra di casa, limitati a ${REST_CAP}. Una squadra che ha appena finito un turno infrasettimanale avrà 3-4 giorni, una che riposava da tempo avrà ${REST_CAP}.`,
      },
      {
        nome: "away_rest_days",
        etichetta: `Giorni di riposo trasferta (max ${REST_CAP})`,
        dettaglio: `stesso calcolo per l'ospite. Il modello riceve i due valori assoluti, non la differenza: da soli non basta a distinguere "uno riposato e uno no" da "entrambi stanchi".`,
      },
    ],
  },
  {
    gruppo: "Scontri diretti",
    spiegazione:
      "Percentuale di vittorie in casa negli scontri diretti storici fra le due squadre. Nei primi turni di una stagione può essere tutto neutro, perché le due squadre potrebbero non essersi mai incrociate.",
    campi: [
      {
        nome: "h2h_home_winrate",
        etichetta: "Vittorie in casa negli scontri diretti",
        dettaglio:
          "se le due squadre non si sono mai affrontate, il valore è 0.5 (neutro). Altrimenti la quota di partite in cui la squadra di casa ha vinto.",
      },
    ],
  },
  {
    gruppo: "Stagione corrente",
    spiegazione:
      "Un indicatore per le squadre appena promosse. Non viene scritto a mano nel modello con uno sconto: il flag dice solo che questa squadra era assente nella stagione precedente, e il modello impara da solo quanto scontare guardando i dati storici delle promozioni passate.",
    campi: [
      {
        nome: "home_promoted",
        etichetta: "Neopromossa (casa)",
        dettaglio:
          "1 se la squadra di casa era in Serie A la stagione precedente ma non lo è più, quindi è appena salita. Il modello impara da solo il relativo declassamento.",
      },
      {
        nome: "away_promoted",
        etichetta: "Neopromossa (trasferta)",
        dettaglio: "stesso flag per la squadra in trasferta.",
      },
    ],
  },
  {
    gruppo: "Prestazione (ultimi 5)",
    spiegazione:
      "Le statistiche di quello che la squadra ha effettivamente creato, non solo il risultato: tiri totali, tiri in porta, angoli e cartellini, calcolati sulle ultime 5 partite, separati per casa e trasferta. I tiri in porta sono il proxy di expected goals più economico e documentato che esista: misurano la qualità delle occasioni create, non il numero di conclusioni. Le medie sono spostate di una posizione, quindi la partita che si sta prevedendo non entra mai nel proprio valore.",
    campi: [
      {
        nome: "home_shots_5 / away_shots_5",
        etichetta: "Tiri totali (ultimi 5)",
        dettaglio: "media dei tiri nelle ultime 5 partite, lato casa e lato trasferta.",
      },
      {
        nome: "home_shots_t_5 / away_shots_t_5",
        etichetta: "Tiri in porta (ultimi 5)",
        dettaglio: "proxy di expected goals: la qualità delle occasioni create.",
      },
      {
        nome: "home_corners_5 / away_corners_5",
        etichetta: "Angoli (ultimi 5)",
        dettaglio: "proxy di pressione e possesso, indipendente dal risultato.",
      },
      {
        nome: "home_cards_5 / away_cards_5",
        etichetta: "Cartellini (ultimi 5)",
        dettaglio: "disciplina: più cartellini tendono a correlare con più fallimenti e più subiti.",
      },
    ],
  },
  {
    gruppo: "Elo con decadimento",
    spiegazione:
      "Un rating Elo (K=20, vantaggio casa 65 punti) che si aggiorna dopo ogni partita, con decadimento del 25% per stagione non giocata: chi rientra dopo un anno in B non porta con sé il rating di due anni fa. A differenza del Pi-Rating (che usa solo i gol), l'Elo usa i risultati e quindi reagisce subito ai cambiamenti di squadra. Alla partita da prevedere il rating è quello successivo all'ultima partita già giocata: nessun FUTURE.",
    campi: [
      {
        nome: "elo_home / elo_away",
        etichetta: "Rating Elo delle due squadre",
        dettaglio: "stato del rating prima della partita, con decadimento stagionale.",
      },
      {
        nome: "elo_home_expect / elo_away_expect",
        etichetta: "Probabilità attesa dall'Elo",
        dettaglio: "la previsione dell'Elo per la partita, con vantaggio casa: il modello la confronta con le altre feature.",
      },
    ],
  },
  {
    gruppo: "Mercato (quote di apertura)",
    spiegazione:
      "Le probabilità implicite dei bookmaker, private del loro margine con il metodo 'additive' (scelto empiricamente come il più accurato su 6135 partite, verificato a 1e-6 contro penaltyblog). Sono le quote di APERTURA, note prima del fischio: usarle non è hindsight. Il modello impara a correggere il mercato invece di ricostruirlo da zero. Dove una fonte manca (per esempio Pinnacle sulle partite senza copertura) il valore è assente, non zero.",
    campi: [
      {
        nome: "mkt_b365_pH/pD/pA",
        etichetta: "Probabilità implicite bet365 (senza margine)",
        dettaglio: "le quote di apertura di bet365 de-viggate, più il suo margine (overround, ~5.7%).",
      },
      {
        nome: "mkt_pinnacle_pH/pD/pA",
        etichetta: "Probabilità implicite Pinnacle (senza margine)",
        dettaglio: "Pinnacle è il bookmaker più 'sharp': il suo margine medio è 2.6% contro il 5.7% di bet365, quindi la sua stima è più vicina alla verità.",
      },
      {
        nome: "mkt_avg_pH/pD/pA",
        etichetta: "Probabilità implicite del consenso di mercato",
        dettaglio: "la media delle quote di più bookmaker: il consensus batte il singolo bookmaker come predittore.",
      },
    ],
  },
];

const STEPS = [
  {
    n: 1,
    titolo: "Scaricare i dati",
    testo:
      "Ogni lunedì e giovedi una GitHub Action scarica le 17 stagioni di Serie A da football-data.co.uk, dalla 2010/11 a oggi. Nessun dato viene inserito a mano: se la fonte non risponde, la pipeline fallisce invece di produrre previsioni su dati parziali.",
  },
  {
    n: 2,
    titolo: "Pulire e validare",
    testo:
      "Si tengono solo le partite con risultato valido (H/D/A), le date si interpretano nel formato giorno/mese/anno e si verifica che siano in ordine cronologico crescente. L'ordine non è un dettaglio: senza di esso il Pi-Rating userebbe risultati futuri per stimare il passato.",
  },
  {
    n: 3,
    titolo: "Calcolare i Pi-Rating",
    testo:
      "Per ogni squadra si stimano forza d'attacco e solidità difensiva con massima verosimiglianza su un modello Poisson, correggendo per il vantaggio casalingo e per la correlazione fra 0-0 e 1-1 tipica del calcio. I risultati più vecchi di 2 anni pesano la metà.",
  },
  {
    n: 4,
    titolo: "Costruire le feature",
    testo:
      "Ogni partita viene trasformata in una riga di 41 numeri: 17 di base (Pi-Rating, forma, gol, scontri diretti), 8 di prestazione (tiri, tiri in porta, angoli, cartellini), 4 di Elo e 12 di mercato (le quote implicite de-viggate di bet365, Pinnacle e la media di mercato). Tutte le finestre mobili sono spostate di una posizione, così una partita non usa mai informazioni sul proprio esito: le quote usate sono solo quelle di apertura, note prima del fischio.",
  },
  {
    n: 5,
    titolo: "Addestrare e valutare",
    testo:
      "Il modello viene addestrato su tutto lo storico fino al 2025/08/01 e valutato sulle partite successive, che non ha mai visto. 5 combinazioni di iperparametri vengono confrontate con cross-validation temporale e vince la migliore.",
  },
  {
    n: 6,
    titolo: "Prevedere, calibrare e pubblicare",
    testo:
      "Per ogni partita si calcolano le feature usando solo le partite già giocate e si ottengono le probabilità. Queste vengono poi ricalibrate: una calibrazione Dirichlet (Kull et al., NeurIPS 2019) stima su previsioni fuori campione quanto il modello sbaglia la propria confidenza, e corregge le percentuali di conseguenza. Le stime delle partite giocate vengono ricalcolate a ogni aggiornamento con il modello e i dati di oggi: il modello impara dai risultati reali, quindi congelarle sarebbe mostrarre informazioni obsolete.",
  },
];

export default function ModelPage() {
  return (
    <div className="container container-wide doc-page">
      <header className="header">
        <h1>Come funziona il modello</h1>
        <div className="update-line">
          CatBoost + Pi-Rating Dixon-Coles · {DATASET.partite.toLocaleString("it-IT")}{" "}
          partite · {DATASET.stagioni} stagioni
        </div>
      </header>

      <nav className="doc-toc">
        <span>In questa pagina</span>
        <a href="#panoramica">Panoramica</a>
        <a href="#dati">I dati</a>
        <a href="#pirating">Pi-Rating</a>
        <a href="#catboost">CatBoost</a>
        <a href="#feature">Le 41 feature</a>
        <a href="#metriche">Quanto è buono</a>
        <a href="#limiti">Limiti noti</a>
      </nav>

      <section id="panoramica">
        <h2>Panoramica in breve</h2>
        <p>
          Il sistema stima la probabilità di ciascuno dei tre esiti (vincere,
          pareggiare, perdere) per ogni partita. Funziona in due strati: prima
          calcola un <b>Pi-Rating</b> per ogni squadra — una misura di forza
          puramente statistica, non legata alla classifica — poi passa quei
          numeri insieme ad altre 13 variabili a un modello di{" "}
          <b>CatBoost</b>, che impara come combinarli.
        </p>
        <p>
          La scelta di usare due strati non è estetica. I rating da soli
          funzionano bene ma non distinguono "questa squadra è in forma" da
          "questa squadra è stata sfortunata". Aggiungere forma, riposo e
          gol rolling permette a CatBoost di correggere quei rating quando i
          dati recenti dicono altro.
        </p>
        <div className="doc-callout">
          Perché le previsioni su partite già giocate non cambiano mai? Perché
          ricalcolarle significherebbe usare il risultato della partita per
          indovinare il risultato della partita. È il bias da hindsight, e
          renderebbe ogni statistica del sito priva di significato.
        </div>
      </section>

      <section id="dati">
        <h2>I dati</h2>
        <p>
          Fonte unica: <b>football-data.co.uk</b>, il provider che rifornisce
          anche i bookmaker. Non viene usato nessun dato da altre fonti, e la
          pipeline non ha eccezioni: se il download fallisce, la run si ferma.
        </p>
        <ul className="doc-stats">
          <li>
            <b>{DATASET.partite.toLocaleString("it-IT")}</b> partite
          </li>
          <li>
            <b>{DATASET.stagioni}</b> stagioni
          </li>
          <li>
            <b>{DATASET.squadre}</b> squadre distinte
          </li>
          <li>
            dal {DATASET.dal}
          </li>
        </ul>
        <h3>Cosa contiene ogni riga</h3>
        <p>
          Per ogni partita: data, squadra di casa, ospite, gol fatti e subiti,
          esito, e le quote di <b>Bet365</b> (1, X, 2) sia per la partita
          giocata sia per le partite ancora da giocare. Le quote sono lo
          strumento di confronto: sono l&apos;unico benchmark disponibile,
          perché incorporano l&apos;informazione di tutto il mercato.
        </p>
        <h3>La distribuzione degli esiti</h3>
        <p>
          Su {DATASET.partite.toLocaleString("it-IT")} partite:{" "}
          <b>
            {((DATASET.esiti.H / DATASET.partite) * 100).toFixed(1)}%{" "}
            vittorie in casa
          </b>
          ,{" "}
          <b>
            {((DATASET.esiti.D / DATASET.partite) * 100).toFixed(1)}%
            pareggi
          </b>
          ,{" "}
          <b>
            {((DATASET.esiti.A / DATASET.partite) * 100).toFixed(1)}% vittorie
            in trasferta
          </b>
          . Il vantaggio casalingo è il primo effetto che il modello deve
          cogliere, ed è esattamente ciò che il termine <code>home_adv</code>{" "}
          del Pi-Rating cattura.
        </p>
      </section>

      <section id="pirating">
        <h2>Il Pi-Rating (Dixon-Coles)</h2>
        <p>
          Ogni squadra riceve due numeri stimati dai propri gol:{" "}
          <b>attacco</b> (quanto segna) e <b>difesa</b> (quanto subisce). Il
          modello assume che i gol di una partita seguano una distribuzione di
          Poisson:
        </p>
        <p className="doc-formula">
          λ<sub>casa</sub> = exp(attacco<sub>casa</sub> − difesa<sub>ospite</sub> +
          vantaggio casalingo)
        </p>
        <p>
          I parametri si ottengono per <b>massima verosimiglianza</b>: si
          cercano quei valori che rendono i risultati osservati più probabili.
          L&apos;ottimizzatore è <code>{PI_RATING.optimizer}</code> con{" "}
          {PI_RATING.maxiter} iterazioni.
        </p>
        <h3>La correzione Dixon-Coles</h3>
        <p>
          Il Poisson puro sbaglia una cosa nota: sottostima la frequenza degli
          0-0 e sovrastima quella degli 1-1, perché i risultati a basso punteggio
          sono più frequenti di quanto il modello indichi. Dixon e Coles hanno
          aggiunto un fattore correttivo per questi due casi.
        </p>
        <h3>Il decay temporale</h3>
        <p>
          Un&apos;attacco valido nel 2012 non dice molto di una squadra nel
          2026. Ogni partita pesa{" "}
          <code>0.5^(anni/2)</code>: un risultato di due anni fa conta la metà,
          di quattro anni un quarto. La <b>mezza vita è di{" "}
          {PI_RATING.decay} anni</b>. Questa è la scelta che rende i rating
          ricambiabili partita per partita invece che stagionali.
        </p>
        <h3>Perché questo precede il modello</h3>
        <p>
          Un&quot;idea ricorrente nel calcio è usare la classifica. Non lo
          facciamo: la classifica è un effetto delle partite passate, quindi
          usarla per prevedere le future non aggiunge nulla. Il Pi-Rating
          guarda ai <i>gol</i> al netto del calendario avverso, ed è per questo
          che a fine stagione non si «incasina» come una classifica.
        </p>
      </section>

      <section id="catboost">
        <h2>Il modello CatBoost</h2>
        <p>
          CatBoost è un modello a alberi potenziati (gradient boosting). Costruisce
          una sequenza di alberi decisionali, ognuno dei quali corregga gli
          errori del precedente. Su ogni foglia si registra una probabilità
          per i tre esiti, e la previsione finale è la somma pesata di tutte le
          foglie.
        </p>
        <h3>Perché CatBoost e non una regressione logistica</h3>
        <p>
          Il problema non è lineare: la relazione tra «8 giorni di riposo» e
          probabilità di vincere non è una retta, e le variabili si influenzano
          a vicenda. Gli alberi colgono automaticamente le interazioni (per
          esempio «form alta <i>e</i> trasferta tirata» pesano più di ciascuna
          delle due da sola) senza che nessuno debba scriverle a mano.
        </p>
        <h3>Perdere addestrando in ordine cronologico</h3>
        <p>
          Un errore comune sarebbe mescolare le partite a caso: il modello
          vedrebbe il futuro mentre impara il passato, e le metriche sarebbero
          gonfiate e false. Qui la divisione è <b>temporale</b>: tutto ciò che
          viene prima del 1º agosto 2025 va in addestramento, tutto ciò che viene
          dopo va in validazione e non viene mai visto.
        </p>
        <h3>Iperparametri</h3>
        <ul className="doc-list">
          <li>
            <b>Funzione di perdita:</b> MultiClass (tre classi: H, D, A)
          </li>
          <li>
            <b>Profondità:</b> {CATBOOST.depth} — controlla quanti passi di
            decisione può fare ciascun albero
          </li>
          <li>
            <b>Learning rate:</b> {CATBOOST.learningRate} — quanto pesa ogni
            albero nuovo rispetto al modello costruito finora
          </li>
          <li>
            <b>Alberi:</b> {CATBOOST.iterazioni} (baseline), con ricerca su{" "}
            {CATBOOST.grid.length} combinazioni via{" "}
            {CATBOOST.foldCount}-fold cross-validation temporale
          </li>
          <li>
            <b>Seme fisso:</b> {CATBOOST.seed}, per riproducibilità completa
          </li>
        </ul>
        <p>
          La ricerca suiperparametri avviene solo sui dati di addestramento: la
          validazione finale resta un test mai visto. È il motivo per cui si
          accetta un margine di errore più ampio sui risultati pubblicati.
        </p>
      </section>

      <section id="feature">
        <h2>Le 41 feature, una per una</h2>
        <p>
          Il modello riceve esattamente questi {FEATURES.reduce(
            (n, g) => n + g.campi.length,
            0
          )}{" "}
          numeri per partita. La lista è identica in addestramento e in
          previsione: CatBoost abbina le colonne per <i>nome</i>, quindi se le
          due liste divergessero le previsioni sarebbero sbagliate senza che
          nulla segnali l&apos;errore.
        </p>
        {FEATURES.map((g) => (
          <div key={g.gruppo} className="doc-group">
            <h3>{g.gruppo}</h3>
            <p className="doc-group-intro">{g.spiegazione}</p>
            {g.campi.map((c) => (
              <div key={c.nome} className="doc-feature">
                <div className="doc-feature-head">
                  <code>{c.nome}</code>
                  <span>{c.etichetta}</span>
                </div>
                <p>{c.dettaglio}</p>
              </div>
            ))}
          </div>
        ))}
      </section>

      <section id="metriche">
        <h2>Quanto è buono, davvero</h2>
        <p>
          Due misure, perché rispondono a domande diverse.{" "}
          <b>Accuracy</b> dice «quante volte ha indovinato l&apos;esito
          esatto». <b>RPS</b> (Ranked Probability Score) dice «quanto sono
          bene distribuite le sue probabilità» — per questo una previsione che
          dice 40-30-30 senza sbagliare l&apos;esito è comunque migliore di
          una che dice 99-0,5-0,5 e indovina. Per il calcio l&apos;RPS è la
          misura giusta, perché le quote alte sono quasi sempre sbagliate.
        </p>
        <ul className="doc-metrics">
          <li>
            <b>{(METRICHE.accuracy * 100).toFixed(1)}%</b>
            <span>
              accuracy su {428} partite mai viste — un&apos;impresa sola, il
              modello indovina circa una partita su due
            </span>
          </li>
          <li>
            <b>{METRICHE.rpsModello.toFixed(4)}</b>
            <span>RPS del modello (più basso è meglio)</span>
          </li>
          <li>
            <b>{METRICHE.rpsBookmaker.toFixed(4)}</b>
            <span>RPS di Bet365 — il benchmark da battere</span>
          </li>
        </ul>
        <div className="doc-callout warn">
          Il modello è <b>leggermente peggio del bookmaker</b> su RPS. La
          differenza ({((1 - METRICHE.rpsModello / METRICHE.rpsBookmaker) * 100).toFixed(1)}% in termini di errore relativo) è
          piccola, e su 50 partite non è abbastanza per concludere che il
          modello abbia o non abbia edge reale: il margine di errore statistico
          è molto più ampio della differenza osservata. Essere sotto il
          bookmaker è comunque la norma: il bookmaker incorpora più
          informazione (il proprio book di scommesse) e ha un margine di
          profitto da coprire.
        </div>
        <p>
          Per dare un ordine di grandezza: un RPS di circa{" "}
          {METRICHE.rpsNeutro.toFixed(2)} è considerato il livello di un
         Pronostico ragionevole, e un bookmaker professionista sta intorno a{" "}
          {(METRICHE.rpsBookmaker * 10).toFixed(2)}–{METRICHE.rpsBookmaker.toFixed(2)} su
          questo campione.
        </p>
        <h3>La simulazione Kelly dice qualcosa in più</h3>
        <p>
          La pagina{" "}
          <Link href="/kelly" className="doc-link">
            simulazione Kelly
          </Link>{" "}
          applica la regola di Kelly su ogni partita dove il modello ha
          probabilità maggiore del bookmaker. Il risultato è netto: in questa
          stagione parziale la simulazione è in perdita. Non è un bug — è
          la misura onesta di cosa succede quando si scommette con un edge
          stimato: se l&apos;edge non è reale, la strategia lo diluisce fino a
          farla perdere.
        </p>
      </section>

      <section id="limiti">
        <h2>Limiti noti</h2>
        <ul className="doc-list">
          <li>
            <b>Una sola stagione.</b> Le previsioni pubblicate sono su 50
            partite. Per distinguere un modello abile da uno sfortunato servono
            centinaia di partite, quindi le percentuali di accuratezza pubblicate
            vanno lette con prudenza.
          </li>
          <li>
            <b>Sostanzialmente pari al bookmaker.</b> Sul test set (438
            partite, split temporale) il modello ha il 56.2% di accuratezza e
            un F1 macro di 0.489, contro il 54.3% e 0.417 di bet365: lo
            batte sull'esito. Ma sul RPS — la metrica che premia la
            calibrazione della probabilità, non solo la scelta dell'esito — è
            praticamente in pari (0.13047 contro 0.13011). In sostanza
            eguaglia il mercato e lo corregge marginalmente. Un modello che
            lo presentasse come "battitore di bookmaker" starebbe esagerando:
            il vantaggio è reale ma piccolo, e il ROI ottenuto alle quote
            offerte non è statisticamente significativo.
          </li>
          <li>
            <b>Il modello non anticipa il mercato.</b> Misurando quanto il
            mercato sposta le proprie quote fra apertura e chiusura, l'esito
            preferito dal modello viene seguito dal mercato il 51% delle
            volte, contro il 54% per l'esito che il mercato già preferiva: il
            modello non ha informazione privata sul movimento, lo copia
            meglio.
          </li>
          <li>
            <b>Le quote future non esistono.</b> Le partite non ancora giocate
            mostrano «non disponibile»: le quote vengono pubblicate solo poco
            prima del match.
          </li>
          <li>
            <b>Le variabili inizio stagione sono neutre.</b> A settembre
            forma, H2H e gol rolling non hanno ancora dati. Le previsioni delle
            prime giornate dipendono quindi quasi solo dal Pi-Rating. È
            verificabile dal pannello «Variabili di prediction».
          </li>
          <li>
            <b>Niente anomalie o notizie.</b> Il modello vede solo numeri di
            statistiche e quote. Infortuni, cambi di allenatore, squalifiche non
            entrano in nessun calcolo.
          </li>
        </ul>
      </section>

      <div className="kelly-back">
        <Link href="/">← Torna alle predizioni</Link>
      </div>

      <footer className="footer">
        Modello: CatBoost (MultiClass) + Pi-Rating Dixon-Coles, decay {PI_RATING.decay}{" "}
        anni
        <br />
        Storico di training: Serie A {DATASET.dal} → {DATASET.al} ·{" "}
        {DATASET.partite.toLocaleString("it-IT")} partite · Dati: football-data.co.uk
        <br />
        Le percentuali sono stime statistiche, non consigli.
      </footer>
    </div>
  );
}
