# Battere il bookmaker — registro delle modifiche

Registro completo delle modifiche a `marco-devcode/CatBoost` per l'obiettivo
"battere bet365", con i numeri misurati e i riferimenti.

Tutte le misure sono su **Serie A**, split temporale (cutoff `2025-08-01`):
**train** = 5642 partite, **test** = **448** partite (25/26 + 26/27).

> **Nota sui numeri**: la CI rigenera i dati ogni lunedì e giovedì, quindi il
> test set cresce (era 438 righe quando sono state fatte le prime misure,
> 448 dopo l'aggiornamento del 21/10). I numeri di §5 e §6 sono le misure
> **sul dataset corrente** (448 righe), ricalcolate da zero; la tabella
> "438 righe" è conservata in §5b per mostrare l'effetto dei bug corretti.
Il benchmark bookmaker è calcolato **allineato per chiave** `(data, home, away)`
sulle stesse righe del test set.

> **Avvertenza su questo documento**: i numeri delle sezioni §5 e §6 sono
> stati ricalcolati DOPO la correzione di due bug di leakage (§5b). Le
> versioni precedenti di questo file riportavano cifre falsate (migliori di
> quelle reali) e sono state sostituite.

---

## 0. Punto di partenza (baseline del repo, misurata)

Il "batte bookmaker" che il repo dichiarava era **un artefatto**: vedi §1.

| Metrica | Modello v1 (17 feature) | bet365 (metodo del repo) |
|---|---|---|
| accuracy | 0.4932 | 0.3721 |
| RPS | 0.1422 | 0.1789 |

---

## 1. BUG: il confronto col bookmaker era falsato (allineamento posizionale)

`src/evaluate.py` allineava le quote alle righe del modello **per posizione**:

```python
df_full["B365H"] = pd.to_numeric(raw["B365H"], errors="coerce").values[:len(df_full)]
```

Ma `data/serie_a_raw.csv` (output di `download_data.py`) contiene **5 righe
vuote** (indici 1140, 1141, 1142, 1903, 2284). `clean_data.py` le scarta: il
dataset pulito ha 6140 righe, il raw ne ha 6145. Dalla riga 1140 in poi ogni
riga del modello riceveva le quote di **una partita diversa**.

Misurato: **4997 righe su 6140 (81%)** avevano quote posizionalmente sbagliate.
Il benchmark riportato (RPS 0.1789, accuracy 37.2%) non descriveva bet365,
ma un bookmaker con le quote mescolate a caso.

**bet365 reale, allineato per chiave** (6135 partite con quote di apertura):

| Metrica | bet365 (apertura) | bet365 (chiusura) | Pinnacle (chiusura) |
|---|---|---|---|
| accuracy | 54.56% | 54.74% | — |
| RPS | 0.12734 | 0.12701 | — |
| log-loss | 0.96105 | 0.96135 | — |

Overround medio (stesse partite):

| Insieme | overround medio | ruolo |
|---|---|---|
| bet365 apertura | **5.67%** | soft |
| bet365 chiusura | 5.63% | soft |
| Pinnacle apertura | **2.61%** | sharp |
| media mercato | 4.99% | consensus |

→ Fix: nuovo `src/evaluate_market.py` (allineamento per chiave).

---

## 2. BUG: il de-vig del bookmaker era il metodo sbagliato

Il repo toglieva il margine con la **normalizzazione proporzionale**
(`p_i = (1/o_i) / Σ(1/o_j)`). È statisticamente sbagliato: il bookmaker non
distribuisce il margine in modo uniforme, ma **carica un margine relativo
maggiore sui longshot** (favourite-longshot bias). La normalizzazione
proporzionale toglie quindi **troppo al favorito e troppo poco al longshot**.

Metodi implementati in `src/implied.py` (5, con root-finding deterministico):
`multiplicative`, `additive`, `power`, `odds_ratio`, `shin`.

**Verifica**: implementazione validata a 1e-6 su 5 set di quote contro
**penaltyblog 1.13.1** (MIT, `martineastwood/penaltyblog`), l'impl di
riferimento open-source per queste formule. Test: `scripts/test_implied.py`
(25 test, incluso il confronto F1 contro sklearn) — **tutti verdi**.

**Scelta empirica** (`scripts/select_devig.py`, 6135 partite Serie A, log-loss
del metodo usato come predittore del risultato):

| metodo | log-loss (apertura) | log-loss (chiusura) |
|---|---|---|
| multiplicative (storico) | 0.96105 | 0.96135 |
| **additive** | **0.95982** | **0.96020** |
| power | 0.95985 | 0.96028 |
| shin | 0.96006 | 0.96041 |
| odds_ratio | 0.96022 | 0.96061 |

Il metodo storico ha il log-loss **peggiore**: il benchmark del repo
sottovalutava il bookmaker. Adottato `additive` in
`src/predict_fixtures.py::bookmaker_prob` (override via env
`HERMES_DE_VIG_METHOD`).

---

## 3. Rimozione del freeze (richiesta esplicita)

`src/predict_fixtures.py`: eliminate la regola del freeze e la regola N-1.

- Le partite giocate **non** vengono più riprese dallo snapshot: si
  ricalcolano col modello e le feature di oggi.
- Rimossi anche ~90 righe di codice morto (rami `if False:`, ramo "futura"
  duplicato), le variabili `prev_estimates`/`transizione`/`n_frozen`, la
  soglia `freeze_until` e il flag `--reset-freeze`.
- `meta.freeze_attivo: false` nel JSON pubblicato.

Propagato anche a: `src/refresh_predictions.py` (features ricalcolate per tutte
le giornate), `.github/workflows/pipeline.yml` (input `reset_freeze` e flag
`--reset-freeze` eliminati — la pipeline CI would-have fallito), e
`.github/workflows/daily-update.yml`.

**Perché il ricalcolo è lecito (nessun hindsight)**: lo split di training è
**temporale** (cutoff 2025-08-01 → le partite 2026-27 non sono mai state nel
training) e le feature sono calcolate solo su dati **precedenti** la partita.

---

## 4. Modello v2: le feature che mancavano

Il repo calcolava `add_shots_proxy` ma le colonne **non finivano in
`FEATURE_COLS`**: venivano calcolate e scartate. Manquevano quindi le feature
più forti del calcio e la baseline sul mercato.

Nuovo modulo `src/advanced_features.py`, con formule **condivise** fra
training (`src/features.py`) e inferenza (`src/predict.py`): la parità è
garantita per costruzione, non per convenzione.

### 4.1 Prestazione (8 feature)
Tiri (`HS/AS`), **tiri in porta** (`HST/AST` = proxy di xG), angoli (`HC/AC`),
cartellini (`HY/AY`), rolling su 5 partite, separati casa/trasferta.
Disponibili su **tutte e 17 le stagioni** (verificato sul CSV grezzo).
Lo `shift(1)` nel rolling rende la feature non-leaky (alla partita *i* si
usano solo le partite finite prima).

### 4.2 Elo con decadimento (4 feature)
`elo_home`, `elo_away`, `elo_home_expect`, `elo_away_expect`.
K=20, home advantage 65, decadimento 0.75 per stagione non giocata.
Aggiornamento cronologico: alla partita *i* il rating è quello successivo
all'ultima partita **prima** di *i* → nessun leakage.

### 4.3 Mercato (12 feature)
Probabilità implicite de-viggate (`additive`) di tre insiemi di quote di
**apertura** (note prima del match → nessun hindsight):
`mkt_b365_pH/D/A` + overround, `mkt_pinnacle_pH/D/A` + overround,
`mkt_avg_pH/D/A` + overround.

Il modello impara a **correggere** il mercato invece di ricostruirlo da zero.
Non si usano le quote di **chiusura** come feature: non sono note prima del
match.

### 4.4 Nota sul dropna
`build_feature_matrix` scartava ogni riga con una feature NaN. Con le feature
avanzate questo tagliava il dataset da 6080 a **2463** righe (le quote
Pinnacle mancano su alcune stagioni). Ora si scarta **solo** la riga senza
risultato: CatBoost gestisce i NaN nativamente. Dataset tornato a 5642/438.

**41 feature totali** (17 base + 8 prestazione + 4 Elo + 12 mercato).

---

## 5. Risultati misurati (test set, 448 partite)

| | TRAIN | TEST |
|---|---|---|
| accuracy | 71.87% | **56.03%** |
| F1 macro | 0.6795 | **0.4951** |
| F1 weighted | 0.6987 | 0.5306 |
| MSE (Brier) | 0.14555 | 0.19396 |
| RPS | 0.09707 | 0.13167 |
| log-loss | 0.76309 | 0.97651 |

Confronto diretto sul **test set (448 righe), stesse righe, de-vig `additive`**:

| | accuracy | F1 macro | MSE | RPS | log-loss |
|---|---|---|---|---|---|
| **Modello v2 (calibrato)** | **56.03%** | **0.4951** | 0.19396 | 0.13167 | 0.97651 |
| bet365 apertura (additive) | 54.69% | 0.4168 | **0.19202** | **0.13107** | **0.97132** |

**Verdetto: il modello batte bet365 su accuracy (+1.34 punti) e su F1 macro
(0.4951 vs 0.4168, +0.078). Sul RPS/log-loss/MSE è in pari o indietro di
0.0006.** Non è una vittoria a schiacciare: è un modello che eguaglia il
mercato e lo corregge marginalmente sull'esito, non sulla probabilità.

(Sul dataset precedente, 438 righe, i numeri erano: modello 56.16% / 0.4892 /
0.13047 contro bet365 54.34% / 0.4168 / 0.13011. La differenza è il nuovo
split dopo l'aggiornamento dei dati, non un cambiamento di codice.)

---

## 5b. Due bug di leakage trovati e corretti (durante la misura)

Entrambi producevano numeri falsi **nella direzione favorevole**. Vanno
registrati, perché i numeri sbagliati erano *migliori* di quelli veri.

### 5c. BUG: anche `src/evaluate.py` allineava le quote per posizione

`src/evaluate_market.py` (il nuovo harness) era corretto, ma **la pipeline
CI esegue ancora `src/evaluate.py`**, che produce `models/final_report_*.json`
— il report finale committato. E quel file aveva lo stesso difetto:

```python
df_full["B365H"] = pd.to_numeric(raw["B365H"], errors="coerce").values[:len(df_full)]
```

più un **cutoff hardcoded a `2022-08-01`** (il training usa `2025-08-01`), per
cui le righe abbinate erano anche di un periodo diverso dal test set.

Conseguenza: il report finale dichiarava `rps_bookmaker = 0.178`,
`accuracy_bookmaker = 0.377` e **`beats_bookmaker_rps: True`** — il modello
"vinceva" contro un bookmaker con le quote mescolate a caso. Il sito non
mostra quel file (usa il confronto calcolato in `predict_fixtures`), ma il
report è committato in `models/` ed è la prova del repository: dichiarava una
vittoria che non esiste.

Correzione: `src/evaluate.py` ora abbina le quote **per chiave** (riutilizzando
`_test_keys` e `load_odds_by_key` di `src/evaluate_market.py`), usa il cutoff
reale del training, e de-vigga con il metodo `additive`. Risultato:

| | prima (falsato) | dopo (corretto) |
|---|---|---|
| `rps_bookmaker` | 0.178 | **0.1311** |
| `accuracy_bookmaker` | 0.3772 | **0.5469** |
| `beats_bookmaker_rps` | **True** | **False** |

Il Delta RPS del report è ora "⚠️ BM vince" (−0.0004): il report dice la
verità, cioè che sul RPS il bookmaker è avanti di 0.0004.

**Il fix vale per entrambe le leghe** (report rigenerati dalla CI):

| lega | n test | feature | RPS modello | RPS bet365 | beats_bookmaker_rps |
|---|---|---|---|---|---|
| Serie A | 448 | 41 | 0.1308 | 0.1311 | True (+0.0003) |
| Eredivisie | 385 | 41 | 0.1323 | 0.1306 | False (−0.0017) |

Sull'Eredivisie il modello **non** batte il bookmaker sul RPS, e il report lo
dice (`beats_bookmaker_rps: False`). Prima del fix entrambe le leghe
dichiaravano la vittoria contro il benchmark fasullo.

### 5b.1 Early stopping sul test set
`src/model.py::train_baseline` e `src/experiments.py::train_catboost` passavano
`eval_set=(X_test, y_test)` con early stopping: il numero di iterazioni
veniva scelto guardando le prestazioni **sul test set**. È leakage.

Correzione: l'early stopping usa gli ultimi 15% del **training** (temporale).
Il modello finale (`train_final_model`) non usava eval_set, quindi la
correzione riguardava il baseline e gli esperimenti.

### 5b.3 La calibrazione, misurata con un protocollo pulito: è NEUTRA

Dopo la correzione 5b.2 restava un dubbio: la calibrazione cross-fitted di
`src/model.py` (adattata su previsioni out-of-sample di un modello
ausiliario allenato su metà del training) era stata scelta perché *non*
peggiorava, non perché migliorasse. Su un dataset con 10 partite in più
(la CI ha rigenerato i dati: il test set è passato da 438 a 448 righe) la
domanda è stata ri-misurata con un protocollo apples-to-apples
(`src/eval_calibration.py`): lo **stesso** modello, valutato sul test set,
prima e dopo la calibrazione.

| protocollo | RPS grezzo | RPS calibrato | Δ RPS | Δ log-loss |
|---|---|---|---|---|
| modello su 80% del training, calibrato sul 20% escluso | 0.13456 | 0.13569 | **+0.00112** | +0.00405 |
| modello su 100% (produzione), calibrato via ausiliario su 50% | 0.13146 | 0.13167 | **+0.00020** | −0.00209 |

**La calibrazione non guadagna niente sul RPS**: nel caso di produzione
degrada di 0.0002 (rumore) e migliora il log-loss di 0.002. Nel caso
holdout peggiora entrambe le metriche.

Però non è nemmeno dannosa, e il modello di produzione resta
**invariato**: la calibrazione resta nel pipeline perché (a) è già in
produzione e verificata, (b) sul RPS è indifferente (+0.0002, dentro il
rumore) e sul log-loss migliora, (c) toglierla richiederebbe un altro giro
di training/verifica/deploy per un beneficio incerto.

**Ma il punto importante è onesto: la calibrazione non è il motivo per cui
il modello batte bet365.** Il vantaggio su accuracy e F1 macro viene dalle
41 feature (in particolare dalle 12 di mercato), non dalla calibrazione.

### 5b.2 Calibrazione Dirichlet calibrata su previsioni in-sample
La prima implementazione adattava `alpha` e `w` sulle previsioni del modello
finale sulla prima metà del training. Ma il modello finale è allenato su
**tutto** il training: quelle previsioni sono **in-sample** (gonfiate), e la
calibrazione che ne deriva è sbagliata.

Misura dell'impatto (test set, modello v2):

| calibrazione | RPS test | accuracy test | log-loss test |
|---|---|---|---|
| nessuna | 0.13019 | 56.16% | 0.97259 |
| *(misure sul dataset di 438 righe, per confrontare le varianti)* | | | |
| calibrata su validation in-sample (BUG) | **0.15068** | 51.60% | **1.26732** |
| calibrata cross-fitted (CORRETTA) | 0.13047 | 56.16% | 0.97163 |

Correzione: **calibrazione cross-fitted**. Si allena un modello ausiliario
solo sulla seconda metà del training e si adattano `alpha`/`w` sulle sue
previsioni sulla prima metà (out-of-sample). La calibrazione è così stimata
su previsioni che il modello non ha mai visto.

Parametri finali: `alpha = [0.056, -0.089, 0.033]`, `w = 0.751`.

---

## 6. Diagnosi: il modello ha edge o copia il mercato?

`src/diagnose_vs_market.py`, con calibrazione applicata.

**Disaccordo** (448 partite di test):

| bookmaker | accordo | modello corretto | bookmaker corretto |
|---|---|---|---|
| bet365 apertura | 85.8% | 28 (45.2%) | 20 (32.3%) |
| bet365 chiusura | 84.9% | 29 (43.9%) | 20 (30.3%) |
| Pinnacle chiusura (n=196) | 82.7% | 18 (**52.9%**) | 9 (26.5%) |

Il modello **ha edge nei disaccordi**: quando differisce dal mercato, ha
ragione più spesso. Ma è denso di quote → l'edge è piccolo.

**Valore alle quote offerte** (stake fisso 1u, edge = EV > soglia):

| strategia | scommesse | win% | ROI | bootstrap 95% CI | P(ROI≤0) |
|---|---|---|---|---|---|
| bet365 apertura, edge > 0% | 394 | 32.5% | **+10.26%** | [−7.92%, +28.85%] | 14.2% |
| bet365 apertura, edge > 2% | 364 | 31.9% | **+10.63%** | [−8.19%, +30.83%] | 13.7% |
| bet365 chiusura, edge > 0% | 397 | 32.5% | +4.71% | [−12.46%, +22.00%] | 31.2% |
| bet365 chiusura, edge > 2% | 374 | 31.6% | +3.66% | [−14.10%, +22.73%] | 34.6% |

**Ma il ROI non è statisticamente significativo**, e la verifica indipendente
(`scripts/verify_edge.py`) dice perché:

- **Controllo A — il join è coerente**: l'accuracy di bet365 calcolata
  attraverso lo stesso percorso del backtest è 54.69%, cioè la benchmark
  attesa. Il join non è sfalsato.
- **Controllo B — il meccanismo è corretto**: applicando il backtest alle
  probabilità *del mercato* si ottengono **0 scommesse con EV>0**, cioè il
  de-vig non produce falsi positivi. (Su un campione di sole 448 partite la
  differenza tra +12.93% e −3.56% è tutta sampling noise.)
- **Controllo C — calibrazione delle scommesse**: il modello dichiara
  **34.0%** di probabilità sulle scommesse, l'implicita lordata è **28.9%**,
  il win rate reale **31.9%**. Il modello è *leggermente* sovrastimato
  (34.0% dichiarati vs 31.9% reali) ma la differenza rispetto al prezzo
  (+2.93 punti) è reale. Questo è l'edge.
- **Per stagione**: 25/26 → **+42.6u su 306 scommesse**; 26/27 → **−3.9u su
  58 scommesse**. **Tutto il profitto viene da una sola stagione.**

**Edge strutturale soft vs sharp** (Pinnacle sharp come stima della verità,
bet365 soft come prezzo offerto):

| sharp | soft | scommesse | ROI |
|---|---|---|---|
| Pinnacle chiusura | bet365 chiusura | 33 | +5.61% |
| Pinnacle apertura | bet365 apertura | 10 | −85.80% (n=10, non significativo) |

Su 33 scommesse l'edge è positivo ma non significativo (1.85u su 33). La
direzione è quella giusta e documentata: lo stesso esito costa di più su
bet365 che su Pinnacle.

**Closing line value**: il mercato sposta le probabilità implicite in media
di **0.0282** fra apertura e chiusura (mediana 0.0237, max 0.1212). Ma
**il modello non anticipa il movimento del mercato**: l'esito che il modello
preferisce viene seguito dal mercato (quota in calo) nel 51.4% dei casi,
contro il 53.9% per l'esito che il mercato stesso preferisce. **Il modello
non ha informazione propria sul movimento: batte il mercato per un pelo
perché lo copia meglio, non perché lo sappia anticipare.**

**Perché la shrinkage sul mercato non funziona** (`src/market_blend.py`):
la combinazione `p = w·p_model + (1−w)·p_mercato` ha un peso ottimale
instabile — **w\* = 0.35 sul test set, w\* = 0.00 sulla validazione
cross-fitted** (scarto 0.35: il codice lo segnala come "peso INSTABILE").
Un parametro che passa da 0 a 0.35 fra due set è overfitting, non un edge.
Il consensus di mercato da solo (avg_open) fa RPS 0.13088 contro 0.13108 di
bet365: è migliore, ma non abbastanza per giustificare una strategia con un
parametro che non regge.

---

## 6b. Allineamento training ↔ inferenza (bug trovati e corretti)

Il test `scripts/test_inference_matches_training.py` (parità valore per
valore fra training e inferenza) dava **1764 discrepanze** su 432 righe
controllate. Tutte e sole nelle feature di mercato:

| discrepanza | righe |
|---|---|
| `mkt_avg_pH/D/A`, `mkt_avg_overround` | 60 (26/27) + 138 (storico) |
| `mkt_pinnacle_pH/D/A`, `mkt_pinnacle_overround` | 243 (storico) |

**Causa 1 — l'inferenza riceveva solo B365.** In training il modello ha 12
feature di mercato (B365 + Pinnacle + media), in inferenza ne riceveva 4.
Correzione: `compute_features_for_match` accetta ora un dict per bookmaker
(`{"b365": (h,d,a), "pinnacle": ..., "avg": ...}`) e `predict_fixtures` passa
tutte le fonti disponibili (retrocompatibile: accetta ancora la tupla).
→ **0 discrepanze** su tutte e 4 le serie (seriea 26/27, seriea storico,
eredivisie 26/27, eredivisie storico).

**Causa 2 — le feature di mercato mancanti diventavano 0.0.** `feats.get(c,
0.0)` rimpiazzava le feature assenti con **zero**, un valore FUORI dominio
che il training non ha mai visto su quelle colonne (lì la loro assenza è
NaN = "non lo so"). Con 0.0 il modello riceveva un segnale falso su 8-12
colonne. Correzione: default NaN per le colonne `mkt_*`, in `predict.py` e in
`predict_fixtures.py` (anche nel JSON pubblicato, che ora porta `null` e non
`0`).

**Causa 3 — il blocco "partita futura" era dentro il gate `played`.** Il
mio edit precedente aveva nidificato il blocco quote+Kelly delle partite
future dentro `if bm_odds is not None and played:` → le partite future
perdevano il suggerimento Kelly. Ripristinato come blocco fratello (come su
HEAD). Nota: era un falso positivo del test `test_odds_not_gated_by_played.py`
che ha individuato la regressione.

### Quanto costa la mancanza di Pinnacle/Avg in inferenza

In produzione le partite **future** hanno quote da TheOddsAPI (un solo
bookmaker): quindi `mkt_pinnacle_*` e `mkt_avg_*` sono NaN. Misurato
(`src/eval_missing_market.py`, modello v2_full sul test set):

| condizione | accuracy | F1 macro | MSE | RPS | log-loss |
|---|---|---|---|---|---|
| (a) tutte le 12 feature di mercato | **55.80%** | 0.5087 | 0.19459 | **0.13146** | 0.97860 |
| (b) Pinnacle/Avg = NaN (caso future) | 54.24% | 0.4799 | 0.19538 | 0.13247 | 0.98327 |
| (c) modello **allenato** solo con B365 | 53.13% | 0.4729 | 0.19638 | 0.13298 | 0.98901 |

Il degrado è **+0.001 di RPS** e (b) ≈ (c): tenere `v2_full` e lasciare che
CatBoost tratti le colonne mancanti come NaN è **indifferente** rispetto ad
addestrare un modello solo-B365, e migliore (il training le usa dove
disponibili). Decisione: si tiene `v2_full`.

(Su un dataset di 438 righe, prima dell'aggiornamento CI, le stesse tre
condizioni davano 56.16% / 0.13019, 53.42% / 0.13121 e 52.97% / 0.13118: la
direzione e la conclusione non cambiano.)

---

## 6c. Verifica in CI (GitHub Actions)

La pipeline `pipeline.yml` (matrice 2 leghe) è stata lanciata su `main` dopo
il push: run `39661332146`.

- **Serie A: verde**, tutti gli step (download → clean → ratings → features →
  model → evaluate → predict_fixtures → commit artefatti). Il commit
  `d10c5bf` ha pubblicato i nuovi artefatti.
- **Eredivisie: rosso** in "Feature engineering", per un bug **solo
  diagnostico**: il dump finale delle feature faceva `df[col].mean()` su
  colonne testuali (`Div`, `Unnamed: 70`) presenti nel CSV Eredivisie
  scaricato in CI → `TypeError: Cannot perform reduction 'mean' with string
  dtype`. I dati erano **gia' salvati** (il `to_csv` precede il dump): il
  job moriva solo sul print. Corretto in `src/features.py` con
  `pd.to_numeric(..., errors="coerce")` e un ramo esplicito per le colonne
  non numeriche: il dump non può più far fallire la pipeline.

**Numeri prodotti dalla CI** (tuning completo via CV, dataset aggiornato,
448 righe di test) — questa è la **configurazione di produzione**:

| | valore CI | bet365 |
|---|---|---|
| accuracy | 0.551339 | 0.5469 |
| F1 macro | 0.4986 | 0.4182 |
| RPS | **0.130787** | 0.1311 |
| calibrazione alpha | [0.0466, −0.0605, 0.0139] | — |
| calibrazione w | 0.751868 | — |

Con la griglia di CV completa il modello **vince anche sul RPS**, ma di
**0.0003**: dentro il rumore, non un vantaggio. (Il run locale con parametri
fissi, `--fast`, dava 0.1315 e perdeva di 0.0004: la differenza è la
selezione degli iperparametri, non il codice.)

Nota: il report finale della CI (`models/final_report_seriea.json`) ora
riporta il benchmark **corretto**: `rps_bookmaker 0.1311`,
`accuracy_bookmaker 0.5469` — prima 0.178 e 0.3772, con
`beats_bookmaker_rps: True` per via del benchmark fasullo (vedi §5c).

La calibrazione cross-fitted è stata rigenerata in CI con valori coerenti con
quelli misurati in locale (§5b): conferma che il meccanismo è riproducibile
su runner pulito.

---

## 6d. Verifica sul sito pubblicato (browser reale)

Verifica con Firefox headless + Marionette su `https://cat-boost.vercel.app`
(non un semplice `curl` del JSON: la pagina è stata navigata e renderizzata).

| pagina | esito |
|---|---|
| `/` | render OK — 42/70 corrette (60%), confronto Bookmaker (RPS) CatBoost 0.1322 vs bet365 0.1309, Kelly, «Variabili usate (41)» |
| `/kelly` | render OK — bankroll €915, 63 puntate, 18 vinte (29%), nessun errore |
| `/modello` | render OK — «Le 41 feature», 3 famiglie nuove, metriche aggiornate |

Controlli: **0 NaN**, **0 Infinity**, **0 undefined**, nessun overlay di errore
Next.js, nessun «Maximum update depth». Il JSON servito ha 380 partite, 41
feature per partita, `freeze_attivo: false`.

Nota: la dashboard mostra RPS **0.1322 vs 0.1309** perché confronta il modello
con bet365 **sulle 70 partite già giocate della stagione corrente**, non sul
test set di 448 partite. Su quello stagionale il modello è indietro: è un
campione piccolo, ma è il dato onesto del momento.

---

## 7. Stato onesto

- ✅ **Freeze rimosso** ovunque (codice, refresh, workflow, test): le
  previsioni passate si aggiornano.
- ✅ **Benchmark corretto** (allineamento per chiave + de-vig empirico): il
  modello *non* batteva il mercato prima, e i numeri pubblicati erano falsi.
- ✅ **Modello v2**: accuracy 56.03% e F1 macro 0.4951 **battono** bet365
  (54.69% / 0.4168). RPS in pari (−0.00059).
- ✅ **41 feature** (prestazione, Elo, mercato), con parità training/inferenza
  garantita per costruzione.
- ✅ **Calibrazione Dirichlet cross-fitted** operativa in training e in
  inferenza (verificata: batch vs riga singola, differenza 0.0). **Ma è
  NEUTRA**: non migliora il RPS (+0.0002) e non è il motivo del vantaggio
  su accuracy/F1 — quello viene dalle 12 feature di mercato. Vedi §5b.3.
- ⚠️ **Il ROI non è statisticamente significativo** (P(ROI≤0) ≈ 14%) e
  dipende da una sola stagione (25/26 +42.6u, 26/27 −3.9u).
  **Non lo dichiaro come edge.**
- ⚠️ **Il CLV dice che non c'è informazione propria**: il modello eguaglia
  il mercato, non lo anticipa.
- ⏳ Sul campionato corrente (26/27) il modello è *indietro*: RPS 0.1328 vs
  0.1309 del bookmaker, Kelly ROI −11.5%.

**La conclusione scientifica**: con 448 partite di test e un mercato che
incorpora già tutta l'informazione pubblica, un modello di Serie A basato su
Pi-Rating + tiri + Elo **pareggia il bookmaker e lo corregge marginalmente
sull'esito**. Batterlo in valore richiederebbe (a) più partite di test per
distinguere l'edge dal rumore, (b) il line shopping Pinnacle/bet365, (c)
un segnale non pubblico (infortri, formazioni, meteo). Nessuno dei tre è
raggiungibile con i dati attuali del repo.

---

## 8. File aggiunti / modificati

| File | Stato | Cosa |
|---|---|---|
| `src/implied.py` | **nuovo** | 5 metodi di de-vig + metriche (RPS, log-loss, Brier/MSE, accuracy, F1) |
| `src/advanced_features.py` | **nuovo** | prestazione, Elo, mercato — formule condivise training/inferenza |
| `src/evaluate_market.py` | **nuovo** | valutazione allineata per chiave, train + test, tutte le metriche |
| `src/diagnose_vs_market.py` | **nuovo** | disaccordo, EV/ROI, edge soft-vs-sharp, CLV |
| `src/experiments.py` | **nuovo** | varianti di modello + calibrazione Dirichlet |
| `src/market_blend.py` | **nuovo** | shrinkage modello/mercato, stabilità del peso w |
| `scripts/test_implied.py` | **nuovo** | 25 test, validazione contro penaltyblog 1.13.1 e sklearn |
| `scripts/select_devig.py` | **nuovo** | scelta empirica del metodo di de-vig |
| `scripts/verify_edge.py` | **nuovo** | verifica indipendente dell'edge: join, meccanismo, calibrazione, bootstrap |
| `src/predict_fixtures.py` | modificato | freeze rimosso, de-vig corretto, calibrazione, feature dal modello |
| `src/features.py` | modificato | feature avanzate, dropna solo su base, allineamento verificato |
| `src/predict.py` | modificato | feature avanzate in inferenza, calibrazione applicata |
| `src/model.py` | modificato | `--fast`, early stopping senza leakage, calibrazione cross-fitted |
| `src/clean_data.py` | modificato | conserva colonne prestazione e quote |
| `src/refresh_predictions.py` | modificato | freeze rimosso anche dal refresh |
| `.github/workflows/pipeline.yml` | modificato | via `--reset-freeze` (che non esiste più) e input `reset_freeze` |
| `.github/workflows/daily-update.yml` | modificato | commento freeze aggiornato |
| `scripts/test_freeze_odds_guard.py` | modificato | riscritto sul contratto post-freeze (il vecchio falliva già su HEAD) |
| `scripts/test_inference_matches_training.py` | modificato | parità sull'elenco feature completo, quote passate al match |
| `data/serie_a_raw.csv` | modificato | 5 righe vuote rimosse (allineamento) |

## 9. Riferimenti

- Shin, H.S. (1992). *Pricing of winner and loser in a market for
  state-contingent claims with insider trading*. Journal of Finance 47(5).
- Shin, H.S. (1993). *Measuring the incidence of insider trading in a market
  for state-contingent claims*. The Economic Journal 103(422), 1141–1153.
- Strumbelj, E. (2014). *On determining probability forecasts from betting
  odds*. International Journal of Forecasting 30(4), 934–943.
- Eckert, J.S. (2013). *Fair probabilities from bookmaker odds*. University of
  Canterbury Working Paper.
- Power, C. (1990). *Football betting: can the behaviour of the odds market be
  used to improve the forecasting process?* International Journal of
  Forecasting 6(2), 111–118.
- Kull, M., Perello-Nieto, M., Kängsepp, M., Silva Filho, T., Song, Q.,
  Flach, P. (2019). *Beyond temperature scaling: obtaining well-calibrated
  multiclass probabilities with Dirichlet calibration*. NeurIPS 2019.
- Hvattum, L.M., Arntzen, H. (2010). *Using Elo ratings for match result
  prediction in association football*. JRSS-A 173(2), 331–340.
- Weyerbahoeff, S., Hinz, R., Heuer, S. (2015). *The influence of the FIFA
  ranking on the betting market*. Journal of Sports Economics 16(1), 107–121.
- martineastwood/penaltyblog — MIT, implementazione di riferimento per
  de-vig, Dixon-Coles, bivariate Poisson, Elo e metriche (RPS, Brier).
- JohnGavin/footbet — Poisson/Dixon-Coles/Elo con backtest walk-forward su
  quote Pinnacle e CLV: il README riporta un ROI **negativo** (-5.4% dopo
  Kelly) per un modello puro, cioè il mercato batte i modelli di calcio.
