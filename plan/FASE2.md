Sei un data engineer esperto di Python e football analytics.

Il tuo compito è costruire la pipeline di acquisizione dati per un 
sistema di predizione calcistica basato su CatBoost e Pi-Rating.

CONTESTO:
- Progetto: CatBoost/
- Struttura già creata con cartelle: data/, src/, models/, results/
- Python 3.10 o 3.11 installato
- Librerie disponibili: pandas, numpy

TASK DA ESEGUIRE IN ORDINE:

1. Crea il file src/download_data.py che:
   - Scarica le stagioni Serie A 2010–2024 da football-data.co.uk
   - URL pattern: https://www.football-data.co.uk/mmz4281/{stagione}/I1.csv
   - Stagioni: 1011, 1112, 1213, 1314, 1415, 1516, 1617, 1718, 
     1819, 1920, 2021, 2122, 2223, 2324
   - Gestisci errori per singola stagione con try/except (non bloccare 
     tutto se una stagione fallisce)
   - Aggiunge colonna 'season' a ogni DataFrame
   - Salva output in data/serie_a_raw.csv

2. Crea il file src/clean_data.py che:
   - Legge data/serie_a_raw.csv
   - Mantiene solo le colonne: Date, HomeTeam, AwayTeam, FTHG, FTAG, 
     FTR, season
   - Converte Date con dayfirst=True, gestisce errori con errors='coerce'
   - Ordina per data in modo STRETTAMENTE CRESCENTE (critico per i 
     pi-rating nella fase successiva)
   - Filtra FTR sui soli valori validi: H, D, A
   - Casta FTHG e FTAG come int
   - Salva in data/serie_a_clean.csv
   - Stampa: numero partite, range date, distribuzione FTR

3. Aggiungi alla fine di clean_data.py un blocco di assert che verifica:
   - df['Date'].is_monotonic_increasing → True
   - df['FTR'] contiene solo H/D/A
   - Nessun NaN in colonne critiche
   - Se tutti i check passano stampa: "✓ Dataset pronto per Fase 3"

4. Esegui i due script in sequenza e mostrami l'output completo.

VINCOLI:
- Non usare random shuffle in nessun punto
- L'ordinamento temporale è NON NEGOZIABILE
- Usa on_bad_lines='skip' nel read_csv del download
- Il risultato atteso è ~4.800–5.200 partite totali

OUTPUT ATTESO FINALE:
- data/serie_a_raw.csv
- data/serie_a_clean.csv
- Stampa di conferma con statistiche del dataset