# Serie A Dashboard — Predizioni CatBoost

Dashboard mobile-first che mostra le predizioni giornata-per-giornata della
Serie A 26/27, generate dalla pipeline ML (`src/predict_fixtures.py`).

**Come funziona**: la GitHub Action (`pipeline.yml`) gira ogni lunedì e
giovedì alle 06:00 UTC → ri-allena il modello → calcola le probabilità H/D/A
di tutte le partite del calendario → salva `webapp/public/predictions.json`
e fa push. Il push fa scattare il redeploy automatico su Vercel.

---

## Deploy su Vercel (passo-passo)

### 1. Import del progetto
1. Vai su [vercel.com/new](https://vercel.com/new)
2. **Add New… → Project** → seleziona `marco-devcode/CatBoost`
3. Nella schermata di configurazione:
   - **Framework Preset**: Next.js (rilevato automaticamente)
   - **Root Directory**: premi **Edit** e seleziona `webapp`
4. Lascia tutto il resto di default (nessuna env var necessaria: il JSON è statico)

### 2. Deploy
- Premi **Deploy**
- Al primo deploy Vercel ti dà l'URL tipo `catboost-xxxx.vercel.app`
- Da lì in poi: ogni push su `main` (incluso quelli automatici della Action) → redeploy

### 3. Aggiornamenti
- Automatici: lun + gio alle 06:00 UTC (pipeline GitHub Actions)
- Manuali: tab Actions → "Pipeline Serie A" → Run workflow

---

## Sviluppo locale

```bash
cd webapp
npm install
npm run dev        # http://localhost:3000
npm run typecheck  # verifica TypeScript
```

Il JSON `public/predictions.json` viene generato dalla pipeline; per testare
in locale senza aspettare la Action basta il placeholder già nel repo.

## Struttura

```
webapp/
├── app/
│   ├── layout.tsx        # html/body + metadata
│   ├── page.tsx          # dashboard (client component)
│   └── globals.css       # tema dark mobile-first
├── lib/types.ts          # tipi del JSON
└── public/predictions.json  # generato dalla pipeline, committato
```
