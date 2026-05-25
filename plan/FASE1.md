## FASE 1 — Setup Ambiente (Giorno 1) — Windows

---

### 1. Verifica versione Python installata

Apri il terminale integrato in Antigravity con `Ctrl+`` e digita:

```powershell
python --version
```

**Cosa può succedere:**

| Risultato | Cosa fare |
|---|---|
| `Python 3.10.x` o `3.11.x` | ✅ Perfetto, vai al passo 2 |
| `Python 3.12.x` o superiore | ⚠️ Devi installare 3.11 |
| `Python 3.9.x` o inferiore | ⚠️ Devi installare 3.11 |
| Errore / comando non trovato | ⚠️ Python non è installato |

---

### 2. (Solo se necessario) Installa Python 3.11

Scarica l'installer da:
```
https://www.python.org/downloads/release/python-3119/
```

Nella schermata di installazione:
- ✅ Spunta **"Add Python to PATH"** (fondamentale)
- Scegli **"Customize installation"**
- ✅ Spunta **"Install for all users"**
- ✅ Spunta **"Add Python to environment variables"**

Dopo l'installazione riavvia Antigravity e verifica di nuovo:
```powershell
python --version   # → Python 3.11.9
```

---

### 3. Seleziona l'interprete in Antigravity

```
Ctrl+Shift+P → digita "Select Python Interpreter" → scegli Python 3.11.x
```

Se non compare nella lista clicca **"Enter interpreter path"** e punta a:
```
C:\Users\<TuoNome>\AppData\Local\Programs\Python\Python311\python.exe
```

---

### 4. Crea il virtual environment

```powershell
# Naviga dove vuoi creare il progetto, es:
cd C:\Users\<TuoNome>\Documents

# Crea la cartella progetto ed entra
mkdir football-prediction
cd football-prediction

# Crea il virtual environment
python -m venv .venv

# Attivalo
.venv\Scripts\activate
```

Il prompt deve cambiare e mostrare `(.venv)` all'inizio — conferma che è attivo.

Se ricevi errore di execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# poi riprova .venv\Scripts\activate
```

---

### 5. Struttura cartelle del progetto

```powershell
# Crea tutte le cartelle
mkdir data, notebooks, src, models, results

# Crea i file Python vuoti in src/
New-Item src\ratings.py, src\features.py, src\model.py, src\evaluate.py -ItemType File
```

Risultato atteso:
```
football-prediction/
├── data/
├── notebooks/
├── src/
│   ├── ratings.py
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
├── models/
├── results/
└── .venv/
```

---

### 6. Installa tutte le dipendenze

```powershell
# Aggiorna pip prima
python -m pip install --upgrade pip

# Installa tutto lo stack
pip install catboost penaltyblog soccerdata pandas numpy scikit-learn matplotlib seaborn jupyter
```

> Il download richiede circa 5 minuti. `penaltyblog` installerà Cython automaticamente — è normale.

---

### 7. Verifica installazione

```powershell
python -c "import catboost; print('catboost OK')"
python -c "import penaltyblog; print('penaltyblog OK')"
python -c "import soccerdata; print('soccerdata OK')"
python -c "import pandas; print('pandas OK')"
```

Tutti e 4 devono stampare OK senza errori.

---

### Checklist fine Giorno 1 ✅

- [ ] `python --version` mostra 3.10.x o 3.11.x
- [ ] Interprete 3.11 selezionato in Antigravity
- [ ] Virtual environment `.venv` creato e attivo (vedi `(.venv)` nel prompt)
- [ ] Struttura cartelle creata
- [ ] Tutti i pacchetti installati senza errori
- [ ] I 4 check `import` stampano OK

