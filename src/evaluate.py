#!/usr/bin/env python3
"""
Final Evaluation — CatBoost (un campionato alla volta)
Confronta i modelli, seleziona il migliore e genera il report.
Inclusa: Kelly betting simulation (FASE6 §6) su test set.
"""
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league  # noqa: E402
from model import compute_market_baseline  # noqa: E402
from kelly_dynamic import (  # noqa: E402
    KELLY_DYNAMIC_CONFIG,
    calculate_dynamic_kelly_stake,
    dynamic_kelly_breakdown,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)

# evaluate.py e' uno script lineare: tutto il corpo gira all'import, quindi
# argparse qui non serve e non puo' precedere gli usi. La lega si risolve da
# un argomento di riga o dalla variabile d'ambiente LEAGUE, che get_league()
# gia' legge. In CI: LEAGUE=eredivisie python src/evaluate.py
LEAGUE = get_league(None)

CLASS_SHORT  = ['H', 'D', 'A']
LABEL_MAP    = {'H': 0, 'D': 1, 'A': 2}

def ranked_probability_score(y_true, y_proba):
    n_classes = y_proba.shape[1]
    rps_list = []
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n_classes)[true])
        cum_pred = np.cumsum(prob)
        rps_list.append(np.mean((cum_pred - cum_true) ** 2))
    return np.mean(rps_list)

def odds_to_prob(bh, bd, ba):
    """Probabilita' implicite SENZA margine (de-vig).

    BUG CORRETTO: qui c'era la normalizzazione proporzionale, che e'
    statisticamente sbagliata (il bookmaker carica un margine relativo
    maggiore sui longshot) e ha il log-loss peggiore di tutti i metodi
    misurati. Il metodo 'additive' e' quello scelto empiricamente su 6135
    partite Serie A (scripts/select_devig.py) e validato a 1e-6 contro
    penaltyblog 1.13.1. Con il metodo vecchio il benchmark del bookmaker
    era SOTTOSTIMATO (log-loss 0.96105 contro 0.95982).
    """
    try:
        from implied import DE_VIG_METHODS
        p = DE_VIG_METHODS["additive"]((float(bh), float(bd), float(ba)))
        return p if p is not None else [0.45, 0.25, 0.30]
    except Exception:
        return [0.45, 0.25, 0.30]

# ── Kelly Dinamico ────────────────────────────────────────────────────────
# Sizing unico della pipeline: src/kelly_dynamic.py. Nessuna frazione fissa.
KELLY_MODE = "dinamico"

print("╔" + "═" * 58 + "╗")
print(f"║  VALUTAZIONE FINALE — CatBoost {LEAGUE.name}"
      .ljust(59) + "║")
print("╚" + "═" * 58 + "╝")

# ── Carica test data ───────────────────────────────────────────────────
X_test_v2 = pd.read_csv(LEAGUE.split_path('X_test_v2'))
y_test_v2 = pd.read_csv(LEAGUE.split_path('y_test_v2')).squeeze()

print(f"\n[Data] v2 test: {X_test_v2.shape}")

# Quote dal raw (se presenti)
b365_present = False
df_test_b365 = None
try:
    raw = pd.read_csv(LEAGUE.raw_path, low_memory=False)
    if 'B365H' in raw.columns:
        # BUG CORRETTO (grave): le quote erano abbinate alle righe del
        # dataset per POSIZIONE (.values[:len(df_full)]). Ma il raw
        # concatenato da download_data.py contiene righe vuote che
        # clean_data.py scarta: dal pezzo in cui si verifica la riga, ogni
        # quote appartiene a una partita DIVERSA (misurato: 4997 righe su
        # 6140 con quote sbagliate). Il cutoff era inoltre hardcoded a
        # 2022-08-01, quindi le righe abbinate erano di un periodo che non
        # coincideva con il test set.
        #
        # Risultato del bug: il report finale dichiarava
        # rps_bookmaker=0.178 e accuracy_bookmaker=0.377, cioe' un
        # bookmaker con le quote mescolate a caso, e 'beats_bookmaker_rps:
        # True' perche' il modello batteva quel benchmark fasullo.
        #
        # Ora si abbina per CHIAVE (data, casa, fuori) con lo stesso
        # helper di src/evaluate_market.py e lo stesso cutoff del training
        # (2025-08-01), cosi' le quote sono delle STESSE partite del test
        # set, nello stesso ordine.
        from evaluate_market import _test_keys, load_odds_by_key
        cutoff = '2025-08-01'
        keys = _test_keys(LEAGUE, cutoff)
        odds_by_key = load_odds_by_key(Path(LEAGUE.raw_path))
        rows = []
        n_with = 0
        for k in keys:
            o = odds_by_key.get(k, {}).get('b365_open')
            if o and all(v is not None and float(v) > 1.0 for v in o):
                rows.append((float(o[0]), float(o[1]), float(o[2])))
                n_with += 1
            else:
                rows.append((np.nan, np.nan, np.nan))
        df_test_b365 = pd.DataFrame(
            rows, columns=['B365H', 'B365D', 'B365A'],
            index=range(len(rows)))
        b365_present = True
        print(f"[Data] Test set allineato per CHIAVE: {len(df_test_b365)} righe "
              f"({n_with} con quote B365, cutoff {cutoff})")
except Exception as e:
    print(f"[INFO] Quote B365 non disponibili (il raw non le include): {e}")

# ── Definizione modelli ─────────────────────────────────────────────────
models_info = {
    'v1_baseline': {
        'path': LEAGUE.model_path,
        'X_test': X_test_v2,
        'y_test': y_test_v2,
        'desc': f'Baseline (Pi-Rating + FE) — {LEAGUE.name}',
    },
}

# ── Valuta tutti i modelli ──────────────────────────────────────────────
pure_path = MODELS_DIR / f"{LEAGUE.model_name}_pure.cbm"
pure_model = None
if pure_path.exists():
    pure_model = CatBoostClassifier()
    pure_model.load_model(str(pure_path))

results = {}
for key, info in models_info.items():
    print(f"\n{'─'*55}")
    print(f"Valutazione: {info['desc']}")

    model = CatBoostClassifier()
    if not info['path'].exists():
        print(f"  [SKIP] modello non trovato: {info['path']}")
        continue
    model.load_model(str(info['path']))

    _feat = list(model.feature_names_)
    X = info['X_test'][_feat]
    y = info['y_test']

    if any(c.startswith("mkt_") for c in _feat):
        eval_pool = Pool(X)
        probs = model.predict_proba(X)
        preds = model.predict(X).flatten()
    else:
        base_test, _ = compute_market_baseline(
            info['X_test'], pure_model=pure_model, fund_cols=_feat
        )
        eval_pool = Pool(X, baseline=base_test)
        probs = model.predict_proba(eval_pool)
        preds = np.argmax(probs, axis=1)

    acc = (preds == y.values).mean()
    rps = ranked_probability_score(y.values, probs)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  RPS      : {rps:.4f}")

    results[key] = {
        'model': model,
        'probs': probs,
        'preds': preds,
        'accuracy': acc,
        'rps': rps,
        'X_test': X,
        'y_test': y,
        'eval_pool': eval_pool,
        'desc': info['desc'],
    }

if not results:
    print(f"\nNessun modello da valutare. Esegui prima src/model.py --league {LEAGUE.key}")
    raise SystemExit(0)

# ── Trova il migliore ───────────────────────────────────────────────────
best_key = min(results, key=lambda k: results[k]['rps'])
best = results[best_key]

print(f"\n{'═'*55}")
print(f"🏆 MIGLIOR MODELLO: {best['desc']} [{best_key}]")
print(f"   RPS      : {best['rps']:.4f}")
print(f"   Accuracy : {best['accuracy']:.4f}")
print(f"{'═'*55}")

print(f"\n{'Modello':<35} {'RPS':>8}  {'Accuracy':>10}")
print(f"{'─'*55}")
for key, r in results.items():
    marker = " ◄ BEST" if key == best_key else ""
    print(f"  {r['desc']:<33} {r['rps']:>8.4f}  {r['accuracy']:>9.2%}{marker}")

# ── Metriche bookmaker (opzionale) ──────────────────────────────────────
rps_bm, acc_bm = None, None
bm_probs_list = []
valid_idx = []
y_best = best['y_test']
probs_best = best['probs']
if b365_present and df_test_b365 is not None:
    y_best = best['y_test']
    probs_best = best['probs']
    n_test = len(y_best)

    bm_probs_list = []
    valid_idx = []
    for i in range(min(n_test, len(df_test_b365))):
        row = df_test_b365.iloc[i]
        bh = row.get('B365H'); bd = row.get('B365D'); ba = row.get('B365A')
        if pd.notna(bh) and pd.notna(bd) and pd.notna(ba) and float(bh) > 0 and float(bd) > 0 and float(ba) > 0:
            bm_probs_list.append(odds_to_prob(bh, bd, ba))
            valid_idx.append(i)

    if bm_probs_list:
        bm_probs_valid = np.array(bm_probs_list)
        y_valid = y_best.values[valid_idx]
        rps_bm = ranked_probability_score(y_valid, bm_probs_valid)
        acc_bm = (np.argmax(bm_probs_valid, axis=1) == y_valid).mean()

        print(f"\n{'═'*55}")
        print(f"  CONFRONTO BOOKMAKER (Bet365)")
        print(f"{'═'*55}")
        print(f"  {'Modello':<30} {'RPS':>8}  {'Accuracy':>10}")
        print(f"  {'─'*50}")
        print(f"  {best['desc']:<30} {best['rps']:>8.4f}  {best['accuracy']:>9.2%}")
        print(f"  {'Bookmaker Bet365':<30} {rps_bm:>8.4f}  {acc_bm:>9.2%}")
        print(f"\n  Delta RPS  : {rps_bm - best['rps']:+.4f}  ({'✅ modello vince' if best['rps'] < rps_bm else '⚠️ BM vince'})")

# ── Kelly betting simulation (FASE6 §6) ────────────────────────────────
# Per ogni partita del test set in cui il modello stima p_model > p_bookmaker
# (edge positivo), punta con Kelly ridotto al 25% sulla giocata con edge migliore.
# Simulazione retrospettiva su test set (didattica, ottimistica per il futuro).
kelly_metrics = None
if b365_present and df_test_b365 is not None and len(bm_probs_list) > 0:
    probs_best = best['probs']
    bankroll = 1000.0
    history_bank = [1000.0]
    n_bets = 0
    stakes_total = 0.0
    capped_flags = []
    bm_probs_valid = np.array(bm_probs_list)
    y_valid = y_best.values[valid_idx]
    for i, vi in enumerate(valid_idx):
        row = df_test_b365.iloc[vi]
        try:
            odds = np.array([float(row['B365H']), float(row['B365D']), float(row['B365A'])])
        except (TypeError, ValueError):
            continue
        if np.any(odds <= 1.0):
            continue
        pm = probs_best[vi]
        pb = bm_probs_valid[i]
        best_k = 0.0
        best_pick = None
        for c in range(3):
            if pm[c] > pb[c]:
                # c_dynamic per scaglioni di quota + hard cap: la quota decide
                # quanto Kelly pieno si prende, non una frazione unica.
                k = calculate_dynamic_kelly_stake(pm[c], odds[c])
                if k > 0.0 and k > best_k:
                    best_k = k
                    best_pick = c
        if best_pick is not None:
            capped_flags.append(dynamic_kelly_breakdown(
                pm[best_pick], odds[best_pick])["capped"])
            stake = bankroll * best_k
            stakes_total += stake
            outcome = y_best.values[vi]
            win = (outcome == best_pick)
            bankroll += stake * (odds[best_pick] - 1.0) if win else -stake
            n_bets += 1
            history_bank.append(bankroll)

    profit = bankroll - 1000.0
    roi = (profit / 1000.0) * 100.0
    kelly_metrics = {
        "bankroll_iniziale": 1000.0,
        "bankroll_finale": round(float(bankroll), 2),
        "profitto": round(float(profit), 2),
        "roi_pct": round(float(roi), 2),
        "scommesse": n_bets,
        "stake_totale_pct": round(float(stakes_total / 1000.0) * 100, 1),
        "mode": KELLY_MODE,
        "odds_thresholds": list(KELLY_DYNAMIC_CONFIG["odds_thresholds"]),
        "c_dynamic": list(KELLY_DYNAMIC_CONFIG["c_dynamic"]),
        "cap_odds_threshold": KELLY_DYNAMIC_CONFIG["cap_odds_threshold"],
        "max_caps_pct": [round(c * 100, 2) for c in KELLY_DYNAMIC_CONFIG["max_caps"]],
        "n_troncate_al_cap": sum(1 for x in capped_flags if x),
    }
    print(f"\n{'═'*55}")
    print(f"  KELLY SIMULATION (dinamico, bankroll €1000)")
    print(f"{'═'*55}")
    print(f"  Scommesse          : {n_bets}")
    print(f"  Bankroll finale    : €{bankroll:.2f}")
    print(f"  Profitto / ROI     : €{profit:.2f} ({roi:+.1f}%)")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(history_bank)), history_bank, color='#22c55e', lw=2)
        ax.axhline(1000.0, color='#9aa3af', ls='--', lw=1, label='Bankroll iniziale')
        ax.set_title(f'Kelly dinamico — Bankroll €1000 → €{bankroll:.0f}')
        ax.set_xlabel('Scommesse piazzate')
        ax.set_ylabel('Bankroll (€)')
        ax.legend()
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'kelly_simulation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Salvato: models/kelly_simulation.png")
    except Exception as e:
        print(f"  [warn] grafico Kelly non salvato: {e}")

# ── Classification Report ───────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  CLASSIFICATION REPORT — {best['desc']}")
print(f"{'═'*55}")
cr = classification_report(best['y_test'], best['preds'], target_names=CLASS_SHORT, digits=4)
print(cr)

# ── Confusion Matrix ────────────────────────────────────────────────────
cm = confusion_matrix(best['y_test'], best['preds'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix — {best["desc"]}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrix_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Salvato: models/confusion_matrix_final.png")

# ── SHAP Analysis per flag neopromossa ──────────────────────────────────
# Usa gli SHAP values NATIVI di CatBoost (get_feature_importance(type='ShapValues'))
# -> niente dipendenze esterne (shap/numba) e niente segfault su CI.
shap_metrics = None
try:
    print(f"\n{'═'*55}")
    print(f"  SHAP ANALYSIS — Feature Importance (test set, CatBoost nativo)")
    print(f"{'═'*55}")

    # CatBoost restituisce (n_samples, n_features+1, n_classes); l'ultima
    # colonna per classe e' il base value. Togliamola.
    eval_pool = best.get('eval_pool') if best.get('eval_pool') is not None else Pool(best['X_test'])
    sv_all = best['model'].get_feature_importance(
        data=eval_pool, type='ShapValues')
    feature_names = list(best['X_test'].columns)
    n_feat = len(feature_names)
    # CatBoost per multiclasse restituisce (n_samples, A, B) dove {A,B} =
    # {n_classes, n_feat (+1 base value)}. L'ordine degli assi non e' garantito,
    # quindi lo rendiamo robusto: identifichiamo l'asse delle feature, scartiamo
    # il base value (ultima colonna) se presente, e lo portiamo in posizione 1.
    if sv_all.shape[1] in (n_feat, n_feat + 1):
        feat_axis = 1
    elif sv_all.shape[2] in (n_feat, n_feat + 1):
        feat_axis = 2
    else:
        feat_axis = 1  # fallback
    if sv_all.shape[feat_axis] == n_feat + 1:
        sv_all = np.take(sv_all, range(n_feat), axis=feat_axis)
    shap_vals = np.moveaxis(sv_all, feat_axis, 1)   # (n_samples, n_feat, n_classes)

    # mean |SHAP| per feature, mediato sulle 3 classi
    mean_abs_shap = np.mean(np.mean(np.abs(shap_vals), axis=0), axis=1)

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)

    print(f"\n  Top 10 feature per mean |SHAP|:")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {row['feature']:20s}  {row['mean_abs_shap']:.6f}")

    for flag in ['home_promoted', 'away_promoted']:
        if flag in feature_names:
            idx = feature_names.index(flag)
            val = mean_abs_shap[idx]
            rank = int(shap_df[shap_df['feature'] == flag].index[0]) + 1
            # SHAP per classe H (indice 0) quando feature=1 vs feature=0
            sv_h = shap_vals[:, idx, 0]
            feat_vals = best['X_test'][flag].values
            shap_1 = sv_h[feat_vals == 1].mean() if (feat_vals == 1).any() else 0.0
            shap_0 = sv_h[feat_vals == 0].mean() if (feat_vals == 0).any() else 0.0
            print(f"\n  {flag}:")
            print(f"    mean |SHAP| = {val:.6f}  (rank {rank}/{len(feature_names)})")
            print(f"    mean SHAP feature=1 (classe H) = {shap_1:+.6f}")
            print(f"    mean SHAP feature=0 (classe H) = {shap_0:+.6f}")
            print(f"    delta = {shap_1 - shap_0:+.6f}")
            shap_metrics = shap_metrics or {}
            shap_metrics[flag] = {
                "mean_abs_shap": float(val),
                "rank": rank,
                "n_features_total": len(feature_names),
                "shap_feature1_classH": float(shap_1),
                "shap_feature0_classH": float(shap_0),
                "delta_classH": float(shap_1 - shap_0),
            }
        else:
            print(f"\n  {flag}: NON PRESENTE nel test set")

    # Grafico bar summary (top 15) con matplotlib nativo
    try:
        top = shap_df.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top['feature'], top['mean_abs_shap'], color='#22c55e')
        ax.set_xlabel('mean |SHAP|')
        ax.set_title('SHAP Feature Importance (CatBoost nativo) — Top 15')
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Salvato: models/shap_summary.png")
    except Exception as e:
        print(f"  [warn] grafico SHAP summary non salvato: {e}")

except Exception as e:
    print(f"  [warn] SHAP analysis fallita: {e}")

draw_recall = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
home_recall = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0
away_recall = cm[2, 2] / cm[2, :].sum() if cm[2, :].sum() > 0 else 0

# ── F1-score (macro + per classe) e recall completo ──────────────────────
from sklearn.metrics import f1_score, recall_score, precision_score

y_true_arr = best['y_test'].values
y_pred_arr = best['preds']

f1_macro = f1_score(y_true_arr, y_pred_arr, average='macro')
f1_weighted = f1_score(y_true_arr, y_pred_arr, average='weighted')
f1_per_class = f1_score(y_true_arr, y_pred_arr, average=None, labels=[0, 1, 2])
recall_per_class = recall_score(y_true_arr, y_pred_arr, average=None, labels=[0, 1, 2])
precision_per_class = precision_score(y_true_arr, y_pred_arr, average=None, labels=[0, 1, 2])

print(f"\n{'═'*55}")
print(f"  F1 / RECALL / PRECISION per classe")
print(f"{'═'*55}")
for i, name in enumerate(['H', 'D', 'A']):
    print(f"  {name}: F1={f1_per_class[i]:.4f}  Recall={recall_per_class[i]:.4f}  Precision={precision_per_class[i]:.4f}")
print(f"  F1 macro      : {f1_macro:.4f}")
print(f"  F1 weighted   : {f1_weighted:.4f}")

final_metrics = {
    "best_model": best_key,
    "best_model_desc": best['desc'],
    "accuracy": round(float(best['accuracy']), 4),
    "rps_model": round(float(best['rps']), 4),
    "f1_macro": round(float(f1_macro), 4),
    "f1_weighted": round(float(f1_weighted), 4),
    "f1_per_class": {"H": round(float(f1_per_class[0]), 4),
                     "D": round(float(f1_per_class[1]), 4),
                     "A": round(float(f1_per_class[2]), 4)},
    "precision_per_class": {"H": round(float(precision_per_class[0]), 4),
                            "D": round(float(precision_per_class[1]), 4),
                            "A": round(float(precision_per_class[2]), 4)},
    "recall_home": round(float(home_recall), 4),
    "recall_draw": round(float(draw_recall), 4),
    "recall_away": round(float(away_recall), 4),
    "recall_per_class": {"H": round(float(recall_per_class[0]), 4),
                         "D": round(float(recall_per_class[1]), 4),
                         "A": round(float(recall_per_class[2]), 4)},
    "n_test": int(len(best['y_test'])),
    "n_features": int(best['X_test'].shape[1]),
    "features": list(best['X_test'].columns),
}
if rps_bm is not None:
    final_metrics["rps_bookmaker"] = round(float(rps_bm), 4)
    final_metrics["accuracy_bookmaker"] = round(float(acc_bm), 4)
    final_metrics["beats_bookmaker_rps"] = bool(best['rps'] < rps_bm)
if kelly_metrics is not None:
    final_metrics["kelly"] = kelly_metrics
if shap_metrics is not None:
    final_metrics["shap_neopromossa"] = shap_metrics

with open(LEAGUE.report_path('final_report'), 'w') as f:
    json.dump(final_metrics, f, indent=2, ensure_ascii=False)
print("\n  Salvato: models/final_report.json")

# ── Report Markdown ─────────────────────────────────────────────────────
report_md = f"""# 🏆 Report Finale — CatBoost {LEAGUE.name} Match Prediction

**Data**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Modello migliore**: {best['desc']} (`{best_key}`)

---

## 1. Confronto Modelli

| Modello | Feature | RPS ↓ | Accuracy |
|---------|---------|-------|----------|
"""
for key, r in results.items():
    marker = " ◄ **BEST**" if key == best_key else ""
    n_feat = r['X_test'].shape[1]
    report_md += f"| {r['desc']} | {n_feat} | {r['rps']:.4f} | {r['accuracy']:.2%}{marker}|\n"

report_md += f"""
---

## 2. Metriche Modello Migliore

| Metrica | Valore | Note |
|---------|--------|------|
| **Accuracy** | {best['accuracy']:.4f} | Target > 0.52 |
| **RPS** | {best['rps']:.4f} | Più basso = meglio |
| **Recall Home Win** | {home_recall:.2%} | |
| **Recall Draw** | {draw_recall:.2%} | Classe più difficile |
| **Recall Away Win** | {away_recall:.2%} | |

"""

if rps_bm is not None:
    report_md += f"""---

## 3. Confronto con Bookmaker Bet365

| | RPS ↓ | Accuracy |
|--|-------|----------|
| **CatBoost ({best_key})** | {best['rps']:.4f} | {best['accuracy']:.2%} |
| **Bet365** | {rps_bm:.4f} | {acc_bm:.2%} |
| **Delta** | {rps_bm - best['rps']:+.4f} {'✅' if best['rps'] < rps_bm else '⚠️'} | {best['accuracy'] - acc_bm:+.4f} {'✅' if best['accuracy'] > acc_bm else '⚠️'} |

"""

if kelly_metrics is not None:
    report_md += f"""---

## 4. Kelly Betting Simulation (didattica, test set)

> ⚠️ Simulazione retrospettiva su dati storici: i risultati reali futuri saranno
> quasi sempre più conservativi. Kelly dinamico: la quota decide quanto
> Kelly pieno si prende (c_dynamic da {int(KELLY_DYNAMIC_CONFIG['c_dynamic'][0]*100)}% a
> {int(KELLY_DYNAMIC_CONFIG['c_dynamic'][-1]*100)}%), con hard cap del
> {int(KELLY_DYNAMIC_CONFIG['max_caps'][0]*100)}% / {int(KELLY_DYNAMIC_CONFIG['max_caps'][1]*100)}%.

| Metrica | Valore |
|---------|--------|
| Scommesse piazzate | {kelly_metrics['scommesse']} |
| Bankroll iniziale | €{kelly_metrics['bankroll_iniziale']:.0f} |
| Bankroll finale | €{kelly_metrics['bankroll_finale']:.2f} |
| Profitto | €{kelly_metrics['profitto']:+.2f} |
| ROI | {kelly_metrics['roi_pct']:+.1f}% |
| Stake totale | {kelly_metrics['stake_totale_pct']}% del bankroll |

![Kelly Simulation](kelly_simulation.png)

"""

if shap_metrics is not None:
    report_md += f"""---

## 4b. SHAP Analysis — Flag Neopromossa

> Mean |SHAP| misura quanto ogni feature sposta la predizione in media sul test set.
> Se ≈ 0 → il flag non porta segnale e può essere rimosso. Se `home_promoted`
> e `away_promoted` hanno valori simili → erano ridondanti, conviene unificarli.

| Flag | mean |SHAP| | Rank su {shap_metrics[list(shap_metrics)[0]]['n_features_total']} feature |
|------|-------------|------|
| home_promoted | {shap_metrics['home_promoted']['mean_abs_shap']:.6f} | {shap_metrics['home_promoted']['rank']}/{shap_metrics['home_promoted']['n_features_total']} |
| away_promoted | {shap_metrics['away_promoted']['mean_abs_shap']:.6f} | {shap_metrics['away_promoted']['rank']}/{shap_metrics['away_promoted']['n_features_total']} |

![SHAP Summary](shap_summary.png)

"""

report_md += f"""---

## 5. Classification Report

```
{cr}
```

---

## 6. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
"""

with open(RESULTS_DIR / f'REPORT_FINALE_{LEAGUE.key}.md', 'w', encoding='utf-8') as f:
    f.write(report_md)
print(f"  Salvato: models/REPORT_FINALE_{LEAGUE.key}.md")
