#!/usr/bin/env python3
"""
Final Evaluation — CatBoost Serie A
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
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)

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
    try:
        raw = [1.0/float(bh), 1.0/float(bd), 1.0/float(ba)]
        total = sum(raw)
        return [p/total for p in raw]
    except Exception:
        return [0.45, 0.25, 0.30]

KELLY_FRACTION = 0.25  # Kelly ridotto al 25% per gestione del rischio

print("╔" + "═"*58 + "╗")
print("║  VALUTAZIONE FINALE — CatBoost Serie A                     ║")
print("╚" + "═"*58 + "╝")

# ── Carica test data ───────────────────────────────────────────────────
X_test_v2 = pd.read_csv(DATA_DIR / 'X_test_v2.csv')
y_test_v2 = pd.read_csv(DATA_DIR / 'y_test_v2.csv').squeeze()

print(f"\n[Data] v2 test: {X_test_v2.shape}")

# Quote dal raw (se presenti)
b365_present = False
df_test_b365 = None
try:
    raw = pd.read_csv(DATA_DIR / 'serie_a_raw.csv', low_memory=False)
    df_full = pd.read_csv(DATA_DIR / 'serie_a_features.csv', parse_dates=['Date'])
    df_full = df_full.reset_index(drop=True)
    raw = raw.reset_index(drop=True)
    if 'B365H' in raw.columns:
        df_full["B365H"] = pd.to_numeric(raw["B365H"], errors="coerce").values[:len(df_full)]
        df_full["B365D"] = pd.to_numeric(raw["B365D"], errors="coerce").values[:len(df_full)]
        df_full["B365A"] = pd.to_numeric(raw["B365A"], errors="coerce").values[:len(df_full)]
        cutoff = '2022-08-01'
        df_test_b365 = df_full[df_full['Date'] >= pd.Timestamp(cutoff)].copy().reset_index(drop=True)
        b365_present = True
        print(f"[Data] Test set con quote B365: {len(df_test_b365)} righe")
except Exception as e:
    print(f"[INFO] Quote B365 non disponibili (il raw non le include): {e}")

# ── Definizione modelli ─────────────────────────────────────────────────
models_info = {
    'v1_baseline': {
        'path': MODELS_DIR / 'catboost_seriea_v1.cbm',
        'X_test': X_test_v2,
        'y_test': y_test_v2,
        'desc': 'Baseline (Pi-Rating + FE)',
    },
}

# ── Valuta tutti i modelli ──────────────────────────────────────────────
results = {}
for key, info in models_info.items():
    print(f"\n{'─'*55}")
    print(f"Valutazione: {info['desc']}")

    model = CatBoostClassifier()
    if not info['path'].exists():
        print(f"  [SKIP] modello non trovato: {info['path']}")
        continue
    model.load_model(str(info['path']))

    X = info['X_test']
    y = info['y_test']

    probs = model.predict_proba(X)
    preds = model.predict(X).flatten()

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
        'desc': info['desc'],
    }

if not results:
    print("\n⚠️ Nessun modello da valutare. Esegui prima src/model.py per allenare catboost_seriea_v1.cbm")
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
        best_k = -1.0
        best_pick = None
        for c in range(3):
            if pm[c] > pb[c]:
                b = odds[c] - 1.0
                q = 1.0 - pm[c]
                k = (b * pm[c] - q) / b
                k = max(0.0, k * KELLY_FRACTION)
                if k > best_k:
                    best_k = k
                    best_pick = c
        if best_pick is not None:
            stake = bankroll * best_k
            stakes_total += stake
            outcome = y_valid[vi]
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
        "fraction": KELLY_FRACTION,
    }
    print(f"\n{'═'*55}")
    print(f"  KELLY SIMULATION (fraction={KELLY_FRACTION}, bankroll €1000)")
    print(f"{'═'*55}")
    print(f"  Scommesse          : {n_bets}")
    print(f"  Bankroll finale    : €{bankroll:.2f}")
    print(f"  Profitto / ROI     : €{profit:.2f} ({roi:+.1f}%)")
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(history_bank)), history_bank, color='#22c55e', lw=2)
        ax.axhline(1000.0, color='#9aa3af', ls='--', lw=1, label='Bankroll iniziale')
        ax.set_title(f'Kelly Simulation (25%) — Bankroll €1000 → €{bankroll:.0f}')
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
    sv_all = best['model'].get_feature_importance(
        data=best['X_test'], type='ShapValues')
    # sv_all: (n_samples, n_features+1, n_classes)
    shap_vals = sv_all[:, :-1, :]          # (n_samples, n_features, n_classes)
    feature_names = list(best['X_test'].columns)

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

with open(RESULTS_DIR / 'final_report.json', 'w') as f:
    json.dump(final_metrics, f, indent=2, ensure_ascii=False)
print("\n  Salvato: models/final_report.json")

# ── Report Markdown ─────────────────────────────────────────────────────
report_md = f"""# 🏆 Report Finale — CatBoost Serie A Match Prediction

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
> quasi sempre più conservativi. Kelly ridotto al {int(KELLY_FRACTION*100)}%.

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

with open(RESULTS_DIR / 'REPORT_FINALE.md', 'w', encoding='utf-8') as f:
    f.write(report_md)
print("  Salvato: models/REPORT_FINALE.md")
