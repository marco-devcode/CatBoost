#!/usr/bin/env python3
"""
Final Evaluation — CatBoost Serie A
Confronta i modelli, seleziona il migliore e genera il report.
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

report_md += f"""---

## 4. Classification Report

```
{cr}
```

---

## 5. Confusion Matrix

![Confusion Matrix](confusion_matrix_final.png)

---

*Generato automaticamente da `src/evaluate.py`*
"""

with open(RESULTS_DIR / 'REPORT_FINALE.md', 'w', encoding='utf-8') as f:
    f.write(report_md)
print("  Salvato: models/REPORT_FINALE.md")
