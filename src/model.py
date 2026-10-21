# src/model.py
"""
Fase 4 — Modello CatBoost per predizione risultati (un campionato alla volta).
Contiene: training baseline, hyperparameter tuning, valutazione.
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
from catboost import CatBoostClassifier, cv, Pool
from sklearn.metrics import classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leagues import get_league  # noqa: E402
from experiments import (  # noqa: E402
    apply_dirichlet_calibration,
    fit_dirichlet_calibration,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS_BASELINE = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A']


def load_data(league):
    """Carica i dati splittati dalla Fase 5 (features.py -> *_v2.csv).

    I path passano da League.split_path: due campionati non possono leggere
    lo stesso split.
    """
    X_train = pd.read_csv(league.split_path('X_train_v2'))
    X_test  = pd.read_csv(league.split_path('X_test_v2'))
    y_train = pd.read_csv(league.split_path('y_train_v2')).squeeze()
    y_test  = pd.read_csv(league.split_path('y_test_v2')).squeeze()
    return X_train, X_test, y_train, y_test


def ranked_probability_score(y_true, y_proba):
    """RPS: più basso è meglio. Bookmaker benchmark: ~0.202"""
    n_classes = y_proba.shape[1]
    rps_list = []
    for true, prob in zip(y_true, y_proba):
        cum_true = np.cumsum(np.eye(n_classes)[true])
        cum_pred = np.cumsum(prob)
        rps_list.append(np.mean((cum_pred - cum_true) ** 2))
    return np.mean(rps_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.1 Training Baseline
# ═══════════════════════════════════════════════════════════════════════════════
def train_baseline(X_train, X_test, y_train, y_test):
    """Addestra il modello baseline CatBoost."""
    print("=" * 55)
    print("4.1 — Training Baseline")
    print("=" * 55)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=100
    )

    # L'early stopping NON deve guardare il test set: scegliere il numero
    # di iterazioni sulle prestazioni sul test e' leakage (i numeri riportati
    # diventano ottimisticamente falsati). Uso una fetta temporale del
    # training. Il modello finale (train_final_model) non usa eval_set.
    n = len(y_train)
    n_es = max(int(n * 0.15), 20)
    if n - n_es >= 50:
        model.fit(
            X_train.iloc[:n - n_es], list(y_train[:n - n_es]),
            eval_set=(X_train.iloc[n - n_es:], list(y_train[n - n_es:])),
            early_stopping_rounds=50,
        )
    else:
        model.fit(X_train, list(y_train))

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4.2 Valutazione
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, label="Baseline"):
    """Valuta il modello e stampa metriche."""
    print(f"\n{'='*55}")
    print(f"4.2 — Valutazione {label}")
    print(f"{'='*55}")

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test).flatten()

    rps = ranked_probability_score(y_test.values, probs)
    acc = (preds == y_test.values).mean()

    print(f"Accuracy : {acc:.4f}   → target > 0.52")
    print(f"RPS      : {rps:.4f}   → target < 0.202")
    print(f"Batti bookmaker? {'✅ SÌ' if rps < 0.202 else '❌ NO (target: < 0.202)'}")

    print(f"\nClassification Report:")
    print(classification_report(
        y_test, preds,
        target_names=['Home Win', 'Draw', 'Away Win'],
        digits=4
    ))

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=['Home Win', 'Draw', 'Away Win'],
        yticklabels=['Home Win', 'Draw', 'Away Win'],
        cmap='Blues', ax=ax
    )
    ax.set_title(f'Confusion Matrix — CatBoost {label}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix salvata in models/confusion_matrix.png")

    return {"accuracy": acc, "rps": rps, "probs": probs, "preds": preds}


# ═══════════════════════════════════════════════════════════════════════════════
# 4.3 Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════
def hyperparameter_tuning(X_train, y_train):
    """Cross-validation temporale per trovare i migliori iperparametri."""
    print(f"\n{'='*55}")
    print("4.3 — Hyperparameter Tuning (CV temporale)")
    print(f"{'='*55}")

    train_pool = Pool(X_train, y_train)

    param_grid = [
        {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6},
        {'iterations': 1000, 'learning_rate': 0.03, 'depth': 8},
        {'iterations': 1000, 'learning_rate': 0.05, 'depth': 6},
        {'iterations': 1000, 'learning_rate': 0.05, 'depth': 8},
        {'iterations': 1500, 'learning_rate': 0.02, 'depth': 6},
    ]

    best_score = 0
    best_params = None
    all_results = []

    for params in param_grid:
        full_params = {
            **params,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': 42
        }
        results = cv(
            pool=train_pool,
            params=full_params,
            fold_count=5,
            shuffle=False,
            stratified=False,
            verbose=False
        )
        score = results['test-Accuracy-mean'].max()
        std  = results['test-Accuracy-std'].max()
        all_results.append({**params, 'cv_accuracy': round(score, 4), 'cv_std': round(std, 4)})
        print(f"  Params: {params} → CV Accuracy: {score:.4f} (±{std:.4f})")

        if score > best_score:
            best_score = score
            best_params = params

    print(f"\nMigliori parametri: {best_params}")
    print(f"CV Accuracy: {best_score:.4f}")

    return best_params, all_results


def train_final_model(X_train, y_train, best_params):
    """Riaddestra il modello finale con i migliori parametri su tutto il training set."""
    print(f"\nTraining finale con best params...")
    final_model = CatBoostClassifier(
        **best_params,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=200
    )
    final_model.fit(X_train, y_train)
    return final_model


# ═══════════════════════════════════════════════════════════════════════════════
# 4.5 Salvataggio
# ═══════════════════════════════════════════════════════════════════════════════
def save_model_and_metrics(model, metrics, feature_cols, league, label="v1"):
    """Salva modello e metriche per una lega.

    Il path viene da League.model_path, cosi' due leghe non possono
    sovrascrivere lo stesso .cbm. `label` resta per varianti (v1, v2...).
    """
    model_path = (league.model_path if label == "v1"
                  else MODELS_DIR / f"{league.model_name}_{label}.cbm")
    model.save_model(str(model_path))
    print(f"\nModello salvato in {model_path}")

    model_check = CatBoostClassifier()
    model_check.load_model(str(model_path))

    metrics_path = league.report_path(f'metrics_{label}')
    serializable = {
        k: (round(v, 6) if isinstance(v, float) else v)
        for k, v in metrics.items()
        if k not in ('probs', 'preds')
    }
    serializable['features'] = feature_cols
    serializable['league'] = league.key
    with open(metrics_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Metriche salvate in {metrics_path}")

    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main(league, fast: bool = False):
    print("╔" + "═" * 53 + "╗")
    title = f"FASE 4 — CatBoost Model Training & Evaluation ({league.name})"
    print("║" + title.ljust(54)[:53] + "║")
    print("╚" + "═" * 53 + "╝")

    X_train, X_test, y_train, y_test = load_data(league)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Feature ({len(X_train.columns)}): {list(X_train.columns)}")

    MODEL_PATH = league.model_path

    if len(X_train) == 0:
        print("\nAVVERTENZA: X_train vuoto (dataset parziale < cutoff).")
        print("Caricamento modello pre-addestrato esistente...")

        if not MODEL_PATH.exists():
            print(f"ERRORE: modello {MODEL_PATH} non trovato e nessun training possibile.")
            print("Impossibile procedere senza dati storici o modello pre-addestrato.")
            sys.exit(1)

        final_model = CatBoostClassifier()
        final_model.load_model(str(MODEL_PATH))
        print(f"Modello caricato da {MODEL_PATH}")

        # Valutazione sul test set disponibile
        if len(X_test) > 0:
            final_metrics = evaluate_model(final_model, X_test, y_test, "Loaded")
        else:
            print("Nessun dato di test disponibile per la valutazione.")
            final_metrics = {"accuracy": 0.0, "rps": 0.0}

        baseline_metrics = final_metrics
        best_params = {}
        cv_results = []
    else:
        baseline_model = train_baseline(X_train, X_test, y_train, y_test)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline")

        if fast:
            # Modalita' rapida: salta la griglia di CV temporale (5 fold x 5
            # configurazioni = 25 training). Usata per le misure LOCALI e per
            # gli esperimenti: la pipeline CI normale mantiene la griglia.
            best_params = {'iterations': 1000, 'learning_rate': 0.03, 'depth': 6}
            cv_results = [{"params": best_params, "note": "griglia CV saltata (--fast)"}]
            print(f"  [fast] parametri fissi: {best_params}")
        else:
            best_params, cv_results = hyperparameter_tuning(X_train, y_train)
        final_model = train_final_model(X_train, y_train, best_params)

        final_metrics = evaluate_model(final_model, X_test, y_test, "Tuned")

        # ── Calibrazione Dirichlet (Kull et al., NeurIPS 2019) ────────────
        # Su 3 classi la calibrazione e' 3 parametri + 1 temperatura, adattati
        # su un set di VALIDAZIONE.
        #
        # ATTENZIONE (bug corretto): la validazione NON puo' essere una
        # parte del training usata anche per addestrare il modello finale.
        # Il modello finale e' allenato su TUTTO il training, quindi le sue
        # previsioni sulle righe di validazione sono IN-SAMPLE (gonfiate) e
        # la calibrazione che ne deriva e' sbagliata: sul test set il RPS
        # pegliorava di 0.02 e poi peggiorava di 0.02 applicata al modello
        # vero. Misurato: con validazione in-sample il RPS test andava da
        # 0.13019 a 0.15068.
        #
        # Soluzione: calibrazione CROSS-FITTED. Si allena un modello
        # ausiliario solo sulla seconda meta' del training e si usano le sue
        # previsioni sulla prima meta' (out-of-sample) per adattare alpha e
        # w. La calibrazione e' cosi' stimata su previsioni che il modello
        # non ha mai visto, e si applica al modello finale.
        n_val = max(len(y_train) // 2, 10)
        calib_model = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=6,
            loss_function="MultiClass", eval_metric="Accuracy",
            random_seed=42, verbose=False,
        )
        calib_model.fit(X_train.iloc[n_val:], list(y_train[n_val:]))
        probs_val = calib_model.predict_proba(X_train.iloc[:n_val])
        alpha, w = fit_dirichlet_calibration(list(y_train[:n_val]), probs_val)
        final_metrics["calibration_alpha"] = [float(a) for a in alpha]
        final_metrics["calibration_w"] = float(w)
        final_metrics["calibration_n_val"] = int(n_val)
        print(f"\n  Calibrazione Dirichlet: alpha={np.round(alpha, 4).tolist()} w={w:.4f} "
              f"(validation = prime {n_val} righe del training)")

        # La calibrazione va salvata SEPARATAMENTE dal .cbm: e' una funzione
        # post-processing delle probabilita', non parte del modello CatBoost.
        # predict_fixtures e predict_match la caricano da qui.
        cal_path = MODELS_DIR / f"calibration_{league.key}.json"
        with open(cal_path, "w") as f:
            json.dump({"alpha": [float(a) for a in alpha], "w": float(w),
                       "n_val": int(n_val), "method": "dirichlet"},
                      f, indent=2)
        print(f"  Calibrazione salvata in {cal_path}")

        save_model_and_metrics(final_model, final_metrics,
                               list(X_train.columns), league, "v1")

    print(f"\n{'═'*55}")
    print(f"  FASE 4 COMPLETATA — {league.name}")
    print(f"  Modello: {league.model_name}")
    if 'baseline_metrics' in dir():
        print(f"  Baseline → Accuracy: {baseline_metrics['accuracy']:.4f}, RPS: {baseline_metrics['rps']:.4f}")
    if 'final_metrics' in dir():
        print(f"  Tuned    → Accuracy: {final_metrics['accuracy']:.4f}, RPS: {final_metrics['rps']:.4f}")
    print(f"{'═'*55}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Training CatBoost")
    ap.add_argument("--league", default=None, help="chiave lega (default: seriea)")
    ap.add_argument("--fast", action="store_true",
                    help="salta la griglia di hyperparameter tuning (CV 5-fold): "
                         "per misure locali ed esperimenti")
    args = ap.parse_args()
    main(get_league(args.league), fast=args.fast)
