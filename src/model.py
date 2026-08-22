# src/model.py
"""
Fase 4 — Modello CatBoost per predizione risultati Serie A.
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

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLS_BASELINE = ['pi_home_H', 'pi_away_H', 'pi_home_A', 'pi_away_A']


def load_data():
    """Carica i dati splittati dalla Fase 3 (features.py -> *v2.csv)."""
    X_train = pd.read_csv(DATA_DIR / 'X_train_v2.csv')
    X_test  = pd.read_csv(DATA_DIR / 'X_test_v2.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train_v2.csv').squeeze()
    y_test  = pd.read_csv(DATA_DIR / 'y_test_v2.csv').squeeze()
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

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )

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
def save_model_and_metrics(model, metrics, feature_cols, label="v1"):
    """Salva modello e metriche."""
    model_path = MODELS_DIR / f'catboost_seriea_{label}.cbm'
    model.save_model(str(model_path))
    print(f"\nModello salvato in {model_path}")

    model_check = CatBoostClassifier()
    model_check.load_model(str(model_path))

    metrics_path = RESULTS_DIR / f'metrics_{label}.json'
    serializable = {
        k: (round(v, 6) if isinstance(v, float) else v)
        for k, v in metrics.items()
        if k not in ('probs', 'preds')
    }
    serializable['features'] = feature_cols
    with open(metrics_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Metriche salvate in {metrics_path}")

    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔" + "═"*53 + "╗")
    print("║  FASE 4 — CatBoost Model Training & Evaluation (Serie A)  ║")
    print("╚" + "═"*53 + "╝")

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Feature: {list(X_train.columns)}")

    baseline_model = train_baseline(X_train, X_test, y_train, y_test)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline")

    best_params, cv_results = hyperparameter_tuning(X_train, y_train)
    final_model = train_final_model(X_train, y_train, best_params)

    final_metrics = evaluate_model(final_model, X_test, y_test, "Tuned")

    save_model_and_metrics(final_model, final_metrics, list(X_train.columns), "v1")

    print(f"\n{'═'*55}")
    print(f"  FASE 4 COMPLETATA")
    print(f"  Baseline → Accuracy: {baseline_metrics['accuracy']:.4f}, RPS: {baseline_metrics['rps']:.4f}")
    print(f"  Tuned    → Accuracy: {final_metrics['accuracy']:.4f}, RPS: {final_metrics['rps']:.4f}")
    print(f"{'═'*55}")
