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
def train_baseline(X_train, X_test, y_train, y_test, base_train=None):
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

    n = len(y_train)
    n_es = max(int(n * 0.15), 20)
    if n - n_es >= 50:
        tr_pool = Pool(
            X_train.iloc[:n - n_es], list(y_train[:n - n_es]),
            baseline=(base_train[:n - n_es] if base_train is not None else None),
        )
        val_pool = Pool(
            X_train.iloc[n - n_es:], list(y_train[n - n_es:]),
            baseline=(base_train[n - n_es:] if base_train is not None else None),
        )
        model.fit(
            tr_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
        )
    else:
        tr_pool = Pool(X_train, list(y_train), baseline=base_train)
        model.fit(tr_pool)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4.2 Valutazione
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, label="Baseline", base_test=None):
    """Valuta il modello e stampa metriche."""
    print(f"\n{'='*55}")
    print(f"4.2 — Valutazione {label}")
    print(f"{'='*55}")

    if base_test is not None:
        probs = model.predict_proba(Pool(X_test, baseline=base_test))
        preds = np.argmax(probs, axis=1)
    else:
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


# ─────────────────────────────────────────────────────────────────────────────
# ARCHITETTURA RESIDUAL LOGIT-OFFSET (Mercato come Prior + Fondamentali)
# ─────────────────────────────────────────────────────────────────────────────
# Negli alberi entrano SOLO le 29 feature fondamentali calcistiche (zero `mkt_*`).
# Quando le quote Bet365 sono presenti, log(p_b365) viene passato come baseline
# continua al Pool CatBoost: gli alberi stimano soltanto lo scostamento residuo
# rispetto alla linea di mercato, evitando che Kelly punti sull'errore di
# quantizzazione degli split sulle quote.
SIMPLIFIED_EXCLUDE_PREFIXES = ("mkt_",)
SIMPLIFIED_EXCLUDE_SUFFIXES = ("_overround",)
RESIDUAL_PARAMS_BY_LEAGUE = {
    "seriea": {"iterations": 60, "learning_rate": 0.02, "depth": 3, "l2_leaf_reg": 30},
    "eredivisie": {"iterations": 160, "learning_rate": 0.04, "depth": 4, "l2_leaf_reg": 10},
}
DEFAULT_PARAMS = {"iterations": 80, "learning_rate": 0.02, "depth": 3, "l2_leaf_reg": 30}
PURE_PARAMS = {"iterations": 250, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 15}


def simplified_feature_set(columns):
    """Le 29 feature fondamentali calcistiche (esclude tutte le colonne `mkt_*`)."""
    return [c for c in columns
            if not c.startswith(SIMPLIFIED_EXCLUDE_PREFIXES)
            and not c.endswith(SIMPLIFIED_EXCLUDE_SUFFIXES)]


def compute_market_baseline(
    df_with_mkt: pd.DataFrame,
    pure_model=None,
    fund_cols=None,
    use_sharp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Estrae log(p_prior) come baseline continua.

    - Con use_sharp=False (usato nel fit sugli alberi): usa `mkt_b365_p*` in modo
      che gli alberi apprendano i bias specifici rispetto alla linea Bet365.
    - Con use_sharp=True (usato in valutazione/inferenza): usa il consenso sharp
      0.65 * Pinnacle + 0.35 * Avg (con fallback a Pinnacle, media Avg+B365,
      solo B365 o modello puro se mancano le quote).
    """
    n = len(df_with_mkt)

    def _get_probs(cols):
        if not all(c in df_with_mkt.columns for c in cols):
            return np.full((n, 3), np.nan), np.zeros(n, dtype=bool)
        arr = df_with_mkt[list(cols)].to_numpy(dtype=float)
        valid = np.all(np.isfinite(arr) & (arr > 0.0), axis=1)
        return arr, valid

    p_b365, has_b365 = _get_probs(("mkt_b365_pH", "mkt_b365_pD", "mkt_b365_pA"))

    if not use_sharp:
        p_raw = p_b365
        has_mkt = has_b365
    else:
        p_pin, has_pin = _get_probs(("mkt_pinnacle_pH", "mkt_pinnacle_pD", "mkt_pinnacle_pA"))
        p_avg, has_avg = _get_probs(("mkt_avg_pH", "mkt_avg_pD", "mkt_avg_pA"))
        p_raw = np.where(has_b365[:, None], p_b365, 1.0 / 3.0)
        for i in range(n):
            if has_pin[i] and has_avg[i]:
                p_raw[i] = 0.65 * p_pin[i] + 0.35 * p_avg[i]
            elif has_pin[i]:
                p_raw[i] = p_pin[i]
            elif has_avg[i] and has_b365[i]:
                p_raw[i] = 0.50 * p_avg[i] + 0.50 * p_b365[i]
        has_mkt = has_b365 | has_pin | has_avg

    if pure_model is not None and fund_cols is not None and (~has_mkt).any():
        p_pure = pure_model.predict_proba(df_with_mkt[list(fund_cols)])
        p_safe = np.where(has_mkt[:, None], p_raw, 1.0 / 3.0)
        base = np.where(
            has_mkt[:, None],
            np.log(np.clip(p_safe, 1e-6, 1.0)),
            np.log(np.clip(p_pure, 1e-6, 1.0)),
        )
    else:
        p_safe = np.where(has_mkt[:, None], p_raw, 1.0 / 3.0)
        base = np.where(has_mkt[:, None], np.log(np.clip(p_safe, 1e-6, 1.0)), 0.0)
    return base, has_mkt


# ═══════════════════════════════════════════════════════════════════════════════
# 4.3 Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════
def hyperparameter_tuning(X_train, y_train, base_train=None):
    """Cross-validation temporale per trovare i migliori iperparametri."""
    print(f"\n{'='*55}")
    print("4.3 — Hyperparameter Tuning (CV temporale)")
    print(f"{'='*55}")

    train_pool = Pool(X_train, y_train, baseline=base_train)

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


def train_final_model(X_train, y_train, best_params, base_train=None):
    """Riaddestra il modello finale con i migliori parametri sul training set."""
    print(f"\nTraining finale con best params...")
    final_model = CatBoostClassifier(
        **best_params,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=200
    )
    final_model.fit(Pool(X_train, y_train, baseline=base_train))
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
def main(league, fast: bool = False, tune: bool = False):
    print("╔" + "═" * 53 + "╗")
    title = f"FASE 4 — CatBoost Model Training & Evaluation ({league.name})"
    print("║" + title.ljust(54)[:53] + "║")
    print("╚" + "═" * 53 + "╝")

    X_train_full, X_test_full, y_train, y_test = load_data(league)
    print(f"Train: {len(X_train_full)} | Test: {len(X_test_full)}")

    _all_cols = list(X_train_full.columns)
    keep = simplified_feature_set(_all_cols)
    dropped = [c for c in _all_cols if c not in keep]
    X_train = X_train_full[keep]
    X_test = X_test_full[keep]
    print(f"Feature fondamentali negli alberi ({len(keep)}): {keep}")
    if dropped:
        print(f"Feature di mercato escluse dagli alberi ({len(dropped)}, usate solo come prior baseline)")

    MODEL_PATH = league.model_path
    PURE_PATH = MODELS_DIR / f"{league.model_name}_pure.cbm"

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

        pure_model = None
        if PURE_PATH.exists():
            pure_model = CatBoostClassifier()
            pure_model.load_model(str(PURE_PATH))

        if len(X_test) > 0:
            _feat = list(final_model.feature_names_)
            if any(c.startswith("mkt_") for c in _feat):
                final_metrics = evaluate_model(final_model, X_test_full[_feat], y_test, "Loaded")
            else:
                base_test, _ = compute_market_baseline(X_test_full, pure_model=pure_model, fund_cols=_feat)
                final_metrics = evaluate_model(final_model, X_test_full[_feat], y_test, "Loaded", base_test=base_test)
        else:
            print("Nessun dato di test disponibile per la valutazione.")
            final_metrics = {"accuracy": 0.0, "rps": 0.0}

        baseline_metrics = final_metrics
        best_params = {}
        cv_results = []
    else:
        # 1. Modello Puro di fallback (solo 29 fondamentali, senza quote di mercato)
        pure_model = CatBoostClassifier(
            **PURE_PARAMS,
            loss_function='MultiClass',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=False,
        )
        pure_model.fit(X_train, y_train)
        pure_model.save_model(str(PURE_PATH))
        print(f"Modello puro di fallback salvato in {PURE_PATH}")

        # 2. Training degli alberi sul residuo rispetto a B365 (`use_sharp=False`);
        #    in valutazione/inferenza si usa il prior sharp (`use_sharp=True`).
        base_train_all, has_mkt_train = compute_market_baseline(
            X_train_full, pure_model=pure_model, fund_cols=keep, use_sharp=False
        )
        base_test, _ = compute_market_baseline(
            X_test_full, pure_model=pure_model, fund_cols=keep, use_sharp=True
        )

        if has_mkt_train.sum() >= 20:
            X_res_tr = X_train.loc[has_mkt_train].reset_index(drop=True)
            y_res_tr = y_train.loc[has_mkt_train].reset_index(drop=True)
            base_res_tr = base_train_all[has_mkt_train]
        else:
            X_res_tr = X_train.reset_index(drop=True)
            y_res_tr = y_train.reset_index(drop=True)
            base_res_tr = base_train_all

        baseline_model = train_baseline(X_res_tr, X_test, y_res_tr, y_test, base_train=base_res_tr)
        baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Baseline", base_test=base_test)

        if fast:
            best_params = dict(RESIDUAL_PARAMS_BY_LEAGUE.get(league.key, DEFAULT_PARAMS))
            cv_results = [{"params": best_params, "note": "parametri residuali (--fast)"}]
            print(f"  [fast] parametri residuali: {best_params}")
        elif tune:
            best_params, cv_results = hyperparameter_tuning(X_res_tr, y_res_tr, base_train=base_res_tr)
        else:
            best_params = dict(RESIDUAL_PARAMS_BY_LEAGUE.get(league.key, DEFAULT_PARAMS))
            cv_results = [{"params": best_params,
                           "note": "parametri residuali per value betting su baseline log(p_b365)"}]
            print(f"  parametri residuali ({league.key}): {best_params}")

        final_model = train_final_model(X_res_tr, y_res_tr, best_params, base_train=base_res_tr)
        final_metrics = evaluate_model(final_model, X_test, y_test, "Residual", base_test=base_test)

        # Il prior di mercato B365 de-viggato e' gia' calibrato sul log-loss;
        # la calibrazione post-processing resta identita' per non spostare l'offset.
        alpha = [0.0, 0.0, 0.0]
        w = 1.0
        n_val = int(len(y_res_tr))
        final_metrics["calibration_alpha"] = alpha
        final_metrics["calibration_w"] = w
        final_metrics["calibration_n_val"] = n_val
        print(f"\n  Calibrazione residuale: identita' alpha={alpha}, w={w} (prior mercato log(p_b365))")

        cal_path = MODELS_DIR / f"calibration_{league.key}.json"
        with open(cal_path, "w") as f:
            json.dump({"alpha": alpha, "w": w, "n_val": n_val, "method": "residual_logit_offset"}, f, indent=2)
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
    ap.add_argument("--tune", action="store_true",
                    help="riesegui la griglia di iperparametri con CV "
                         "temporale invece di usare quelli gia' scelti su training")
    ap.add_argument("--fast", action="store_true",
                    help="salta la griglia di hyperparameter tuning (CV 5-fold): "
                         "per misure locali ed esperimenti")
    args = ap.parse_args()
    main(get_league(args.league), fast=args.fast, tune=args.tune)
