#!/usr/bin/env python3
"""
SELEZIONE DELLE FEATURE: il modello piu' semplice che non perde qualita'.

PERCHE' SERVE
-------------
Il modello ha 41 feature. Molte sono ridondanti (per esempio Elo e la sua
"expect", o 12 feature di mercato che contengono la stessa informazione in
piu' forme). La letteratura e' netta sul punto: con dati di calcio piu' che
poco voluminosi, l'overfitting e il rischio di sovrastimazione crescono
(Meinshausen & Buhlmann 2010 sulla stabilita' della selezione; Hastie,
Tibshirani & Friedman ESL cap. 10 sulla selezione del modello; e per il
calcio Rathke & Wiemann 2015 che ottengono previsioni competitive con un
modello Dixon-Coles a pochissimi parametri).

IL VINCOLO STATISTICO (ed e' il punto)
---------------------------------------
La selezione NON deve guardare il test set: scegliere le feature su 438
partite e poi valutarle sulle stesse 438 partite e' overfitting del test set,
e produrrebbe un numero falso. Quindi:

  FASE 1 (selezione): solo sul TRAINING, con cross-validation TEMPORALE.
                       A ogni passo si prova a togliere un gruppo di feature
                       e si tiene la rimozione se il degrado non e'
                       statisticamente significativo (test paired sulle
                       perdite per partita dei fold out-of-fold).
  FASE 2 (verifica): il set selezionato e il modello completo vengono
                      confrontati sul TEST set, UNA volta sola, con test
                      paired (bootstrap + Wilcoxon + Diebold-Mariano).

Cosa significa "non perdere qualita'": il criterio e' la RPS (la metrica
usata dal repo per confrontare il modello col bookmaker: premia la
calibrazione della probabilita', non solo la scelta dell'esito) e il
log-loss. Un peggioramento non significativo e' accettato; uno significativo no.

Uso:
  python src/select_features.py --mode select   # fase 1, solo training
  python src/select_features.py --mode final    # fase 2, test set
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from implied import (  # noqa: E402
    accuracy,
    brier_mse,
    f1_scores,
    log_loss,
    ranked_probability_score,
)

LABELS = ["H", "D", "A"]


# ─────────────────────────────────────────────────────────────────────────────
# Test statistici paired
# ─────────────────────────────────────────────────────────────────────────────
def paired_bootstrap(loss_a, loss_b, n_boot=10000, seed=42):
    """Bootstrap paired sulla differenza di perdita media (b - a).

    loss_a, loss_b: perdite PER OSSERVAZIONE (RPS o log-loss per partita).
    Restituisce la differenza, il suo IC 95% e p(two-sided) bootstrap.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(loss_b) - np.asarray(loss_a)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    obs = d.mean()
    # p-value bootstrap a due code: quanto spesso il bootstrap e' dall'altra
    # parte dello zero rispetto allo zero.
    if obs >= 0:
        p = 2.0 * (boot <= 0).mean()
    else:
        p = 2.0 * (boot >= 0).mean()
    return {
        "diff": float(obs),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_bootstrap": float(min(1.0, p)),
        "n": int(n),
    }


def paired_wilcoxon(loss_a, loss_b):
    """Wilcoxon signed-rank sulle differenze per partita (non parametrico)."""
    from scipy.stats import wilcoxon
    d = np.asarray(loss_b) - np.asarray(loss_a)
    if np.allclose(d, 0):
        return {"stat": 0.0, "p": 1.0}
    res = wilcoxon(d, zero_method="zsplit", alternative="two-sided")
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def dm_test(loss_a, loss_b, horizon=None):
    """Diebold-Mariano (o Clark-West se si passa il modello annidato).

    Serve a stabilire se la differenza di accuratezza prevedibile e'
    significativa: e' il test standard per confrontare due modelli
    predittivi (Diebold & Mariano 1995; Harvey, Leybourne & Newbold 1997
    per la correzione a campione piccolo; bashtage/arch implementa
    entrambi, con correzione HAC per l'autocorrelazione).
    """
    d = np.asarray(loss_b) - np.asarray(loss_a)
    try:
        from arch.test import dm_test as _dm
        stat, p = _dm(d, power=2, horizon=horizon)
        return {"stat": float(stat), "p": float(p), "source": "arch.dm_test"}
    except Exception:
        pass
    # Implementazione propria, con la correzione per campione piccolo di
    # Harvey, Leybourne & Newbold (1997): la statistica e' una t con
    # gradi di liberta' T-1 invece che una normale, e la varianza e' quella
    # long-run (autocorrelazione delle differenze seriali).
    from scipy import stats
    T = len(d)
    d_bar = float(np.mean(d))
    gamma = float(np.sum(d[:-1] * d[1:]))
    l = d_bar + (1.0 - 1.0 / T) * (gamma / T) if T > 1 else d_bar
    if l <= 0:
        return {"stat": None, "p": None, "source": "varianza long-run non positiva"}
    dm = d_bar / np.sqrt(l / T)
    pval = 2.0 * (1.0 - stats.t.cdf(abs(dm), df=T - 1))
    return {"stat": float(dm), "p": float(pval),
            "source": "DM (Harvey-Leybourne-Newbold 1997, t con df=T-1)"}


# ─────────────────────────────────────────────────────────────────────────────
# Valutazione
# ─────────────────────────────────────────────────────────────────────────────
def rps_per_match(y_true, probs):
    """RPS per singola partita (vettore), per i test paired."""
    out = np.zeros(len(y_true))
    for i, (y, p) in enumerate(zip(y_true, probs)):
        cum_p = np.cumsum(p)
        cum_y = np.cumsum([1.0 if j == y else 0.0 for j in range(3)])
        out[i] = np.sum((cum_p[:-1] - cum_y[:-1]) ** 2) / 2.0
    return out


def logloss_per_match(y_true, probs):
    out = np.zeros(len(y_true))
    for i, (y, p) in enumerate(zip(y_true, probs)):
        out[i] = -np.log(max(p[y], 1e-12))
    return out


def temporal_folds(n, k=4):
    """Fold temporali: si valuta sempre sul FUTURO del training.

    Fold 0: addestra [0, n*1/4)  valida [n*1/4, n*2/4)
    ...
    L'ultimo fold non addestra sui dati 'a cavallo' della validazione.
    """
    edges = [int(n * i / k) for i in range(k + 1)]
    folds = []
    for i in range(k - 1):
        tr_end = edges[i + 1]
        va_end = edges[i + 2]
        folds.append((np.arange(0, tr_end), np.arange(tr_end, va_end)))
    return folds


def train_model(X, y, iterations, lr, depth, seed=42):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=iterations, learning_rate=lr, depth=depth,
                           loss_function="MultiClass", eval_metric="Accuracy",
                           random_seed=seed, verbose=False)
    m.fit(X, list(y))
    return m


def cv_losses(X, y, cols, iterations, lr, depth, k=4):
    """Cross-validation temporale sul TRAINING: restituisce le perdite
    per osservazione out-of-fold (RPS e log-loss) e le previsioni."""
    n = len(y)
    folds = temporal_folds(n, k)
    oof_rps = np.full(n, np.nan)
    oof_ll = np.full(n, np.nan)
    oof_pred = np.full(n, -1)
    oof_probs = np.full((n, 3), np.nan)
    for tr_idx, va_idx in folds:
        m = train_model(X.iloc[tr_idx][cols], list(np.asarray(y)[tr_idx]),
                        iterations, lr, depth)
        p = m.predict_proba(X.iloc[va_idx][cols])
        yy = np.asarray(y)[va_idx]
        oof_rps[va_idx] = rps_per_match(yy, p)
        oof_ll[va_idx] = logloss_per_match(yy, p)
        oof_pred[va_idx] = np.argmax(p, axis=1)
        oof_probs[va_idx] = p
    valid = ~np.isnan(oof_rps)
    return {
        "rps": oof_rps[valid],
        "ll": oof_ll[valid],
        "pred": oof_pred[valid],
        "probs": oof_probs[valid],
        "idx": np.where(valid)[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gruppi candidati alla rimozione (dal basso verso l'alto dell'importanza)
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = [
    # ridondanze interne alle famiglie
    ("overround (4 variabili di mercato)", ["mkt_b365_overround", "mkt_pinnacle_overround",
                                            "mkt_avg_overround"]),
    ("Elo expect (2)", ["elo_home_expect", "elo_away_expect"]),
    ("Elo rating (2)", ["elo_home", "elo_away"]),
    ("consensus mercato (4)", ["mkt_avg_pH", "mkt_avg_pD", "mkt_avg_pA",
                               "mkt_avg_overround"]),
    ("Pinnacle (4)", ["mkt_pinnacle_pH", "mkt_pinnacle_pD", "mkt_pinnacle_pA",
                      "mkt_pinnacle_overround"]),
    ("bet365 (4)", ["mkt_b365_pH", "mkt_b365_pD", "mkt_b365_pA", "mkt_b365_overround"]),
    ("mercato intero (12)", ["mkt_b365_pH", "mkt_b365_pD", "mkt_b365_pA", "mkt_b365_overround",
                              "mkt_pinnacle_pH", "mkt_pinnacle_pD", "mkt_pinnacle_pA",
                              "mkt_pinnacle_overround",
                              "mkt_avg_pH", "mkt_avg_pD", "mkt_avg_pA", "mkt_avg_overround"]),
    ("cartellini (2)", ["home_cards_5", "away_cards_5"]),
    ("angoli (2)", ["home_corners_5", "away_corners_5"]),
    ("tiri in porta (2)", ["home_shots_t_5", "away_shots_t_5"]),
    ("tiri totali (2)", ["home_shots_5", "away_shots_5"]),
    ("prestazione intera (8)", ["home_shots_5", "away_shots_5", "home_shots_t_5",
                                "away_shots_t_5", "home_corners_5", "away_corners_5",
                                "home_cards_5", "away_cards_5"]),
    ("h2h (1)", ["h2h_home_winrate"]),
    ("neopromossa (2)", ["home_promoted", "away_promoted"]),
    ("riposo (2)", ["home_rest_days", "away_rest_days"]),
    ("differenza reti (2)", ["home_goal_diff_5", "away_goal_diff_5"]),
    ("forma (2)", ["home_form_5", "away_form_5"]),
    ("gol rolling (4)", ["home_goals_scored_5", "home_goals_conceded_5",
                          "away_goals_scored_5", "away_goals_conceded_5"]),
    ("Pi-Rating attacco (2)", ["pi_home_H", "pi_away_H"]),
    ("Pi-Rating difesa (2)", ["pi_home_A", "pi_away_A"]),
    ("Pi-Rating intero (4)", ["pi_home_H", "pi_away_H", "pi_home_A", "pi_away_A"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="select", choices=["select", "params", "final"])
    ap.add_argument("--league", default="seriea")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--feature-set", default="all",
                    choices=["all", "selected", "no_market_ext"],
                    help="quale set di feature usare: all=41, selected=quello "
                         "scelto dalla selezione, no_market_ext=41 senza le 8 "
                         "di Pinnacle/consensus e senza i 4 overround (che in "
                         "produzione sulle partite future mancano comunque)")
    args = ap.parse_args()

    from leagues import get_league
    league = get_league(args.league)

    X_train = pd.read_csv(league.split_path("X_train_v2"))
    X_test = pd.read_csv(league.split_path("X_test_v2"))
    y_train = pd.read_csv(league.split_path("y_train_v2")).squeeze().astype(int).tolist()
    y_test = pd.read_csv(league.split_path("y_test_v2")).squeeze().astype(int).tolist()
    ALL = list(X_train.columns)
    if args.feature_set == "selected":
        with open(ROOT / "models" / f"selected_features_{league.key}.json", encoding="utf-8") as f:
            ALL = json.load(f)["selected"]
    elif args.feature_set == "no_market_ext":
        drop = [c for c in X_train.columns
                if c.startswith("mkt_pinnacle_") or c.startswith("mkt_avg_")
                or c.endswith("_overround")]
        ALL = [c for c in X_train.columns if c not in drop]
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    print("=" * 100)
    print(f"  SELEZIONE FEATURE — {league.name} — mode={args.mode} — "
          f"{len(ALL)} feature iniziali")
    print("=" * 100)

    # ── Configurazione di produzione (iperparametri scelti dalla CV) ──────
    PROD = {"seriea": dict(iterations=1500, lr=0.02, depth=6),
            "eredivisie": dict(iterations=1000, lr=0.03, depth=8)}
    prod = PROD.get(args.league, dict(iterations=1000, lr=0.03, depth=6))

    if args.mode == "params":
        # ── Scelta degli IPERPARAMETRI, sempre sul TRAINING ───────────────
        # Confronto sul test set dei parametri sarebbe ottimizzare sul test
        # set: i numeri che ne uscirebbero sarebbero falsi. Quindi la scelta
        # avviene qui, con cross-validation temporale sul training, e il
        # test set viene usato UNA volta sola, per la verifica finale.
        selected = ALL  # il set richiesto da --feature-set
        configs = [
            {"iterations": 1500, "lr": 0.02, "depth": 6},   # produzione attuale
            {"iterations": 1000, "lr": 0.03, "depth": 6},
            {"iterations": 1000, "lr": 0.03, "depth": 4},
            {"iterations": 500, "lr": 0.05, "depth": 6},
            {"iterations": 300, "lr": 0.05, "depth": 4},
            {"iterations": 300, "lr": 0.05, "depth": 6},
            {"iterations": 200, "lr": 0.05, "depth": 4},
        ]
        print(f"\n  Scelta iperparametri sul TRAINING (CV temporale, {args.folds} fold)")
        print(f"  set di feature: {len(selected)} ({selected})\n")
        rows = []
        for cfg in configs:
            t0 = time.time()
            res = cv_losses(X_train, y_train, selected, cfg["iterations"],
                            cfg["lr"], cfg["depth"], args.folds)
            acc = float((res["pred"] == y_train[res["idx"]]).mean())
            rows.append({**cfg, "rps_oof": float(res["rps"].mean()),
                         "ll_oof": float(res["ll"].mean()), "acc": acc,
                         "secondi": round(time.time() - t0, 1)})
            print(f"  iter={cfg['iterations']:>4} lr={cfg['lr']:.2f} depth={cfg['depth']} | "
                  f"RPS OOF {res['rps'].mean():.5f} | LL {res['ll'].mean():.5f} | "
                  f"acc {acc:.2%} | {time.time()-t0:.0f}s")
        # scelta: la RPS OOF migliore; a parita' il modello piu' semplice
        best = min(rows, key=lambda r: (round(r["rps_oof"], 5),
                                         r["iterations"], r["depth"]))
        print(f"\n  → scelta: {best} (RPS OOF {best['rps_oof']:.5f})")
        out = Path(args.out) if args.out else (ROOT / "models" / f"selected_params_{league.key}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "best": best, "features": selected},
                      f, indent=2, default=float)
        print(f"  Report: {out}")
        return

    if args.mode == "select":
        # ── FASE 1: selezione sul TRAINING, CV temporale ─────────────────
        it, lr, depth = args.iterations, args.lr, args.depth
        print(f"\n  Fase 1: selezione su TRAINING con CV temporale "
              f"({args.folds} fold, iter={it}, lr={lr}, depth={depth})")
        print("  Criterio: si tiene la rimozione se il peggioramento della RPS")
        print("  out-of-fold NON e' significativo (bootstrap paired, p >= 0.05).\n")

        t0 = time.time()
        base = cv_losses(X_train, y_train, ALL, it, lr, depth, args.folds)
        base_rps = base["rps"]
        print(f"  baseline ({len(ALL)} feature): RPS OOF {base_rps.mean():.5f} "
              f"| LL {base['ll'].mean():.5f} | acc {(base['pred'] == y_train[base['idx']]).mean():.2%} "
              f"({time.time()-t0:.0f}s)")

        current = list(ALL)
        trace = []
        for name, group in GROUPS:
            group = [g for g in group if g in current]
            if not group:
                continue
            cand = [c for c in current if c not in group]
            if not cand:
                continue
            t1 = time.time()
            res = cv_losses(X_train, y_train, cand, it, lr, depth, args.folds)
            # il confronto e' PAIRED: si usano le STESSE osservazioni
            # out-of-fold (indice di riga), altrimenti le differenze non
            # sono accoppiate e il test perde potere.
            common = np.intersect1d(base["idx"], res["idx"])
            m_base = base["rps"][np.isin(base["idx"], common)]
            m_cand = res["rps"][np.isin(res["idx"], common)]
            bs = paired_bootstrap(m_base, m_cand)
            kept = bs["p_bootstrap"] >= 0.05
            trace.append({
                "gruppo": name, "rimosse": group, "n_da": len(current), "n_a": len(cand),
                "rps_base": float(m_base.mean()), "rps_cand": float(m_cand.mean()),
                "diff_rps": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                "p_bootstrap": bs["p_bootstrap"], "rimossa": bool(kept),
                "secondi": round(time.time() - t1, 1),
            })
            flag = "RIMOSSE" if kept else "tenute "
            print(f"  {flag} {name:<34} {len(current):>3}->{len(cand):>3} feat | "
                  f"RPS {m_base.mean():.5f} -> {m_cand.mean():.5f} "
                  f"({bs['diff']:+.5f}) p={bs['p_bootstrap']:.3f} "
                  f"[{bs['ci_low']:+.5f},{bs['ci_high']:+.5f}] {time.time()-t1:.0f}s")
            if kept:
                current = cand
                base = {"rps": m_cand, "ll": res["ll"][np.isin(res["idx"], common)],
                        "pred": res["pred"][np.isin(res["idx"], common)],
                        "idx": common}
                base_rps = base["rps"]
        print(f"\n  → set selezionato: {len(current)} feature (da {len(ALL)})")
        print("   " + ", ".join(current))
        out = Path(args.out) if args.out else (ROOT / "models" / f"selected_features_{league.key}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"selected": current, "trace": trace,
                       "params": {"iterations": it, "lr": lr, "depth": depth},
                       "folds": args.folds}, f, indent=2, default=float)
        print(f"  Report: {out}")
        return

    # ── FASE 2: verifica finale sul TEST set ───────────────────────────
    sel_path = ROOT / "models" / f"selected_features_{league.key}.json"
    with open(sel_path, encoding="utf-8") as f:
        sel = json.load(f)
    selected = sel["selected"]

    print(f"\n  Fase 2: confronto sul TEST set ({len(y_test)} partite)")
    print(f"  modello completo: {len(ALL)} feature | modello semplificato: {len(selected)}")
    print(f"  iperparametri di produzione: {prod}\n")

    results = {}
    # I parametri del modello semplificato sono quelli SCELTI SUL TRAINING
    # (mode params), non quelli migliori sul test set.
    chosen = None
    ppath = ROOT / "models" / f"selected_params_{league.key}.json"
    if ppath.exists():
        with open(ppath, encoding="utf-8") as f:
            chosen = json.load(f)["best"]
    else:
        chosen = dict(iterations=300, lr=0.05, depth=4)
    for tag, cols, p in (("completo", ALL, prod),
                          ("semplificato", selected, chosen),
                          ("completo_con_parametri_semplici", ALL, chosen),
                          ("semplificato_con_parametri_produzione", selected, prod)):
        t0 = time.time()
        # NB: train_model(X, y, iterations, lr, depth) prende gia' il
        # sottoinsieme di colonne come X (la firma non ha un parametro
        # 'cols': passarlo shiftava le colonne dentro 'iterations').
        m = train_model(X_train[cols], y_train, p["iterations"], p["lr"], p["depth"])
        pr = m.predict_proba(X_test[cols])
        results[tag] = {
            "n_features": len(cols),
            "params": p,
            "rps": float(ranked_probability_score(list(y_test), pr.tolist())),
            "log_loss": float(log_loss(list(y_test), pr.tolist())),
            "accuracy": float(accuracy(list(y_test), [int(np.argmax(x)) for x in pr])),
            "f1_macro": float(f1_scores(list(y_test), [int(np.argmax(x)) for x in pr])["macro"]),
            "brier": float(brier_mse(list(y_test), pr.tolist())),
            "rps_per_match": rps_per_match(y_test, pr),
            "ll_per_match": logloss_per_match(y_test, pr),
            "secondi": round(time.time() - t0, 1),
        }
        r = results[tag]
        print(f"  {tag:<28} {r['n_features']:>3} feat | acc {r['accuracy']:.2%} | "
              f"F1 {r['f1_macro']:.4f} | MSE {r['brier']:.5f} | RPS {r['rps']:.5f} | "
              f"LL {r['log_loss']:.5f} ({r['secondi']}s)")

    print(f"\n  {'─'*100}\n  TEST PAIRED (le perdite sono per partita, quindi accoppiate)\n  {'─'*100}")
    full = results["completo"]
    for tag in ("semplificato", "completo_con_parametri_semplici",
                "semplificato_con_parametri_produzione"):
        r = results[tag]
        bs = paired_bootstrap(full["rps_per_match"], r["rps_per_match"])
        wc = paired_wilcoxon(full["rps_per_match"], r["rps_per_match"])
        dm = dm_test(full["rps_per_match"], r["rps_per_match"])
        print(f"\n  {tag} vs completo (RPS):")
        print(f"    differenza media  : {bs['diff']:+.6f}  (positivo = peggio)")
        print(f"    bootstrap 95% CI   : [{bs['ci_low']:+.6f}, {bs['ci_high']:+.6f}]")
        print(f"    p bootstrap        : {bs['p_bootstrap']:.4f}")
        print(f"    Wilcoxon p         : {wc['p']:.4f}")
        print(f"    Diebold-Mariano    : stat={dm['stat']} p={dm['p']} ({dm['source']})")
        verdict = "NON SIGNIFICATIVO (equivalente)" if bs["p_bootstrap"] >= 0.05 else "SIGNIFICATIVO (peggiora)"
        print(f"    → {verdict}")
        r["test_vs_completo"] = {"bootstrap": bs, "wilcoxon": wc, "dm": dm}

    out = Path(args.out) if args.out else (ROOT / "models" / f"simplified_{league.key}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"selected": selected, "results": {
            k: {kk: vv for kk, vv in v.items() if not kk.endswith("_per_match")}
            for k, v in results.items()}}, f, indent=2, default=float)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    main()
