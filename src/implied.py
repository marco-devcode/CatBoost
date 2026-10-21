# src/implied.py
"""
Probabilita' implicite dalle quote 1X2: rimozione del margine (de-vig) e
metriche probabilistiche.

PERCHE' ESISTE
---------------
Il repo storicamente usava UNA sola rimozione del margine, la normalizzazione
proporzionale ("multiplicative"):

    p_i = (1/o_i) / sum_j (1/o_j)

Questo e' statisticamente SBAGLIATO per il calcio 1X2. Le quote contengono un
margine ("overround") e i bookmaker non lo distribuiscono in modo uniforme:
caricano un margine PIU' ALTO sui longshot (favorite-longshot bias). La
normalizzazione proporzionale rimuove quindi troppo dal favorito e troppo
poco dal longshot, e sbaglia il verso esatto.

Conseguenza pratica nel repo: il confronto "il modello batte il bookmaker"
usava la stima SBAGLIATA del bookmaker, quindi il benchmark era piu' debole
del bookmaker reale. Confronto corretto -> il margine di vittoria del modello
si riduce (o sparisce). Serve misurarlo, non evitarlo.

METODI IMPLEMENTATI (tutti su quote 1X2, tre esiti)
---------------------------------------------------
1. multiplicative (= proporzionale, il metodo storico del repo)
2. additive       (margine equidistribuito)
3. power          (Potra 1990)
4. odds_ratio     (rapporti di quota)
5. shin           (Shin 1992/1993, "insider trading" -> favourite-longshot)

Formula di Shin usata (Shin 1993, con PI = sum delle probabilita' implicite
lordate, cioe' 1 + overround):

    p_i(z) = ( sqrt(z^2 + 4(1-z) * pi_i^2 / PI) - z ) / (2(1-z))

e z e' l'unica radice in (0, 1) tale che sum_i p_i(z) = 1. La funzione e'
monotona in z: si risolve con bisezione (deterministica, senza scipy).

RIFERIMENTI
-----------
- Shin, H.S. (1992). "Pricing of winner and loser in a market for
  state-contingent claims with insider trading". Journal of Finance 47(5).
- Shin, H.S. (1993). "Measuring the incidence of insider trading in a market
  for state-contingent claims". The Economic Journal 103(422), 1141-1153.
- Power, C. (1990). "Football betting: can the behaviour of the odds market be
  used to improve the forecasting process?" International Journal of
  Forecasting 6(2), 111-118.
- Strumbelj, E. (2014). "On determining probability forecasts from betting
  odds". International Journal of Forecasting 30(4), 934-943.
- Eckert, J.S. (2013). "Fair probabilities from bookmaker odds". University of
  Canterbury Working Paper.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

__all__ = [
    "implied_probs",
    "overround",
    "de_vig_multiplicative",
    "de_vig_additive",
    "de_vig_power",
    "de_vig_odds_ratio",
    "de_vig_shin",
    "DE_VIG_METHODS",
    "ranked_probability_score",
    "log_loss",
    "brier_mse",
    "accuracy",
    "f1_scores",
    "calibration_table",
]


# ─────────────────────────────────────────────────────────────────────────────
# Implied probabilities
# ─────────────────────────────────────────────────────────────────────────────
def implied_probs(odds: Sequence[float]) -> list[float] | None:
    """Probabilita' implicite LORDATE 1/odds, o None se le quote non sono valide.

    Non rimuove il margine: e' il punto di partenza di ogni de-vig.
    """
    try:
        vals = [float(o) for o in odds]
    except (TypeError, ValueError):
        return None
    if len(vals) != 3:
        return None
    if any(not math.isfinite(v) or v <= 1.0 for v in vals):
        return None
    return [1.0 / v for v in vals]


def overround(odds: Sequence[float]) -> float | None:
    """Overround = sum(1/odds) - 1. 0.05 = 5% di margine."""
    p = implied_probs(odds)
    if p is None:
        return None
    return sum(p) - 1.0


def _normalise(vec: list[float]) -> list[float]:
    tot = sum(vec)
    if tot <= 0:
        return []
    return [v / tot for v in vec]


# ── 1. Multiplicative (proporzionale): il metodo storico del repo ──────────
def de_vig_multiplicative(odds: Sequence[float]) -> list[float] | None:
    """Rimuove il margine in proporzione: p_i = (1/o_i) / sum(1/o_j).

    Storicamente usato dal repo. E' sbagliato quando il margine non e'
    uniforme (favourite-longshot bias): toglie troppo al favorito.
    """
    p = implied_probs(odds)
    if p is None:
        return None
    return _normalise(p)


# ── 2. Additive: margine equidistribuito ────────────────────────────────────
def de_vig_additive(odds: Sequence[float]) -> list[float] | None:
    """p_i = pi_i - overround/3 (stesso margine assoluto su ogni esito)."""
    p = implied_probs(odds)
    if p is None:
        return None
    m = (sum(p) - 1.0) / 3.0
    out = [v - m for v in p]
    if any(v <= 0 for v in out):
        return None
    return out


# ── 3. Power (Potra 1990) ───────────────────────────────────────────────────
def de_vig_power(odds: Sequence[float], tol: float = 1e-12) -> list[float] | None:
    """Metodo della potenza: cerca k con sum_i pi_i^k = 1, poi p_i = pi_i^k.

    La potenza sposta massa verso i longshot rispetto al proporzionale
    (l'esponente k > 1 schiaccia i valori grandi e alza i piccoli in termini
    relativi al margine).
    """
    p = implied_probs(odds)
    if p is None:
        return None
    tot = sum(p)
    if tot <= 1.0:
        return _normalise(p)

    def f(k: float) -> float:
        return sum(v ** k for v in p) - 1.0

    lo, hi = 1.0, 10.0
    if f(hi) > 0:
        return _normalise(p)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = (lo + hi) / 2.0
    out = [v ** k for v in p]
    return _normalise(out)


# ── 4. Odds ratio: rapporto di quota costante ───────────────────────────────
def de_vig_odds_ratio(odds: Sequence[float], tol: float = 1e-12) -> list[float] | None:
    """Risolve c tale che p_i = c*pi_i / (1 - pi_i + c*pi_i) e sum_i p_i = 1.

    Il rapporto (p_i/(1-p_i)) / (pi_i/(1-pi_i)) e' costante su tutti gli esiti.
    """
    p = implied_probs(odds)
    if p is None:
        return None
    tot = sum(p)
    if tot <= 1.0:
        return _normalise(p)

    def f(c: float) -> float:
        s = 0.0
        for v in p:
            denom = (1.0 - v) + c * v
            if denom <= 0:
                return float("inf")
            s += (c * v) / denom
        return s - 1.0

    lo, hi = 1e-9, 1e9
    if f(lo) > 0 or f(hi) < 0:
        return _normalise(p)
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol * max(1.0, hi):
            break
    c = (lo + hi) / 2.0
    out = []
    for v in p:
        denom = (1.0 - v) + c * v
        out.append((c * v) / denom if denom > 0 else 0.0)
    return _normalise(out)


# ── 5. Shin (1993) ──────────────────────────────────────────────────────────
def _shin_probs_for_z(p: Sequence[float], z: float) -> list[float]:
    pi_sum = sum(p)
    if pi_sum <= 0:
        return []
    out = []
    for v in p:
        disc = z * z + 4.0 * (1.0 - z) * (v * v / pi_sum)
        out.append((math.sqrt(max(disc, 0.0)) - z) / (2.0 * (1.0 - z)))
    return out


def de_vig_shin(odds: Sequence[float], tol: float = 1e-10) -> list[float] | None:
    """De-vig di Shin: risolve z con sum_i p_i(z) = 1 (bisezione su z in (0,1)).

    E' il metodo raccomandato per il calcio 1X2: modella esplicitamente il
    favourite-longshot bias (gli "insider" spostano margine sui favoriti, quindi
    i longshot sono SOTTOSTIMATI dalle quote e vanno valutati piu' in alto).
    """
    p = implied_probs(odds)
    if p is None:
        return None
    tot = sum(p)
    if tot <= 1.0:
        return _normalise(p)

    def f(z: float) -> float:
        return sum(_shin_probs_for_z(p, z)) - 1.0

    lo, hi = 1e-9, 1.0 - 1e-9
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # Nessuna radice interna: fallback al proporzionale.
        return _normalise(p)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if f(mid) * f_lo > 0:
            lo = mid
            f_lo = f(mid)
        else:
            hi = mid
        if hi - lo < tol:
            break
    z = (lo + hi) / 2.0
    return _normalise(_shin_probs_for_z(p, z))


DE_VIG_METHODS = {
    "multiplicative": de_vig_multiplicative,
    "additive": de_vig_additive,
    "power": de_vig_power,
    "odds_ratio": de_vig_odds_ratio,
    "shin": de_vig_shin,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────────────────────
def ranked_probability_score(y_true: Iterable[int], y_proba: Sequence[Sequence[float]]) -> float:
    """RPS (Ranked Probability Score). Piu' basso e' meglio.

    RPS = (1/K) * sum_k (CDF_prev - CDF_true)^2, con K = numero esiti.
    """
    y_true = list(y_true)
    n = len(y_true)
    if n == 0:
        return float("nan")
    K = len(y_proba[0])
    total = 0.0
    for true, prob in zip(y_true, y_proba):
        cum_true = 0.0
        cum_pred = 0.0
        acc = 0.0
        for k in range(K):
            cum_true += 1.0 if k == true else 0.0
            cum_pred += prob[k]
            acc += (cum_pred - cum_true) ** 2
        total += acc / K
    return total / n


def log_loss(y_true: Iterable[int], y_proba: Sequence[Sequence[float]], eps: float = 1e-15) -> float:
    """Log-loss (cross-entropy) media. Piu' basso e' meglio."""
    y_true = list(y_true)
    n = len(y_true)
    if n == 0:
        return float("nan")
    total = 0.0
    for true, prob in zip(y_true, y_proba):
        total -= math.log(max(float(prob[true]), eps))
    return total / n


def brier_mse(y_true: Iterable[int], y_proba: Sequence[Sequence[float]]) -> float:
    """MSE sulle probabilita' (= Brier multiclasse, media su esiti e partite).

    mean_{i,k} (p_ik - y_ik)^2. Richiesta esplicita del task come "MSE".
    """
    y_true = list(y_true)
    n = len(y_true)
    if n == 0:
        return float("nan")
    K = len(y_proba[0])
    total = 0.0
    for true, prob in zip(y_true, y_proba):
        for k in range(K):
            target = 1.0 if k == true else 0.0
            total += (float(prob[k]) - target) ** 2
    return total / (n * K)


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return float("nan")
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def f1_scores(y_true: Iterable[int], y_pred: Iterable[int], K: int = 3) -> dict:
    """F1 macro, F1 weighted e F1 per classe. nomenclatura sklearn: macro/weighted."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    per_class = {}
    f1s = []
    weights = []
    for k in range(K):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == k and p == k)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != k and p == k)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == k and p != k)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[k] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        f1s.append(f1)
        weights.append(tp + fn)
    total = sum(weights)
    macro = sum(f1s) / K
    weighted = (sum(f * w for f, w in zip(f1s, weights)) / total) if total else 0.0
    return {"macro": macro, "weighted": weighted, "per_class": per_class}


def calibration_table(y_true: Iterable[int], y_proba: Sequence[Sequence[float]], bins: int = 10) -> list[dict]:
    """Tabella di calibrazione per esito: probabilita' media predetta vs frequenza reale."""
    y_true = list(y_true)
    K = len(y_proba[0])
    out = []
    for k in range(K):
        rows = [(float(prob[k]), 1.0 if t == k else 0.0) for t, prob in zip(y_true, y_proba)]
        table = []
        for b in range(bins):
            lo = b / bins
            hi = (b + 1) / bins
            sel = [p for p, y in rows if (lo <= p < hi or (b == bins - 1 and p <= hi))]
            if not sel:
                continue
            table.append({
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": len(sel),
                "pred": sum(sel) / len(sel),
                "actual": sum(y for p, y in rows if (lo <= p < hi or (b == bins - 1 and p <= hi))) / len(sel),
            })
        out.append({"class": k, "bins": table})
    return out
