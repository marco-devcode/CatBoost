# scripts/test_implied.py
"""
Test del modulo src/implied.py: de-vig e metriche.

Non richiede catboost: gira con la sola stdlib, quindi si puo' eseguire
localmente (Termux, CI, ovunque).

Uso:  python scripts/test_implied.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from implied import (  # noqa: E402
    DE_VIG_METHODS,
    brier_mse,
    de_vig_additive,
    de_vig_multiplicative,
    de_vig_odds_ratio,
    de_vig_power,
    de_vig_shin,
    f1_scores,
    implied_probs,
    log_loss,
    overround,
    ranked_probability_score,
)

FAILS = []

# Valori di riferimento (penaltyblog 1.13.1, MIT): odds -> [pH, pD, pA]
# per ogni metodo. La nostra implementazione deve coincidere a 1e-6.
REFERENCE = {
    (1.60, 3.80, 5.50): (
        [0.584125, 0.245947, 0.169927],   # multiplicative
        [0.601675, 0.239833, 0.158493],   # additive
        [0.602887, 0.237564, 0.159549],   # power
        [0.595759, 0.240011, 0.164231],   # odds_ratio
        [0.597124, 0.241326, 0.161550],   # shin
    ),
    (7.00, 4.00, 1.40): (
        [0.129032, 0.225806, 0.645161],
        [0.107143, 0.214286, 0.678571],
        [0.109927, 0.207428, 0.682645],
        [0.118738, 0.212272, 0.668990],
        [0.113388, 0.216866, 0.669746],
    ),
    (1.05, 12.00, 30.00): (
        [0.890869, 0.077951, 0.031180],
        [0.929365, 0.060317, 0.010317],
        [0.940868, 0.044854, 0.014278],
        [0.925540, 0.053479, 0.020981],
        [0.919398, 0.063087, 0.017516],
    ),
}
METHOD_ORDER = ["multiplicative", "additive", "power", "odds_ratio", "shin"]


def check(cond, msg):
    if cond:
        print(f"  OK  {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS.append(msg)


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_invalid_odds():
    print("\n[1] Quote non valide -> None (mai crash, mai NaN)")
    for bad in ([1.0, 3.0, 5.0], [0.5, 3.0, 5.0], [-2, 3, 4], [float("nan"), 3, 4], [3, 4]):
        check(implied_probs(bad) is None, f"implied_probs({bad}) is None")
    for name, fn in DE_VIG_METHODS.items():
        check(fn([1.0, 3.0, 5.0]) is None, f"{name} su quote non valide -> None")


def test_sum_to_one():
    print("\n[2] Ogni de-vig restituisce 3 probabilita' positive che sommano a 1")
    odds_sets = [
        [1.50, 4.00, 6.50],   # favorito
        [2.10, 3.40, 3.60],   # equilibrato
        [7.00, 4.00, 1.40],   # sfavorito
        [1.05, 12.0, 30.0],   # favorito estremo
    ]
    for name, fn in DE_VIG_METHODS.items():
        ok_sum, ok_pos = True, True
        for o in odds_sets:
            p = fn(o)
            if p is None or not close(sum(p), 1.0, 1e-6):
                ok_sum = False
            if any(v <= 0 for v in p):
                ok_pos = False
        check(ok_sum, f"{name}: sum(p) == 1 su tutte le quote")
        check(ok_pos, f"{name}: tutte le p > 0")


def test_overround():
    print("\n[3] Overround")
    check(close(overround([2.0, 2.0, 2.0]), 0.5), "quote 2/2/2 -> overround 50%")
    check(overround([1.0, 3, 4]) is None, "quote con 1.0 -> None")


def test_against_reference():
    """Ogni de-vig deve coincidere con penaltyblog 1.13.1 a 1e-6."""
    print("\n[4] Coincidenza con penaltyblog 1.13.1 (riferimento open-source, MIT)")
    for odds, expected in REFERENCE.items():
        for name, exp in zip(METHOD_ORDER, expected):
            got = DE_VIG_METHODS[name](list(odds))
            check(got is not None and all(close(a, b, 1e-6) for a, b in zip(got, exp)),
                  f"{name}{list(odds)} -> {[round(x, 6) for x in got]} == {exp}")


def test_devig_direction():
    """
    VERSO del favourite-longshot bias.

    Il bookmaker carica un margine RELATIVAMENTE maggiore sui longshot
    (favourite-longshot bias). Quindi rimuovere il margine in modo corretto
    deve DARE PIU' PROBABILITA' AL FAVORITO rispetto alla normalizzazione
    proporzionale, che toglie il margine in proporzione (quindi troppo al
    favorito e troppo poco al longshot).

    Verificato sui valori di penaltyblog: multiplicative p_H=0.584125, gli
    altri metodi danno p_H da 0.597 a 0.603 -> tutti piu' alti.
    """
    print("\n[4b] Verso del favourite-longshot bias")
    odds = [1.60, 3.80, 5.50]
    mult = de_vig_multiplicative(odds)
    for name in ("additive", "power", "odds_ratio", "shin"):
        p = DE_VIG_METHODS[name](odds)
        check(p[0] > mult[0] + 1e-9,
              f"{name}: p_H ({p[0]:.6f}) > moltiplicativo ({mult[0]:.6f})")
        check(p[2] < mult[2] - 1e-9,
              f"{name}: p_A ({p[2]:.6f}) < moltiplicativo ({mult[2]:.6f})")

    # Caso senza margine: tutti i metodi devono dare le implicite pure.
    sym = [2.0, 4.0, 4.0]
    for name, fn in DE_VIG_METHODS.items():
        p = fn(sym)
        check(p is not None and close(p[0], 0.5, 1e-9) and close(p[2], 0.25, 1e-9),
              f"{name}: senza margine -> p = [0.5, 0.25, 0.25]")


def test_additive():
    print("\n[5] Additive: margine equidistribuito")
    # Quote 2.5/3.5/4.0: implied = 0.4, 0.285714..., 0.25 ; overround = -0.01428 -> nessun margine
    p = de_vig_additive([2.5, 3.5, 4.0])
    check(close(sum(p), 1.0, 1e-9), "somma 1")
    # Con margine: 1.8/3.4/6.0 -> implied = 0.555556, 0.294118, 0.166667 ; overround = 0.016340
    p2 = de_vig_additive([1.8, 3.4, 6.0])
    imp = implied_probs([1.8, 3.4, 6.0])
    over = sum(imp) - 1.0
    check(close(p2[0], imp[0] - over / 3, 1e-9),
          f"p_H = implied - overround/3 ({p2[0]:.6f})")
    check(close(sum(p2), 1.0, 1e-9), "somma 1 con margine")


def test_power():
    print("\n[6] Power: p_i = pi_i^k con sum(pi_i^k) = 1")
    odds = [1.60, 3.80, 5.50]
    p = de_vig_power(odds)
    check(close(sum(p), 1.0, 1e-6), "somma 1")
    pi = implied_probs(odds)
    # Con k>1 il favorito scende in modo piu' rapido del longshot.
    mult = de_vig_multiplicative(odds)
    check(p[0] > mult[0], f"p_H power ({p[0]:.5f}) > moltiplicativo ({mult[0]:.5f})")
    check(p[2] < mult[2], f"p_A power ({p[2]:.5f}) < moltiplicativo ({mult[2]:.5f})")


def test_odds_ratio():
    print("\n[7] Odds ratio")
    p = de_vig_odds_ratio([1.60, 3.80, 5.50])
    check(close(sum(p), 1.0, 1e-6), "somma 1")
    check(all(v > 0 for v in p), "tutte positive")


def test_metrics():
    print("\n[8] Metriche")
    y = [0, 1, 2, 0]
    perfect = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    check(close(ranked_probability_score(y, perfect), 0.0, 1e-12), "RPS perfetto = 0")
    check(close(log_loss(y, perfect), 0.0, 1e-12), "log-loss perfetto = 0")
    check(close(brier_mse(y, perfect), 0.0, 1e-12), "MSE perfetto = 0")

    uniform = [[1 / 3, 1 / 3, 1 / 3]] * 4
    check(ranked_probability_score(y, uniform) > 0, "RPS uniforme > 0")
    check(close(log_loss(y, uniform), -__import__("math").log(1 / 3), 1e-9),
          "log-loss uniforme = ln(3)")
    check(close(brier_mse(y, uniform), 2 / 9, 1e-9), "MSE uniforme = 2/9")

    # RPS noto: y=0, p=[0.6,0.3,0.1] -> ((0.6-1)^2 + (0.9-1)^2 + (0.9-1)^2)/3
    expected = ((0.6 - 1) ** 2 + (0.9 - 1) ** 2 + (1.0 - 1) ** 2) / 3
    check(close(ranked_probability_score([0], [[0.6, 0.3, 0.1]]), expected, 1e-12),
          "RPS a mano corretto")

    # y=[0,0,1], pred=[0,0,0]: classe 0 ha TP=2, FP=1 -> P=2/3, R=1 -> F1=0.8
    f1 = f1_scores([0, 0, 1], [0, 0, 0])
    check(close(f1["per_class"][0]["f1"], 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0), 1e-9),
          f"F1 classe 0 = 0.8 ({f1['per_class'][0]['f1']:.4f})")
    check(f1["per_class"][1]["f1"] == 0.0, "F1 classe 1 = 0 (nessuna previsione)")
    check(close(f1["weighted"], (0.8 * 2 + 0.0 * 1) / 3, 1e-9),
          f"F1 weighted = 0.5333 ({f1['weighted']:.4f})")
    # Caso perfetto: y=[0,0,1], pred=[0,0,1] -> entrambe le classi predette F1=1
    f1b = f1_scores([0, 0, 1], [0, 0, 1])
    check(close(f1b["per_class"][0]["f1"], 1.0, 1e-9),
          f"F1 classe 0 perfetto = 1.0 ({f1b['per_class'][0]['f1']:.4f})")
    check(close(f1b["per_class"][1]["f1"], 1.0, 1e-9),
          f"F1 classe 1 perfetto = 1.0 ({f1b['per_class'][1]['f1']:.4f})")
    check(close(f1b["weighted"], 1.0, 1e-9),
          f"F1 weighted = 1.0 ({f1b['weighted']:.4f})")


def test_f1_matches_sklearn():
    print("\n[9] F1 contro sklearn (se installato)")
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        print("  SKIP sklearn non installato")
        return
    import random
    random.seed(7)
    y = [random.randint(0, 2) for _ in range(200)]
    p = [random.randint(0, 2) for _ in range(200)]
    mine = f1_scores(y, p)
    check(close(mine["macro"], f1_score(y, p, average="macro"), 1e-9),
          f"F1 macro ({mine['macro']:.6f}) == sklearn")
    check(close(mine["weighted"], f1_score(y, p, average="weighted"), 1e-9),
          f"F1 weighted ({mine['weighted']:.6f}) == sklearn")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST src/implied.py — de-vig e metriche")
    print("=" * 60)
    test_invalid_odds()
    test_against_reference()
    test_sum_to_one()
    test_overround()
    test_devig_direction()
    test_additive()
    test_power()
    test_odds_ratio()
    test_metrics()
    test_f1_matches_sklearn()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} TEST FALLITI:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("TUTTI I TEST PASSATI")
