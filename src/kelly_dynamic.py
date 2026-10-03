# src/kelly_dynamic.py
"""
Fractional Kelly Dinamico — unica fonte di verita' del sizing.

Sostituisce il Kelly fisso (KELLY_FRACTION = 0.25) con una frazione che
dipende dalla quota O, attenuata per scaglioni e limitata da un hard cap.

    b           = O - 1                      (payoff netto)
    f_full      = (p*b - (1-p)) / b           (Kelly pieno)
    c_dynamic   = 0.50 | 0.35 | 0.20 | 0.10   (scaglioni rigidi su O)
    f_dynamic   = f_full * c_dynamic
    stake       = 0 se f_dynamic <= 0, altrimenti min(f_dynamic, MAX_CAP)
                  MAX_CAP = 0.05  per O <= 3.00
                  MAX_CAP = 0.025 per O >  3.00

Perche' la attenuazione dipende dalla quota: la varianza del ritorno di una
scommessa e' b^2 * p * (1-p), quindi cresce con il payoff. Su una quota 6.00
la stessa probabilita' di errore fa oscillare il bankroll sei volte tanto
che su una 1.50. c_dynamic tiene la quota (quindi il rischio) dentro una
frazione costante del pieno, e il cap taglia le code in cui la stima di p
e' ottimista.

Modulo puro: nessuna I/O, nessuna dipendenza, importabile da tutta la
pipeline (predict_fixtures, evaluate, refresh_predictions, recompute_22era).
"""
from types import MappingProxyType
from typing import Any, Mapping

# Soglie e moltiplicatori: oggetto costante immutabile (MappingProxyType,
# che solleva TypeError su qualsiasi tentativo di modifica a runtime).
KELLY_DYNAMIC_CONFIG: Mapping[str, Any] = MappingProxyType({
    # Soglie di quota, in ordine crescente.
    "odds_thresholds": (1.80, 3.00, 5.00),
    # c_dynamic per gli scaglioni definiti dalle soglie (4 soglie, 4 valori).
    "c_dynamic": (0.50, 0.35, 0.20, 0.10),
    # Soglia del cap + cap: O <= 3.00 -> 0.05, O > 3.00 -> 0.025.
    "cap_odds_threshold": 3.00,
    "max_caps": (0.05, 0.025),
})

_ODDS_T_1 = float(KELLY_DYNAMIC_CONFIG["odds_thresholds"][0])   # 1.80
_ODDS_T_2 = float(KELLY_DYNAMIC_CONFIG["odds_thresholds"][1])   # 3.00
_ODDS_T_3 = float(KELLY_DYNAMIC_CONFIG["odds_thresholds"][2])   # 5.00
_C_LOW = float(KELLY_DYNAMIC_CONFIG["c_dynamic"][0])            # 0.50
_C_MID = float(KELLY_DYNAMIC_CONFIG["c_dynamic"][1])           # 0.35
_C_HIGH = float(KELLY_DYNAMIC_CONFIG["c_dynamic"][2])          # 0.20
_C_TOP = float(KELLY_DYNAMIC_CONFIG["c_dynamic"][3])           # 0.10
_CAP_LOW = float(KELLY_DYNAMIC_CONFIG["max_caps"][0])          # 0.05
_CAP_HIGH = float(KELLY_DYNAMIC_CONFIG["max_caps"][1])         # 0.025


def dynamic_kelly_multiplier(odds: float) -> float:
    """c_dynamic in funzione della quota decimale (scaglioni rigidi)."""
    o = float(odds)
    if o <= _ODDS_T_1:
        return _C_LOW
    if o <= _ODDS_T_2:
        return _C_MID
    if o <= _ODDS_T_3:
        return _C_HIGH
    return _C_TOP


def dynamic_kelly_max_cap(odds: float) -> float:
    """Hard cap di stake_fraction per quella quota."""
    return _CAP_LOW if float(odds) <= _ODDS_T_2 else _CAP_HIGH


def calculate_dynamic_kelly_stake(p: float, odds: float) -> float:
    """
    Frazione di bankroll da puntare (0 <= ritorno <= cap).

    p    probabilita' stimata del modello, 0 < p < 1
    odds quota decimale, odds > 1
    """
    try:
        p = float(p)
        o = float(odds)
    except (TypeError, ValueError):
        return 0.0
    b = o - 1.0
    # Quota non scommissibile o stima fuori dominio: nessuna puntata.
    if not (b > 0.0) or not (0.0 < p < 1.0):
        return 0.0
    f_full = (p * b - (1.0 - p)) / b
    f_dynamic = f_full * dynamic_kelly_multiplier(o)
    if f_dynamic <= 0.0:
        return 0.0
    return min(f_dynamic, dynamic_kelly_max_cap(o))


def dynamic_kelly_breakdown(p: float, odds: float) -> dict:
    """Stessa funzione, ma espone i passaggi per UI, JSON e test."""
    try:
        p = float(p)
        o = float(odds)
    except (TypeError, ValueError):
        return {"p": None, "odds": None, "b": None, "f_full": None,
                "c_dynamic": None, "f_dynamic": None, "max_cap": None,
                "capped": False, "stake_fraction": 0.0}
    b = o - 1.0
    if not (b > 0.0) or not (0.0 < p < 1.0):
        return {"p": p, "odds": o, "b": b, "f_full": None, "c_dynamic": None,
                "f_dynamic": None, "max_cap": None, "capped": False,
                "stake_fraction": 0.0}
    f_full = (p * b - (1.0 - p)) / b
    c = dynamic_kelly_multiplier(o)
    f_dynamic = f_full * c
    cap = dynamic_kelly_max_cap(o)
    capped = f_dynamic > cap
    stake = 0.0 if f_dynamic <= 0.0 else min(f_dynamic, cap)
    return {"p": round(p, 6), "odds": round(o, 4), "b": round(b, 6),
            "f_full": round(f_full, 6), "c_dynamic": c,
            "f_dynamic": round(f_dynamic, 6), "max_cap": cap,
            "capped": bool(capped), "stake_fraction": round(stake, 6)}


# Alias mantenuto per i chiamanti che parlano di "frazione" e non di stake:
# il valore e' identico, cambia solo il nome.
def kelly_fraction(p_model: float, odds: float) -> float:
    return calculate_dynamic_kelly_stake(p_model, odds)
