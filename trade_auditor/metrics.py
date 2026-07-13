"""
Layer 3: metrics.py

The shared quant-stats library. Both auditor_engine.py (behavioral audit of
the actual trade log) and discipline.py (rule replay on the real bars) call
into THIS module for every number they report, so the dashboard and the
engine can never disagree on a number.

Every function operates on plain lists of per-trade P&L (dollars) and/or
per-trade R-multiples -- no dataframe or I/O dependency, so they're trivial
to reuse and to unit test.

Named, disclosed thresholds live at the top of the file.
"""

import math

import numpy as np

# --- named, disclosed thresholds -------------------------------------------------
SQN_HARD_TO_TRADE_THRESHOLD = 1.6   # Van Tharp: SQN below this -> hard to trade
SQN_GOOD_THRESHOLD = 2.5            # Van Tharp: SQN above this -> good
CONFIDENCE_Z = 1.96                 # ~95% two-sided z-score for "distinguishable from zero"


def to_native(obj):
    """Recursively convert numpy/pandas scalars into plain JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "isoformat"):  # pandas.Timestamp / datetime
        return obj.isoformat()
    return obj


def _clean(values):
    out = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out.append(fv)
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------
# Core behavioral stats
# --------------------------------------------------------------------------

def win_rate(pnls):
    arr = _clean(pnls)
    if len(arr) == 0:
        return None
    return float((arr > 0).sum() / len(arr))


def avg_win(pnls):
    arr = _clean(pnls)
    wins = arr[arr > 0]
    if len(wins) == 0:
        return None
    return float(wins.mean())


def avg_loss(pnls):
    """Average losing trade, returned as a negative number."""
    arr = _clean(pnls)
    losses = arr[arr < 0]
    if len(losses) == 0:
        return None
    return float(losses.mean())


def payoff_ratio(pnls):
    aw, al = avg_win(pnls), avg_loss(pnls)
    if aw is None or al is None or al == 0:
        return None
    return float(aw / abs(al))


def profit_factor(pnls):
    arr = _clean(pnls)
    if len(arr) == 0:
        return None
    gross_win = arr[arr > 0].sum()
    gross_loss = arr[arr < 0].sum()
    if gross_loss == 0:
        return None
    return float(gross_win / abs(gross_loss))


def expectancy_dollars(pnls):
    arr = _clean(pnls)
    if len(arr) == 0:
        return None
    return float(arr.mean())


def r_multiple(profit, entry_price, qty, stop_pct):
    """1R = stop_pct% x entry_price x qty -- the dollar risk the plan assigned to the trade."""
    if profit is None or entry_price is None or qty is None or stop_pct is None:
        return None
    risk = abs(float(entry_price)) * (abs(float(stop_pct)) / 100.0) * abs(float(qty))
    if risk <= 0:
        return None
    return float(profit) / risk


def r_multiples_for_trades(profits, entry_prices, qtys, stop_pct):
    return [r_multiple(p, e, q, stop_pct) for p, e, q in zip(profits, entry_prices, qtys)]


def expectancy_r(rs):
    arr = _clean(rs)
    if len(arr) == 0:
        return None
    return float(arr.mean())


# --------------------------------------------------------------------------
# Quant stats
# --------------------------------------------------------------------------

def standard_error(values):
    arr = _clean(values)
    if len(arr) < 2:
        return None
    return float(arr.std(ddof=1) / math.sqrt(len(arr)))


def expectancy_with_se(values, unit_label="units"):
    """
    Mean +/- standard error, with a plain-English read on whether the mean is
    distinguishable from zero at roughly a 95% confidence level.
    """
    arr = _clean(values)
    if len(arr) < 2:
        return {
            "mean": (float(arr.mean()) if len(arr) else None),
            "se": None,
            "distinguishable_from_zero": None,
            "plain_english": "Not enough trades yet to estimate a standard error.",
        }
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    if se == 0:
        return {
            "mean": mean, "se": 0.0, "distinguishable_from_zero": True if mean != 0 else False,
            "plain_english": f"Expectancy is {mean:+.4g} {unit_label} with zero variance across trades -- unusual, check the data.",
        }
    distinguishable = abs(mean) > CONFIDENCE_Z * se
    direction = "positive" if mean > 0 else "negative"
    if distinguishable:
        plain = (
            f"Expectancy is {mean:+.4g} {unit_label} (+/- {se:.4g} SE) -- "
            f"{direction} and statistically distinguishable from zero at ~95% confidence."
        )
    else:
        plain = (
            f"Expectancy is {mean:+.4g} {unit_label} (+/- {se:.4g} SE) -- "
            "not statistically distinguishable from zero yet; treat the edge as unproven."
        )
    return {"mean": mean, "se": se, "distinguishable_from_zero": bool(distinguishable), "plain_english": plain}


def sqn(rs):
    """System Quality Number = sqrt(N) x mean(R) / std(R)."""
    arr = _clean(rs)
    if len(arr) < 2:
        return None
    std = arr.std(ddof=1)
    if std == 0:
        return None
    return float(math.sqrt(len(arr)) * arr.mean() / std)


def sqn_label(value):
    """Van Tharp-style read, anchored on the two disclosed thresholds."""
    if value is None:
        return "Not enough trades to compute SQN."
    if value < SQN_HARD_TO_TRADE_THRESHOLD:
        band = f"hard to trade (below {SQN_HARD_TO_TRADE_THRESHOLD:g})"
    elif value <= SQN_GOOD_THRESHOLD:
        band = f"tradable / average (between {SQN_HARD_TO_TRADE_THRESHOLD:g} and {SQN_GOOD_THRESHOLD:g})"
    else:
        band = f"good (above {SQN_GOOD_THRESHOLD:g})"
    return f"SQN {value:.2f} -- {band}. SQN measures how reliable the edge is, not how big it is."


def sharpe_per_trade(pnls):
    """Mean / std of per-trade P&L. Not annualized -- a per-trade consistency measure."""
    arr = _clean(pnls)
    if len(arr) < 2:
        return None
    std = arr.std(ddof=1)
    if std == 0:
        return None
    return float(arr.mean() / std)


def max_drawdown(pnls):
    """Max peak-to-trough decline on the cumulative P&L curve, in the order the trades are given."""
    arr = _clean(pnls)
    if len(arr) == 0:
        return None
    equity = np.cumsum(arr)
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    return float(drawdown.max())


def kelly_fraction(win_rate_value, payoff):
    """Kelly = W - (1-W)/payoff, floored at zero."""
    if win_rate_value is None or payoff is None or payoff <= 0:
        return 0.0
    k = win_rate_value - (1 - win_rate_value) / payoff
    return float(max(0.0, k))


# --------------------------------------------------------------------------
# Bundled reports -- the single source of truth auditor_engine.py and
# discipline.py both call, so their numbers can never disagree.
# --------------------------------------------------------------------------

def compute_core_stats(pnls, rs=None):
    """win rate, avg win, avg loss, payoff, profit factor, expectancy in $ (and R if rs given)."""
    wr = win_rate(pnls)
    aw = avg_win(pnls)
    al = avg_loss(pnls)
    pay = payoff_ratio(pnls)
    pf = profit_factor(pnls)
    exp_d = expectancy_dollars(pnls)

    if wr is None:
        plain = "No trades to summarize."
    else:
        plain = (
            f"{wr:.0%} win rate, average winner "
            f"{('$%.2f' % aw) if aw is not None else 'n/a'}, average loser "
            f"{('$%.2f' % al) if al is not None else 'n/a'}, expectancy "
            f"{('$%.2f' % exp_d) if exp_d is not None else 'n/a'} per trade."
        )

    out = {
        "n_trades": int(len(_clean(pnls))),
        "win_rate": wr,
        "avg_win": aw,
        "avg_loss": al,
        "payoff_ratio": pay,
        "profit_factor": pf,
        "expectancy_dollars": exp_d,
        "plain_english": plain,
    }
    if rs is not None:
        out["expectancy_r"] = expectancy_r(rs)
    return out


def compute_quant_stats(pnls, rs):
    """expectancy +/- SE (R and $), SQN (R), Sharpe per trade ($), payoff, max drawdown ($), Kelly."""
    exp_r_se = expectancy_with_se(rs, unit_label="R")
    exp_d_se = expectancy_with_se(pnls, unit_label="$")
    sqn_val = sqn(rs)
    sharpe = sharpe_per_trade(pnls)
    pay = payoff_ratio(pnls)
    dd = max_drawdown(pnls)
    wr = win_rate(pnls)
    kelly = kelly_fraction(wr, pay)

    return {
        "expectancy_r": exp_r_se,
        "expectancy_dollars": exp_d_se,
        "sqn": {"value": sqn_val, "plain_english": sqn_label(sqn_val)},
        "sharpe_per_trade": {
            "value": sharpe,
            "plain_english": (
                f"Sharpe per trade is {sharpe:.2f} -- steadiness of the P&L stream, not its size; "
                "higher means less noisy, not necessarily more profitable."
            ) if sharpe is not None else "Not enough trades to compute Sharpe.",
        },
        "payoff_ratio": {
            "value": pay,
            "plain_english": (
                f"Winners average {pay:.2f}x the size of losers." if pay is not None else "No losers yet to compare winners against."
            ),
        },
        "max_drawdown_dollars": {
            "value": dd,
            "plain_english": (
                f"Worst peak-to-trough decline across the trade sequence was ${dd:,.2f}." if dd is not None else "Not enough trades to compute drawdown."
            ),
        },
        "kelly_fraction": {
            "value": kelly,
            "plain_english": (
                f"Kelly sizing suggests about {kelly:.1%} of capital per trade at this win rate and payoff "
                "(floored at 0 -- if it hits 0, the math says this edge isn't strong enough to size into at all)."
            ),
        },
    }
