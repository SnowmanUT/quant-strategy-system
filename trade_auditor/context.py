"""
Layer 3: context.py

STYLE CHECK: classify every entry using same-day bars STRICTLY BEFORE the
entry (no lookahead). Requires at least MIN_PRIOR_BARS prior bars, else the
trade is 'unclassified'.

Families: trend continuation, mean reversion, pullback, counter-trend
momentum, breakout. Two axes are also reported: momentum-vs-fade and
with-trend-vs-counter-trend.

style_check() compares the user's DECLARED family (from the strategy
library / AI path) to the observed majority family and produces a
match/mismatch verdict. In the per-family table, families with fewer than
GREY_OUT_MIN_TRADES trades are greyed out -- too few to read.

Thresholds are named constants, disclosed here, at the top of the file:
"""

import numpy as np

import metrics

MIN_PRIOR_BARS = 3          # fewer prior same-day bars than this -> 'unclassified'
GREY_OUT_MIN_TRADES = 20    # per-family table rows below this count are greyed out

FAMILIES = ["trend continuation", "mean reversion", "pullback", "counter-trend momentum", "breakout"]

# proximity-to-range thresholds used to call a close "near the high/low" of the prior bars
NEAR_EXTREME_HIGH = 0.80
NEAR_EXTREME_LOW = 0.20
MOMENTUM_LOOKBACK_BARS = 3  # how many of the most recent prior bars define "recent momentum"


def _prior_bars(instrument_bars, entry_time):
    """Bars for this instrument, same calendar day as entry_time, strictly before it."""
    day = entry_time.date()
    day_bars = instrument_bars[instrument_bars["datetime"].dt.date == day].sort_values("datetime")
    return day_bars[day_bars["datetime"] < entry_time].reset_index(drop=True)


def classify_entry(prior_bars, market_pos):
    """
    Classify a single entry from the bars strictly before it. This is a
    transparent, rule-based heuristic (not a statistical model) -- it exists
    to sanity-check a trader's self-described style against what the tape
    actually looked like going into their entries, not to be a definitive
    taxonomy of market structure.
    """
    if len(prior_bars) < MIN_PRIOR_BARS:
        return {"family": "unclassified", "momentum_axis": None, "trend_axis": None,
                "reason": f"fewer than {MIN_PRIOR_BARS} prior same-day bars"}

    closes = prior_bars["close"].tolist()
    highs = prior_bars["high"].tolist()
    lows = prior_bars["low"].tolist()

    trade_dir = 1 if str(market_pos).strip().title() == "Long" else -1

    trend_change = closes[-1] - closes[0]
    trend_dir = 1 if trend_change > 0 else (-1 if trend_change < 0 else 0)
    with_trend = None if trend_dir == 0 else (trade_dir == trend_dir)

    recent = closes[-MOMENTUM_LOOKBACK_BARS:] if len(closes) >= MOMENTUM_LOOKBACK_BARS else closes
    recent_change = recent[-1] - recent[0]
    momentum_dir = 1 if recent_change > 0 else (-1 if recent_change < 0 else 0)
    is_momentum_entry = None if momentum_dir == 0 else (momentum_dir == trade_dir)

    day_high, day_low = max(highs), min(lows)
    proximity = (closes[-1] - day_low) / (day_high - day_low) if day_high > day_low else 0.5
    near_high, near_low = proximity >= NEAR_EXTREME_HIGH, proximity <= NEAR_EXTREME_LOW
    at_extreme_favoring_trade = (near_high and trade_dir == 1) or (near_low and trade_dir == -1)
    at_extreme_against_trade = (near_high and trade_dir == -1) or (near_low and trade_dir == 1)

    if with_trend and is_momentum_entry and at_extreme_favoring_trade:
        family = "breakout"
    elif with_trend and is_momentum_entry:
        family = "trend continuation"
    elif with_trend and is_momentum_entry is False:
        family = "pullback"
    elif with_trend is False and at_extreme_against_trade:
        family = "mean reversion"
    elif with_trend is False and is_momentum_entry:
        family = "counter-trend momentum"
    else:
        family = "unclassified"

    return {
        "family": family,
        "momentum_axis": ("momentum" if is_momentum_entry else "fade") if is_momentum_entry is not None else None,
        "trend_axis": ("with-trend" if with_trend else "counter-trend") if with_trend is not None else None,
        "proximity_to_range_high": round(float(proximity), 3),
    }


def classify_all_trades(trades_df, bars_df):
    results = []
    for _, row in trades_df.iterrows():
        instrument_bars = bars_df[bars_df["instrument"] == row["Instrument"]]
        prior = _prior_bars(instrument_bars, row["Entry time"])
        c = classify_entry(prior, row["Market pos."])
        results.append({
            "trade_number": row.get("Trade number"),
            "instrument": row["Instrument"],
            "entry_time": row["Entry time"],
            **c,
        })
    return results


def style_check(trades_df, bars_df, declared_family=None):
    """
    Classify every trade, compare the observed majority family to the
    trader's declared family (if given), and build a per-family table with
    thin families greyed out.
    """
    if trades_df.empty:
        raise ValueError("No trades to style-check.")

    classifications = classify_all_trades(trades_df, bars_df)
    df = trades_df.copy().reset_index(drop=True)
    df["family"] = [c["family"] for c in classifications]

    classified = df[df["family"] != "unclassified"]
    family_counts = classified["family"].value_counts().to_dict()
    observed_majority = max(family_counts, key=family_counts.get) if family_counts else None

    verdict = None
    if declared_family:
        declared_norm = declared_family.strip().lower()
        verdict = "match" if (observed_majority and declared_norm == observed_majority) else "mismatch"

    per_family_table = {}
    for fam in FAMILIES:
        fam_trades = classified[classified["family"] == fam]
        n = len(fam_trades)
        profits = fam_trades["Profit"].tolist() if n else []
        per_family_table[fam] = {
            "n_trades": n,
            "win_rate": metrics.win_rate(profits) if n else None,
            "avg_profit": metrics.expectancy_dollars(profits) if n else None,
            "grey_out": n < GREY_OUT_MIN_TRADES,
        }

    momentum_axis_counts = {}
    trend_axis_counts = {}
    for c in classifications:
        if c["momentum_axis"]:
            momentum_axis_counts[c["momentum_axis"]] = momentum_axis_counts.get(c["momentum_axis"], 0) + 1
        if c["trend_axis"]:
            trend_axis_counts[c["trend_axis"]] = trend_axis_counts.get(c["trend_axis"], 0) + 1

    if observed_majority:
        plain = (
            f"Observed style is mostly '{observed_majority}' "
            f"({family_counts.get(observed_majority, 0)} of {len(classified)} classified trades)."
        )
        if declared_family:
            plain += f" Declared family was '{declared_family}' -- {verdict}."
    else:
        plain = "Not enough classified trades to determine a dominant style."

    findings = {
        "thresholds": {
            "min_prior_bars": MIN_PRIOR_BARS,
            "grey_out_min_trades": GREY_OUT_MIN_TRADES,
            "near_extreme_high": NEAR_EXTREME_HIGH,
            "near_extreme_low": NEAR_EXTREME_LOW,
            "momentum_lookback_bars": MOMENTUM_LOOKBACK_BARS,
        },
        "n_total": int(len(df)),
        "n_unclassified": int((df["family"] == "unclassified").sum()),
        "family_counts": family_counts,
        "observed_majority_family": observed_majority,
        "declared_family": declared_family,
        "verdict": verdict,
        "per_family_table": per_family_table,
        "axis_counts": {"momentum_vs_fade": momentum_axis_counts, "with_trend_vs_counter_trend": trend_axis_counts},
        "per_trade": classifications,
        "plain_english": plain,
    }
    return metrics.to_native(findings)


if __name__ == "__main__":
    import trades_io
    import bars as bars_mod
    trades = trades_io.load_trades("trades.csv")
    symbols, start, end = bars_mod.symbols_and_range_from_trades(trades)
    bars, warns = bars_mod.fetch_bars(symbols, start_date=start, end_date=end)
    result = style_check(trades, bars, declared_family="momentum")
    print(result["plain_english"])
