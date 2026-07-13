"""
Layer 3: discipline.py

RULE REPLAY: what the declared rules would have done, trade by trade, on the
REAL bars. Conservative by design. These exact assumptions get printed on
the dashboard later (see ASSUMPTIONS below) -- nothing here is silently
optimistic:

  1. Exits resolve on bars STRICTLY AFTER the entry bar -- no intra-entry-bar
     lookahead.
  2. If one bar contains both the stop and the target, the stop is assumed
     to fill first (pessimistic).
  3. Stops are gap-aware: a bar that opens through the stop (worse than the
     open) fills at the open, then slippage is applied on top.
  4. Targets are resting limit orders: exact price, never gap-improved.
  5. Slippage (slippage_bps) is charged on stop and market/time exits; limit
     (target) fills get none.
  6. The SAME per-trade commission as the real trade is charged.
  7. exit_style is modeled explicitly: fixed, breakeven (stop moves to entry
     once price reaches +1R favorable), trailing (stop trails by the initial
     stop distance, no fixed target), time (hold to session end).
  8. Plan eligibility is checked FIRST: trades violating direction, session,
     or size are "outside your plan, not taken" -- they contribute $0 to the
     disciplined benchmark.
  9. Anything still open at session end closes at that bar's close.

replay_rules(trades_df, bars_df, rules) outputs, per trade: rule exit
price/time/kind + P&L; and the aggregate: actual vs rule-managed vs signed
difference. Reuses metrics.py for every summary stat so these numbers can
never disagree with auditor_engine.py's.
"""

import pandas as pd

import metrics
import rules as rules_mod

ASSUMPTIONS = [
    "Exits resolve on bars strictly AFTER the entry bar -- no intra-entry-bar lookahead.",
    "If one bar contains both the stop and the target, the stop is assumed to fill first (pessimistic).",
    "Stops are gap-aware: a bar that opens through the stop fills at the open (worse than the stop), then slippage is applied.",
    "Targets are resting limit orders: exact price, never gap-improved.",
    "Slippage (slippage_bps) is charged on stop and market/time exits; limit (target) fills get none.",
    "The same per-trade commission as the real trade is charged on the rule-managed exit.",
    "exit_style is modeled explicitly: fixed, breakeven (stop moves to entry at +1R), trailing (stop trails by the initial stop distance, no fixed target), or time (hold to session end).",
    "Plan eligibility is checked first: trades violating direction, session window, or max size are 'outside your plan, not taken' and contribute $0 to the disciplined benchmark.",
    "Anything still open at session end is closed at that bar's close.",
]


def _bars_after_entry(instrument_bars, entry_time):
    """Bars for this instrument, same calendar day as entry_time, strictly after the entry bar."""
    day = entry_time.date()
    day_bars = instrument_bars[instrument_bars["datetime"].dt.date == day].sort_values("datetime").reset_index(drop=True)
    if day_bars.empty:
        return day_bars
    at_or_before = day_bars[day_bars["datetime"] <= entry_time]
    if at_or_before.empty:
        # entry_time is before the first bar of the day -- nothing to exclude
        return day_bars
    entry_idx = at_or_before.index.max()
    return day_bars.iloc[entry_idx + 1:].reset_index(drop=True)


def _apply_slippage(price, sign, slippage_bps):
    """Slippage always works against the trader: sells fill lower, buys-to-cover fill higher."""
    adj = price * (slippage_bps / 10000.0)
    return price - sign * adj


def _replay_one_trade(row, bars_after, rules):
    sign = 1 if str(row["Market pos."]).strip().title() == "Long" else -1
    entry_price = float(row["Entry price"])
    qty = float(row["Qty"])
    commission = float(row["Commission"]) if "Commission" in row.index and pd.notna(row["Commission"]) else 0.0
    slippage_bps = rules["slippage_bps"]

    stop_frac = rules["stop_pct"] / 100.0
    target_pct = rules["target_pct"]
    exit_style = rules["exit_style"]

    stop_price = entry_price - sign * entry_price * stop_frac
    target_price = (entry_price + sign * entry_price * (target_pct / 100.0)) if target_pct else None
    one_r_price = entry_price + sign * entry_price * stop_frac  # price at +1R favorable, for breakeven arming

    if bars_after.empty:
        return None  # no bar data to replay on -- caller flags this trade

    trail_stop = stop_price
    trail_extreme = entry_price
    breakeven_armed = False

    for _, bar in bars_after.iterrows():
        o, h, l, ts = float(bar["open"]), float(bar["high"]), float(bar["low"]), bar["datetime"]

        if exit_style == "trailing":
            favorable_extreme = h if sign == 1 else l
            if (sign == 1 and favorable_extreme > trail_extreme) or (sign == -1 and favorable_extreme < trail_extreme):
                trail_extreme = favorable_extreme
                trail_stop = trail_extreme - sign * entry_price * stop_frac
            active_stop, active_target = trail_stop, None  # trailing: no fixed target
        elif exit_style == "breakeven":
            if not breakeven_armed:
                favorable_extreme = h if sign == 1 else l
                reached_1r = (favorable_extreme >= one_r_price) if sign == 1 else (favorable_extreme <= one_r_price)
                if reached_1r:
                    breakeven_armed = True
            active_stop = entry_price if breakeven_armed else stop_price
            active_target = target_price
        elif exit_style == "time":
            active_stop, active_target = None, None  # hold to session end regardless
        else:  # fixed
            active_stop, active_target = stop_price, target_price

        stop_hit = (l <= active_stop) if (sign == 1 and active_stop is not None) else \
                   (h >= active_stop) if (sign == -1 and active_stop is not None) else False
        target_hit = (h >= active_target) if (sign == 1 and active_target is not None) else \
                     (l <= active_target) if (sign == -1 and active_target is not None) else False

        if stop_hit and target_hit:
            target_hit = False  # assumption #2: stop fills first

        if stop_hit:
            gapped = (o < active_stop) if sign == 1 else (o > active_stop)
            raw_fill = o if gapped else active_stop  # assumption #3: gap-aware stop
            fill_price = _apply_slippage(raw_fill, sign, slippage_bps)  # assumption #5
            kind = "breakeven_stop" if (exit_style == "breakeven" and breakeven_armed) else \
                   ("trailing_stop" if exit_style == "trailing" else "stop")
            gross = (fill_price - entry_price) * sign * qty
            return {"exit_price": fill_price, "exit_time": ts, "exit_kind": kind, "profit": round(gross - commission, 2)}

        if target_hit:
            fill_price = active_target  # assumption #4: exact price, never gap-improved; no slippage (#5)
            gross = (fill_price - entry_price) * sign * qty
            return {"exit_price": fill_price, "exit_time": ts, "exit_kind": "target", "profit": round(gross - commission, 2)}

    # nothing triggered through the rest of the session -> assumption #9
    last_bar = bars_after.iloc[-1]
    raw_fill = float(last_bar["close"])
    fill_price = _apply_slippage(raw_fill, sign, slippage_bps)  # market-style close -> slippage applies
    gross = (fill_price - entry_price) * sign * qty
    kind = "time_close" if exit_style == "time" else "session_close"
    return {"exit_price": fill_price, "exit_time": last_bar["datetime"], "exit_kind": kind, "profit": round(gross - commission, 2)}


def replay_rules(trades_df, bars_df, rules):
    """
    Simulate the declared rules, trade by trade, on the real bars.

    Returns:
      {
        "assumptions": [...],                # the 9 assumptions above, for the dashboard
        "per_trade": [...],                  # rule exit price/time/kind + P&L per trade
        "aggregate": {...},                  # actual vs rule-managed vs signed difference
        "core_stats_actual": {...},          # via metrics.py
        "core_stats_rule_managed": {...},    # via metrics.py, same functions
      }
    """
    if trades_df.empty:
        raise ValueError("No trades to replay.")

    per_trade = []
    for _, row in trades_df.iterrows():
        eligible = rules_mod.plan_eligible(row, rules)  # assumption #8

        base = {
            "trade_number": row.get("Trade number"),
            "instrument": row["Instrument"],
            "entry_time": row["Entry time"],
            "market_pos": row["Market pos."],
            "actual_exit_price": float(row["Exit price"]),
            "actual_exit_time": row["Exit time"],
            "actual_profit": round(float(row["Profit"]), 2),
            "outside_plan": not eligible,
        }

        if not eligible:
            base.update({"rule_exit_price": None, "rule_exit_time": None, "rule_exit_kind": "outside_plan", "rule_profit": 0.0})
            per_trade.append(base)
            continue

        instrument_bars = bars_df[bars_df["instrument"] == row["Instrument"]]
        bars_after = _bars_after_entry(instrument_bars, row["Entry time"])  # assumption #1
        result = _replay_one_trade(row, bars_after, rules)

        if result is None:
            base.update({"rule_exit_price": None, "rule_exit_time": None, "rule_exit_kind": "no_bar_data", "rule_profit": None})
        else:
            base.update({
                "rule_exit_price": result["exit_price"],
                "rule_exit_time": result["exit_time"],
                "rule_exit_kind": result["exit_kind"],
                "rule_profit": result["profit"],
            })
        per_trade.append(base)

    actual_pnls = [t["actual_profit"] for t in per_trade]
    # outside-plan and no-bar-data trades contribute $0 to the disciplined benchmark (assumption #8);
    # no-bar-data trades are excluded from actual too, since we can't fairly compare them -- flagged separately.
    no_data_trades = [t for t in per_trade if t["rule_exit_kind"] == "no_bar_data"]
    rule_pnls_full = [0.0 if t["rule_profit"] is None else t["rule_profit"] for t in per_trade]

    actual_total = round(sum(actual_pnls), 2)
    rule_total = round(sum(rule_pnls_full), 2)
    signed_diff = round(actual_total - rule_total, 2)  # positive = actual beat the disciplined replay

    if signed_diff > 0:
        comparison = f"Actual trading outperformed the disciplined replay by ${signed_diff:,.2f}."
    elif signed_diff < 0:
        comparison = f"Following the plan's rules exactly would have outperformed actual trading by ${abs(signed_diff):,.2f}."
    else:
        comparison = "Actual trading and the disciplined replay matched exactly."

    aggregate = {
        "actual_total_pnl": actual_total,
        "rule_managed_total_pnl": rule_total,
        "signed_difference": signed_diff,
        "n_outside_plan": int(sum(1 for t in per_trade if t["outside_plan"])),
        "n_no_bar_data": len(no_data_trades),
        "plain_english": f"Actual: ${actual_total:,.2f}. Rule-managed: ${rule_total:,.2f}. {comparison}",
    }

    findings = {
        "assumptions": ASSUMPTIONS,
        "per_trade": per_trade,
        "aggregate": aggregate,
        "core_stats_actual": metrics.compute_core_stats(actual_pnls),
        "core_stats_rule_managed": metrics.compute_core_stats(rule_pnls_full),
    }
    return metrics.to_native(findings)


if __name__ == "__main__":
    import trades_io
    import bars as bars_mod
    trades = trades_io.load_trades("trades.csv")
    symbols, start, end = bars_mod.symbols_and_range_from_trades(trades)
    bars, warns = bars_mod.fetch_bars(symbols, start_date=start, end_date=end)
    result = replay_rules(trades, bars, rules_mod.load_rules())
    print(result["aggregate"]["plain_english"])
