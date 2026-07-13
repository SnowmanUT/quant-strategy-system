"""
Layer 3: auditor_engine.py

BEHAVIORAL AUDIT, from the trade log alone (plus the declared rules, needed
only to define 1R and to check plan adherence -- no bar data required here).

audit_trades(trades_df, rules) writes everything into ONE findings dict that
the other layers (and the dashboard) reuse:
  - core_stats: win rate, avg win, avg loss, payoff, profit factor,
    expectancy in $ AND in R (1R = stop_pct x entry x qty) -- via metrics.py
  - luck: P&L without the top 1/3/5 trades, and concentration = the top
    decile's share of GROSS WINNING P&L, bounded 0-100%
  - revenge_sizing: avg size after a same-day loss vs otherwise, and every
    trade that exceeded the plan's max_size
  - disposition: avg hold time winners vs losers, MFE capture on winners
    (realized move / MFE), avg give-back (ETD)
  - pnl_by_hour / pnl_by_instrument
  - plan_adherence: direction / session window / max size, with 'Manual'
    exits reported as a separate info line -- a manual exit is NOT a
    violation by itself
"""

import math

import numpy as np
import pandas as pd

import metrics
import rules as rules_mod

TOP_N_LUCK_CHECKS = (1, 3, 5)
TOP_DECILE_FRACTION = 0.10


# --------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------

def _luck_analysis(trades_df):
    profits = trades_df["Profit"].astype(float).tolist()
    total = sum(profits)
    sorted_desc = sorted(profits, reverse=True)

    pnl_without_top_n = {}
    for n in TOP_N_LUCK_CHECKS:
        removed = sorted_desc[:n]
        pnl_without_top_n[n] = {
            "pnl_without": round(total - sum(removed), 2),
            "removed_dollars": round(sum(removed), 2),
        }

    winners = [p for p in profits if p > 0]
    gross_winning = sum(winners)
    concentration = None
    decile_n = 0
    if winners and gross_winning > 0:
        winners_sorted = sorted(winners, reverse=True)
        decile_n = max(1, math.ceil(len(winners_sorted) * TOP_DECILE_FRACTION))
        top_decile_sum = sum(winners_sorted[:decile_n])
        concentration = max(0.0, min(100.0, 100.0 * top_decile_sum / gross_winning))

    return {
        "total_pnl": round(total, 2),
        "pnl_without_top_n": pnl_without_top_n,
        "top_decile_trade_count": decile_n,
        "concentration_top_decile_pct_of_gross_winning": (round(concentration, 1) if concentration is not None else None),
        "plain_english": (
            f"The top {decile_n} winning trade(s) (top decile of winners) account for "
            f"{concentration:.0f}% of all gross winning P&L."
        ) if concentration is not None else "No winning trades to assess concentration.",
    }


def _revenge_sizing(trades_df, rules):
    df = trades_df.sort_values("Entry time").copy()
    df["entry_date"] = df["Entry time"].dt.date

    after_loss_qtys, otherwise_qtys = [], []
    for _, day_df in df.groupby("entry_date"):
        day_df = day_df.sort_values("Entry time")
        had_loss_today = False
        for _, row in day_df.iterrows():
            (after_loss_qtys if had_loss_today else otherwise_qtys).append(float(row["Qty"]))
            if row["Profit"] < 0:
                had_loss_today = True

    avg_after = float(np.mean(after_loss_qtys)) if after_loss_qtys else None
    avg_otherwise = float(np.mean(otherwise_qtys)) if otherwise_qtys else None

    max_size = rules.get("max_size")
    oversized = df[df["Qty"] > max_size] if max_size else df.iloc[0:0]
    oversized_list = [
        {
            "trade_number": r.get("Trade number"),
            "instrument": r["Instrument"],
            "entry_time": r["Entry time"],
            "qty": float(r["Qty"]),
            "max_size": max_size,
        }
        for _, r in oversized.iterrows()
    ]

    if avg_after is not None and avg_otherwise is not None:
        plain = (
            f"Average size after a same-day loss was {avg_after:.1f} vs {avg_otherwise:.1f} otherwise. "
            f"{len(oversized_list)} trade(s) exceeded the plan's max size of {max_size}."
        )
    else:
        plain = "Not enough same-day sequences to assess revenge sizing."

    return {
        "avg_qty_after_same_day_loss": avg_after,
        "avg_qty_otherwise": avg_otherwise,
        "oversized_trades": oversized_list,
        "oversized_count": len(oversized_list),
        "plain_english": plain,
    }


def _disposition(trades_df):
    df = trades_df.copy()
    df["hold_minutes"] = (df["Exit time"] - df["Entry time"]).dt.total_seconds() / 60.0
    winners = df[df["Profit"] > 0]
    losers = df[df["Profit"] < 0]

    avg_hold_win = float(winners["hold_minutes"].mean()) if len(winners) else None
    avg_hold_loss = float(losers["hold_minutes"].mean()) if len(losers) else None

    mfe_capture = None
    if "MFE" in df.columns and df["MFE"].notna().any():
        w = winners[winners["MFE"].notna() & (winners["MFE"] != 0)]
        if len(w):
            realized_move = (w["Exit price"] - w["Entry price"]).abs()
            mfe_capture = float((realized_move / w["MFE"].abs()).mean())

    avg_giveback = None
    if "ETD" in df.columns and df["ETD"].notna().any():
        avg_giveback = float(df["ETD"].dropna().mean())

    if avg_hold_win is not None and avg_hold_loss is not None:
        plain = f"Winners are held {avg_hold_win:.0f} min on average vs {avg_hold_loss:.0f} min for losers."
    else:
        plain = "Not enough winners/losers to compare hold times."

    return {
        "avg_hold_minutes_winners": avg_hold_win,
        "avg_hold_minutes_losers": avg_hold_loss,
        "mfe_capture_ratio_winners": mfe_capture,
        "avg_give_back_etd": avg_giveback,
        "plain_english": plain,
    }


def _pnl_by_hour(trades_df):
    df = trades_df.copy()
    df["entry_hour"] = df["Entry time"].dt.hour
    g = df.groupby("entry_hour")["Profit"].agg(["sum", "count", "mean"])
    return {int(hour): {"sum": round(float(v["sum"]), 2), "count": int(v["count"]), "mean": round(float(v["mean"]), 2)}
            for hour, v in g.iterrows()}


def _pnl_by_instrument(trades_df):
    out = {}
    for instrument, g in trades_df.groupby("Instrument"):
        profits = g["Profit"].tolist()
        out[instrument] = {
            "sum": round(float(sum(profits)), 2),
            "count": int(len(profits)),
            "mean": round(float(np.mean(profits)), 2),
            "win_rate": metrics.win_rate(profits),
        }
    return out


def _plan_adherence(trades_df, rules):
    df = trades_df.copy()
    direction_ok = df.apply(lambda r: rules_mod.direction_matches(r["Market pos."], rules["direction"]), axis=1)
    session_ok = df["Entry time"].apply(lambda t: rules_mod.in_session(t, rules["session_start"], rules["session_end"]))
    size_ok = df["Qty"].apply(lambda q: rules_mod.size_within_plan(q, rules["max_size"]))
    violation_mask = ~(direction_ok & session_ok & size_ok)
    violations = df[violation_mask]

    if "Exit name" in df.columns:
        manual_mask = df["Exit name"].astype(str).str.strip().str.lower() == "manual"
        manual_exits = df[manual_mask]
    else:
        manual_exits = df.iloc[0:0]

    n = len(df)
    adherence_rate = float((n - len(violations)) / n) if n else None

    violation_records = [
        {
            "trade_number": r.get("Trade number"), "instrument": r["Instrument"], "entry_time": r["Entry time"],
            "market_pos": r["Market pos."], "qty": float(r["Qty"]),
        }
        for _, r in violations.iterrows()
    ]
    manual_records = [
        {"trade_number": r.get("Trade number"), "instrument": r["Instrument"], "exit_time": r["Exit time"]}
        for _, r in manual_exits.iterrows()
    ]

    plain = (
        f"{adherence_rate:.0%} of trades matched the plan (direction/session/size). "
        f"{len(manual_records)} trade(s) were closed manually -- informational only, not a violation."
    ) if adherence_rate is not None else "No trades to check against the plan."

    return {
        "n_trades": n,
        "adherence_rate": adherence_rate,
        "direction_violations": int((~direction_ok).sum()),
        "session_violations": int((~session_ok).sum()),
        "size_violations": int((~size_ok).sum()),
        "violations": violation_records,
        "manual_exit_info": {
            "count": len(manual_records),
            "note": "Manual exits are informational only -- a manual exit is not a plan violation by itself.",
            "trades": manual_records,
        },
        "plain_english": plain,
    }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def audit_trades(trades_df, rules):
    """
    Run the full behavioral audit on a loaded trade log (trades_io.load_trades
    output) against the declared rules (rules.py schema). Returns one findings
    dict -- the shared artifact the dashboard and other Layer 3 modules reuse.
    """
    if trades_df.empty:
        raise ValueError("No trades to audit.")

    profits = trades_df["Profit"].astype(float).tolist()
    entries = trades_df["Entry price"].astype(float).tolist()
    qtys = trades_df["Qty"].astype(float).tolist()
    rs = metrics.r_multiples_for_trades(profits, entries, qtys, rules["stop_pct"])

    findings = {
        "core_stats": metrics.compute_core_stats(profits, rs),
        "quant_stats": metrics.compute_quant_stats(profits, rs),
        "luck": _luck_analysis(trades_df),
        "revenge_sizing": _revenge_sizing(trades_df, rules),
        "disposition": _disposition(trades_df),
        "pnl_by_hour": _pnl_by_hour(trades_df),
        "pnl_by_instrument": _pnl_by_instrument(trades_df),
        "plan_adherence": _plan_adherence(trades_df, rules),
    }
    return metrics.to_native(findings)


if __name__ == "__main__":
    import trades_io
    df = trades_io.load_trades("trades.csv")
    active_rules = rules_mod.load_rules()
    findings = audit_trades(df, active_rules)
    print("Core stats:", findings["core_stats"]["plain_english"])
    print("Plan adherence:", findings["plan_adherence"]["plain_english"])
