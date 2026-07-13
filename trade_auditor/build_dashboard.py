"""
Layer 4: build_dashboard.py

Runs the deterministic engine (Layer 3: auditor_engine, discipline, context)
on top of the data layer (Layer 1: trades_io, bars) and the active rules
(Layer 2: rules), and computes everything the dashboard needs into a single
dash_data.json. Then renders dashboard.html by embedding that same JSON
into dashboard_template.html, so the page works when opened directly
(file://) with no local server required -- dash_data.json is still written
separately as a plain, inspectable artifact.

Usage:
    python build_dashboard.py --trades trades.csv [--strategy "My Strategy"]

Everything downstream of this file is deterministic (cyan). The only
optional AI (purple) piece is the Coach's Read panel, which the dashboard
wires up as a stub POST to /api/coach for whenever a backend is listening --
this script doesn't call the AI client itself.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import auditor_engine
import bars as bars_mod
import context
import discipline
import metrics
import rules as rules_mod
import trades_io

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEMPLATE_PATH = os.path.join(THIS_DIR, "dashboard_template.html")

SPARKLINE_MAX_POINTS = 60
DASH_DATA_PLACEHOLDER = "/*__DASH_DATA_JSON__*/null"


# --------------------------------------------------------------------------
# Combined per-trade records (shared by trade replay, constellation, modal)
# --------------------------------------------------------------------------

def _build_combined_trades(trades_df, replay_per_trade, style_per_trade, rules):
    replay_by_num = {t["trade_number"]: t for t in replay_per_trade}
    style_by_num = {t["trade_number"]: t for t in style_per_trade}
    max_size = rules["max_size"]
    stop_pct = rules["stop_pct"]

    combined = []
    for _, row in trades_df.iterrows():
        tn = row["Trade number"]
        r = replay_by_num.get(tn, {})
        s = style_by_num.get(tn, {})
        entry_time = row["Entry time"]
        instrument = row["Instrument"]
        profit = float(row["Profit"])
        qty = float(row["Qty"])
        r_mult = metrics.r_multiple(profit, float(row["Entry price"]), qty, stop_pct)
        outcome = "win" if profit > 0 else ("loss" if profit < 0 else "scratch")
        is_manual = str(row.get("Exit name", "") or "").strip().lower() == "manual"

        combined.append({
            "trade_number": tn,
            "instrument": instrument,
            "side": row["Market pos."],
            "qty": qty,
            "entry_time": entry_time,
            "entry_price": float(row["Entry price"]),
            "exit_time": row["Exit time"],
            "exit_price": float(row["Exit price"]),
            "actual_profit": round(profit, 2),
            "rule_exit_time": r.get("rule_exit_time"),
            "rule_exit_price": r.get("rule_exit_price"),
            "rule_exit_kind": r.get("rule_exit_kind"),
            "rule_profit": r.get("rule_profit"),
            "outside_plan": bool(r.get("outside_plan", False)),
            "r_multiple": r_mult,
            "observed_style": s.get("family", "unclassified"),
            "oversized": qty > max_size,
            "manual": is_manual,
            "outcome": outcome,
            "day_key": f"{instrument}|{entry_time.date().isoformat()}",
        })
    return combined


# --------------------------------------------------------------------------
# Trade replay: per-day candles at native + derived timeframes
# --------------------------------------------------------------------------

def _native_resolution_minutes(day_bars):
    if len(day_bars) < 2:
        return None
    deltas = day_bars["datetime"].diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return None
    return float(deltas.median())


def _ohlc_records(df):
    return [
        {
            "t": row["datetime"].isoformat(),
            "o": round(float(row["open"]), 4), "h": round(float(row["high"]), 4),
            "l": round(float(row["low"]), 4), "c": round(float(row["close"]), 4),
            "v": int(row["volume"]) if pd.notna(row["volume"]) else 0,
        }
        for _, row in df.iterrows()
    ]


def _resample(day_bars, rule):
    idx = day_bars.set_index("datetime")
    agg = idx.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return agg


def _build_replay_days(combined_trades, bars_df):
    day_keys = sorted({t["day_key"] for t in combined_trades})
    days = {}
    for day_key in day_keys:
        instrument, date_str = day_key.split("|", 1)
        day_date = pd.Timestamp(date_str).date()
        instrument_bars = bars_df[bars_df["instrument"] == instrument]
        day_bars = instrument_bars[instrument_bars["datetime"].dt.date == day_date].sort_values("datetime").reset_index(drop=True)
        if day_bars.empty:
            continue

        native_res = _native_resolution_minutes(day_bars)
        has_1m = native_res is not None and native_res <= 1.5

        resolutions = {}
        if has_1m:
            resolutions["1m"] = _ohlc_records(day_bars)
            resolutions["5m"] = _ohlc_records(_resample(day_bars, "5min"))
            resolutions["15m"] = _ohlc_records(_resample(day_bars, "15min"))
        else:
            resolutions["5m"] = _ohlc_records(day_bars)  # native 5m
            resolutions["15m"] = _ohlc_records(_resample(day_bars, "15min"))

        day_trades = [t for t in combined_trades if t["day_key"] == day_key]
        days[day_key] = {
            "instrument": instrument,
            "date": date_str,
            "has_1m": has_1m,
            "resolutions": resolutions,
            "trades": day_trades,
        }
    return days, [k for k in day_keys if k in days]


# --------------------------------------------------------------------------
# KPI row + verdict
# --------------------------------------------------------------------------

def _sparkline(values, max_points=SPARKLINE_MAX_POINTS):
    vals = [v for v in values if v is not None]
    if not vals:
        return []
    if len(vals) <= max_points:
        return [round(float(v), 2) for v in vals]
    idx = np.linspace(0, len(vals) - 1, max_points).round().astype(int)
    return [round(float(vals[i]), 2) for i in idx]


def _build_kpis_and_verdict(combined_trades, audit_findings, replay_findings):
    ordered = sorted(combined_trades, key=lambda t: t["entry_time"])
    profits_seq = [t["actual_profit"] for t in ordered]
    equity = list(np.cumsum(profits_seq)) if profits_seq else []
    net_pnl = round(equity[-1], 2) if equity else 0.0

    sorted_desc = sorted(profits_seq, reverse=True)
    top3_removed = round(sum(sorted_desc[:3]), 2)
    without_top3 = round(net_pnl - top3_removed, 2)

    top3_idx = set(np.argsort(profits_seq)[-3:]) if len(profits_seq) >= 3 else set(range(len(profits_seq)))
    without_top3_seq = [p for i, p in enumerate(profits_seq) if i not in top3_idx]
    without_top3_equity = list(np.cumsum(without_top3_seq)) if without_top3_seq else []

    core = audit_findings["core_stats"]
    quant = audit_findings["quant_stats"]
    luck = audit_findings["luck"]
    agg = replay_findings["aggregate"]
    r_values = [t["r_multiple"] for t in ordered if t["r_multiple"] is not None]

    win_pf_display = "n/a"
    if core.get("win_rate") is not None and core.get("profit_factor") is not None:
        win_pf_display = f"{core['win_rate']:.0%} / {core['profit_factor']:.2f}"

    exp_r_mean = quant["expectancy_r"]["mean"]
    conc = luck["concentration_top_decile_pct_of_gross_winning"]

    kpis = [
        {"id": "net_pnl", "label": "NET P&L", "display": f"${net_pnl:,.0f}",
         "sparkline": _sparkline(equity), "tone": "pos" if net_pnl >= 0 else "neg"},
        {"id": "without_top3", "label": "WITHOUT TOP 3", "display": f"${without_top3:,.0f}",
         "sparkline": _sparkline(without_top3_equity), "tone": "pos" if without_top3 >= 0 else "neg"},
        {"id": "rules_vs_actual", "label": "RULES VS ACTUAL", "display": f"${agg['signed_difference']:,.0f}",
         "sub": f"actual ${agg['actual_total_pnl']:,.0f} / rules ${agg['rule_managed_total_pnl']:,.0f}",
         "sparkline": [], "tone": "pos" if agg["signed_difference"] >= 0 else "neg"},
        {"id": "win_rate_pf", "label": "WIN RATE / PROFIT FACTOR", "display": win_pf_display,
         "sparkline": [], "tone": "neutral"},
        {"id": "expectancy_r", "label": "EXPECTANCY (R)", "display": f"{exp_r_mean:+.2f}R" if exp_r_mean is not None else "n/a",
         "sparkline": _sparkline(r_values), "tone": "pos" if (exp_r_mean or 0) >= 0 else "neg"},
        {"id": "concentration", "label": "CONCENTRATION", "display": f"{conc:.0f}%" if conc is not None else "n/a",
         "sparkline": [], "tone": "warn"},
    ]

    verdict = {
        "sentence": f"This account is up ${net_pnl:,.0f}; remove the top 3 trades and it's ${without_top3:,.0f}.",
        "net_pnl": net_pnl,
        "without_top3": without_top3,
        "top3_removed": top3_removed,
    }
    return kpis, verdict


# --------------------------------------------------------------------------
# Leaks
# --------------------------------------------------------------------------

def _build_leaks(combined_trades, audit_findings, rules):
    max_size = rules["max_size"]

    oversized_cost = 0.0
    for t in combined_trades:
        if t["oversized"] and t["qty"] > 0:
            per_unit = t["actual_profit"] / t["qty"]
            scaled = per_unit * max_size
            oversized_cost += (t["actual_profit"] - scaled)

    manual_deviation = 0.0
    for t in combined_trades:
        if t["manual"] and not t["outside_plan"] and t["rule_profit"] is not None:
            manual_deviation += (t["actual_profit"] - t["rule_profit"])

    outside_plan_pnl = sum(t["actual_profit"] for t in combined_trades if t["outside_plan"])

    luck_top3 = audit_findings["luck"]["pnl_without_top_n"].get(3, {}).get("removed_dollars", 0.0)

    items = [
        {"id": "revenge_sizing", "label": "Revenge sizing (oversized after a loss)", "dollars": round(oversized_cost, 2)},
        {"id": "manual_exits", "label": "Manual exits vs. rule exit", "dollars": round(manual_deviation, 2)},
        {"id": "outside_plan", "label": "Trades outside the plan", "dollars": round(outside_plan_pnl, 2)},
        {"id": "luck_dependency", "label": "Top-3 trade dependency", "dollars": round(luck_top3, 2)},
    ]
    items.sort(key=lambda x: abs(x["dollars"]), reverse=True)
    return items


# --------------------------------------------------------------------------
# Constellation + equity + quant-stats "Read:" sentence
# --------------------------------------------------------------------------

def _build_constellation(combined_trades):
    return [
        {
            "trade_number": t["trade_number"], "instrument": t["instrument"], "day_key": t["day_key"],
            "time": t["entry_time"].isoformat(), "pnl": t["actual_profit"], "qty": t["qty"],
            "oversized": t["oversized"], "outcome": t["outcome"], "side": t["side"],
        }
        for t in combined_trades
    ]


def _build_equity(combined_trades):
    ordered = sorted(combined_trades, key=lambda t: t["entry_time"])
    profits_seq = [t["actual_profit"] for t in ordered]
    actual_equity = list(np.cumsum(profits_seq)) if profits_seq else []

    top3_idx = set(np.argsort(profits_seq)[-3:]) if len(profits_seq) >= 3 else set(range(len(profits_seq)))
    without_top3_seq = [p for i, p in enumerate(profits_seq) if i not in top3_idx]
    without_top3_equity = list(np.cumsum(without_top3_seq)) if without_top3_seq else []

    return {
        "actual": [round(float(v), 2) for v in actual_equity],
        "without_top3": [round(float(v), 2) for v in without_top3_equity],
    }


def _quant_stats_read(quant_stats):
    exp_r = quant_stats["expectancy_r"]
    sqn_info = quant_stats["sqn"]
    kelly = quant_stats["kelly_fraction"]["value"]

    edge_state = "not yet proven" if not exp_r.get("distinguishable_from_zero") else "a real, measurable edge"
    kelly_note = f"sizing math floors out at 0% (don't lean on this edge)" if kelly == 0 else f"sizing math suggests ~{kelly:.0%} of capital per trade"

    sentence = (
        f"Read: this system shows {edge_state} "
        f"({exp_r.get('mean'):+.2f}R average, +/- {exp_r.get('se'):.2f}R SE)"
        if exp_r.get("mean") is not None and exp_r.get("se") is not None else "Read: not enough trades yet to read the edge."
    )
    if sqn_info["value"] is not None:
        sentence += f", SQN says it's {('hard to trade' if sqn_info['value'] < metrics.SQN_HARD_TO_TRADE_THRESHOLD else ('good' if sqn_info['value'] > metrics.SQN_GOOD_THRESHOLD else 'tradable'))}"
    sentence += f", and {kelly_note}."
    return sentence


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def build_dash_data(trades_path="trades.csv", rules_path=rules_mod.RULES_PATH_DEFAULT,
                     strategies_path=rules_mod.STRATEGIES_PATH_DEFAULT, strategy_name=None):
    trades_df = trades_io.load_trades(trades_path)
    active_rules = rules_mod.load_rules(rules_path)

    declared_family = None
    strategy_playbook = None
    if strategy_name:
        entry = rules_mod.load_strategy(strategy_name, path=strategies_path)
        declared_family = entry.get("family")
        strategy_playbook = entry.get("playbook")

    symbols, start, end = bars_mod.symbols_and_range_from_trades(trades_df)
    bars_df, bar_warnings = bars_mod.fetch_bars(symbols, start_date=start, end_date=end, save_path=None)
    if bars_df.empty:
        raise RuntimeError("No bars available for the symbols/date range in this trade log; cannot build the dashboard.")

    audit_findings = auditor_engine.audit_trades(trades_df, active_rules)
    replay_findings = discipline.replay_rules(trades_df, bars_df, active_rules)
    style_findings = context.style_check(trades_df, bars_df, declared_family=declared_family)

    combined_trades = _build_combined_trades(trades_df, replay_findings["per_trade"], style_findings["per_trade"], active_rules)
    kpis, verdict = _build_kpis_and_verdict(combined_trades, audit_findings, replay_findings)
    leaks = _build_leaks(combined_trades, audit_findings, active_rules)
    constellation = _build_constellation(combined_trades)
    equity = _build_equity(combined_trades)
    replay_days, replay_day_order = _build_replay_days(combined_trades, bars_df)
    quant_read = _quant_stats_read(audit_findings["quant_stats"])

    dash_data = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "strategy_name": strategy_name,
            "declared_family": declared_family,
            "strategy_playbook": strategy_playbook,
            "rules": active_rules,
            "n_trades": len(trades_df),
            "bar_warnings": bar_warnings,
        },
        "verdict": verdict,
        "kpis": kpis,
        "quant_stats": audit_findings["quant_stats"],
        "quant_stats_read": quant_read,
        "trade_replay": {"days": replay_days, "day_order": replay_day_order},
        "constellation": constellation,
        "style_panel": style_findings,
        "equity": equity,
        "leaks": leaks,
        "pnl_by_hour": audit_findings["pnl_by_hour"],
        "pnl_by_instrument": audit_findings["pnl_by_instrument"],
        "discipline_cost": {
            "assumptions": replay_findings["assumptions"],
            "aggregate": replay_findings["aggregate"],
            "core_stats_actual": replay_findings["core_stats_actual"],
            "core_stats_rule_managed": replay_findings["core_stats_rule_managed"],
        },
        "behavioral": {
            "luck": audit_findings["luck"],
            "revenge_sizing": audit_findings["revenge_sizing"],
            "disposition": audit_findings["disposition"],
            "plan_adherence": audit_findings["plan_adherence"],
        },
    }
    return metrics.to_native(dash_data)


def render_dashboard_html(dash_data, template_path=DEFAULT_TEMPLATE_PATH):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    json_str = json.dumps(dash_data)
    # escape "</" so the embedded JSON can't prematurely close the <script> tag
    safe_json = json_str.replace("</", "<\\/")
    if DASH_DATA_PLACEHOLDER not in template:
        raise RuntimeError(f"Template is missing the {DASH_DATA_PLACEHOLDER!r} placeholder.")
    return template.replace(DASH_DATA_PLACEHOLDER, safe_json)


def build_dashboard(trades_path="trades.csv", rules_path=rules_mod.RULES_PATH_DEFAULT,
                     strategies_path=rules_mod.STRATEGIES_PATH_DEFAULT, strategy_name=None,
                     output_json="dash_data.json", output_html="dashboard.html",
                     template_path=DEFAULT_TEMPLATE_PATH):
    dash_data = build_dash_data(trades_path, rules_path, strategies_path, strategy_name)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dash_data, f, indent=2)

    html = render_dashboard_html(dash_data, template_path=template_path)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    return dash_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the trade-auditor dashboard.")
    parser.add_argument("--trades", default="trades.csv")
    parser.add_argument("--rules", default=rules_mod.RULES_PATH_DEFAULT)
    parser.add_argument("--strategies", default=rules_mod.STRATEGIES_PATH_DEFAULT)
    parser.add_argument("--strategy", default=None, help="Named strategy (for the declared family in the style check).")
    parser.add_argument("--out-json", default="dash_data.json")
    parser.add_argument("--out-html", default="dashboard.html")
    args = parser.parse_args()

    data = build_dashboard(
        trades_path=args.trades, rules_path=args.rules, strategies_path=args.strategies,
        strategy_name=args.strategy, output_json=args.out_json, output_html=args.out_html,
    )
    print(f"Wrote {args.out_json} and {args.out_html} ({data['meta']['n_trades']} trades).")
