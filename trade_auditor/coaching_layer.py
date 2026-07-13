"""
Layer 5: coaching_layer.py

Implements POST /coach. Builds a distilled findings JSON -- the headline
stats, the luck numbers, the discipline aggregate, the ranked leaks, the
five worst rule-deviation trades, the style match, a sample-size note, plus
the user's free-text playbook. A few KB total. NEVER the price bars.

Calls the provider through Layer 2's ai_client, prints the report to the
terminal, and returns the markdown for the dashboard's Coach's Read panel.
"""

import json

import ai_client

SMALL_SAMPLE_THRESHOLD = 30  # trades below this get an explicit small-sample note

SYSTEM_PROMPT = """You are a blunt trading performance coach reviewing a trader's own \
audited findings. Rules you must follow:

- Every claim you make must cite a specific number from the findings JSON you're given.
- Never invent a number, statistic, or trade detail that isn't in the findings.
- You cannot verify discretionary setup validity (e.g. ICT order blocks, fair value gaps, \
supply/demand zones) from price data alone -- if the trader's playbook mentions this kind \
of discretionary context, say plainly that you can't verify it rather than pretending you can.
- Be direct and blunt, not motivational filler. This is a performance review, not a pep talk.
- End your report with exactly one line starting with "Verdict:" that sums up the account \
in one sentence.

Write in plain markdown -- short paragraphs and a few bullet points is fine. Keep it tight."""


class CoachingError(RuntimeError):
    """Raised for any coaching failure (no key, AI client error). One clear line."""
    pass


def _worst_rule_deviation_trades(dash_data, n=5):
    """The n trades where actual P&L diverged most from what the rules would have done."""
    all_trades = []
    for day in dash_data.get("trade_replay", {}).get("days", {}).values():
        all_trades.extend(day.get("trades", []))

    deviations = []
    for t in all_trades:
        if t.get("rule_profit") is None or t.get("outside_plan"):
            continue
        dev = round(t["actual_profit"] - t["rule_profit"], 2)
        deviations.append({
            "trade_number": t["trade_number"],
            "instrument": t["instrument"],
            "actual_profit": t["actual_profit"],
            "rule_profit": t["rule_profit"],
            "deviation": dev,
            "rule_exit_kind": t.get("rule_exit_kind"),
            "manual_exit": bool(t.get("manual", False)),
        })
    deviations.sort(key=lambda d: abs(d["deviation"]), reverse=True)
    return deviations[:n]


def _get_top_n_bucket(pnl_without_top_n, n):
    """pnl_without_top_n's keys are ints when passed in-process, but strings once
    round-tripped through JSON (e.g. read back from dash_data.json) -- handle both."""
    return pnl_without_top_n.get(n) or pnl_without_top_n.get(str(n)) or {}


def build_distilled_findings(dash_data):
    """
    A few KB, no price bars: headline stats, luck, discipline aggregate,
    ranked leaks, the five worst rule-deviation trades, style match, a
    sample-size note, and the free-text playbook.
    """
    meta = dash_data.get("meta", {})
    quant = dash_data.get("quant_stats", {})
    behavioral = dash_data.get("behavioral", {})
    n_trades = meta.get("n_trades", 0)

    sample_note = (
        f"Only {n_trades} trades -- treat every stat below as a rough read, not a settled verdict."
        if n_trades < SMALL_SAMPLE_THRESHOLD else
        f"{n_trades} trades -- a reasonable sample to draw conclusions from."
    )

    luck = behavioral.get("luck", {})

    return {
        "n_trades": n_trades,
        "declared_family": meta.get("declared_family"),
        "playbook": meta.get("strategy_playbook"),
        "sample_size_note": sample_note,
        "verdict_sentence": dash_data.get("verdict", {}).get("sentence"),
        "headline_stats": {
            "expectancy_r": quant.get("expectancy_r", {}).get("mean"),
            "expectancy_r_se": quant.get("expectancy_r", {}).get("se"),
            "expectancy_r_distinguishable_from_zero": quant.get("expectancy_r", {}).get("distinguishable_from_zero"),
            "sqn": quant.get("sqn", {}).get("value"),
            "sharpe_per_trade": quant.get("sharpe_per_trade", {}).get("value"),
            "payoff_ratio": quant.get("payoff_ratio", {}).get("value"),
            "max_drawdown_dollars": quant.get("max_drawdown_dollars", {}).get("value"),
            "kelly_fraction": quant.get("kelly_fraction", {}).get("value"),
        },
        "luck": {
            "concentration_top_decile_pct_of_gross_winning": luck.get("concentration_top_decile_pct_of_gross_winning"),
            "pnl_without_top_3": _get_top_n_bucket(luck.get("pnl_without_top_n", {}), 3).get("pnl_without"),
        },
        "discipline_aggregate": dash_data.get("discipline_cost", {}).get("aggregate"),
        "ranked_leaks": dash_data.get("leaks", []),
        "worst_rule_deviation_trades": _worst_rule_deviation_trades(dash_data),
        "style_match": {
            "declared_family": dash_data.get("style_panel", {}).get("declared_family"),
            "observed_majority_family": dash_data.get("style_panel", {}).get("observed_majority_family"),
            "verdict": dash_data.get("style_panel", {}).get("verdict"),
        },
        "plan_adherence_rate": behavioral.get("plan_adherence", {}).get("adherence_rate"),
    }


def get_coaching(dash_data, timeout=45):
    """
    Build the distilled findings, call the AI through Layer 2's ai_client,
    print the report to the terminal, and return the markdown for the
    dashboard's Coach's Read panel. Raises CoachingError on any failure --
    callers should check ai_client.ai_available() first for NO-KEY MODE.
    """
    if not ai_client.ai_available():
        raise CoachingError("set a key to enable coaching")

    findings = build_distilled_findings(dash_data)
    user_message = (
        "Here are a trader's audited findings (JSON). Write the performance review "
        "described in your instructions.\n\n" + json.dumps(findings, indent=2)
    )

    try:
        markdown = ai_client.call_ai(SYSTEM_PROMPT, user_message, timeout=timeout, max_tokens=1200)
    except ai_client.AIClientError as e:
        raise CoachingError(str(e))

    print("\n" + "=" * 60)
    print("COACH'S READ")
    print("=" * 60)
    print(markdown)
    print("=" * 60 + "\n")

    return markdown
