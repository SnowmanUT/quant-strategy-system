"""
Layer 5: consult_layer.py

Implements the /consult chat: a multi-turn, persistent AI conversation
where the trader can ask about trading methods/models suited to prop-firm
evaluation trading. Unlike Coach's Read (coaching_layer.py, one-shot,
verdict-style), this is consultative and free to suggest anything -- not
limited to this app's preset library.

Reuses coaching_layer.build_distilled_findings() for the audited-numbers
grounding (no duplication), and adds a daily P&L breakdown on top: prop-firm
rules (daily loss limits, trailing/end-of-day drawdown, consistency rules)
are fundamentally daily concepts that the coach's existing distillation
doesn't cover.
"""

import json

import ai_client
import coaching_layer

MAX_HISTORY_MESSAGES = 20

CONSULT_SYSTEM_PROMPT_BASE = """You are a trading-strategy consultant helping a funded-account \
trader pick methods/models suited to prop-firm evaluation trading. This is an ongoing \
conversation -- ask clarifying questions about the trader's specific firm and rules when you \
need them (max daily loss, trailing vs. end-of-day drawdown, profit target, consistency rule / \
max-day-% cap, scaling plan, contracts allowed) rather than guessing.

Baseline domain knowledge:
- Most prop firms enforce a MAX DAILY LOSS (hit it, that day's trading is locked or the account \
fails) and a separate DRAWDOWN limit, which is either TRAILING (moves up with new equity highs, \
can only get tighter) or END-OF-DAY (fixed relative to the prior day's close, doesn't ratchet \
intraday).
- A CONSISTENCY RULE caps how much of total profit can come from a single day (commonly \
20-40%) -- a trader with one huge day and a mediocre rest can fail evaluation on consistency \
even after hitting the profit target.
- PROFIT TARGETS are usually a fixed dollar amount or % of starting balance, sometimes with a \
minimum number of trading days required.
- SCALING PLANS increase contract size after hitting profit milestones -- relevant when \
recommending position-sizing approaches.

Grounding rules:
- Every claim about the trader's own past performance must cite a specific number from the \
findings JSON below. Never invent a stat.
- You cannot verify discretionary setup validity (e.g. ICT order blocks, fair value gaps, \
supply/demand zones) from price data alone -- say so plainly if it comes up, don't pretend you \
can validate it.
- Suggesting trading methods/models is NOT limited to any preset list -- recommend whatever \
fits the trader's demonstrated behavior and stated constraints, and explain why.
- Be direct, not motivational filler. This is a consult, not a pep talk.

Write in plain markdown -- short paragraphs, bullets where useful. Keep responses focused, not \
exhaustive."""


class ConsultError(RuntimeError):
    """Raised for any consult failure (no key, AI client error). One clear line."""
    pass


def build_daily_pnl_summary(dash_data):
    """
    Per-day net P&L across dash_data["trade_replay"]["days"], plus the best
    day's share of total gross (winning-only) profit -- the number a
    consistency rule check actually needs.
    """
    days = dash_data.get("trade_replay", {}).get("days", {})
    day_rows = []
    gross_profit = 0.0
    for day in days.values():
        trades = day.get("trades", [])
        net = round(sum((t.get("actual_profit") or 0) for t in trades), 2)
        day_rows.append({"date": day.get("date"), "net_pnl": net, "n_trades": len(trades)})
        gross_profit += sum(p for p in (t.get("actual_profit") for t in trades) if p and p > 0)

    if not day_rows:
        return {"n_days": 0, "best_day": None, "worst_day": None, "best_day_pct_of_gross_profit": None}

    day_rows.sort(key=lambda d: d["net_pnl"], reverse=True)
    best, worst = day_rows[0], day_rows[-1]
    best_pct = round(100 * best["net_pnl"] / gross_profit, 1) if gross_profit > 0 and best["net_pnl"] > 0 else None

    return {
        "n_days": len(day_rows),
        "best_day": best,
        "worst_day": worst,
        "best_day_pct_of_gross_profit": best_pct,
    }


def build_consult_system_prompt(dash_data):
    findings = coaching_layer.build_distilled_findings(dash_data)
    findings["daily_pnl_summary"] = build_daily_pnl_summary(dash_data)
    return (
        CONSULT_SYSTEM_PROMPT_BASE
        + "\n\nHere is the trader's audited findings (JSON), including the daily P&L "
        "breakdown relevant to prop-firm rules:\n\n"
        + json.dumps(findings, indent=2)
    )


def get_reply(message, history, dash_data, timeout=45):
    """
    Build the consult system prompt (fresh findings from the CURRENT
    dash_data), take the last MAX_HISTORY_MESSAGES turns of history
    (stripped to bare {role, content}), and get a reply. Raises
    ConsultError on no-key or any AI-client failure.
    """
    if not ai_client.ai_available():
        raise ConsultError("set a key to enable coaching")

    system_prompt = build_consult_system_prompt(dash_data)
    capped = history[-MAX_HISTORY_MESSAGES:]
    api_messages = [{"role": h["role"], "content": h["content"]} for h in capped]

    try:
        return ai_client.call_ai_conversation(system_prompt, api_messages, timeout=timeout, max_tokens=1200)
    except ai_client.AIClientError as e:
        raise ConsultError(str(e))
