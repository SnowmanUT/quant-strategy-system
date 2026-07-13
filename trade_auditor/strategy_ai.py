"""
Layer 2: strategy_ai.py

The PLAIN-ENGLISH PATH. This whole step is the AI (purple) step -- it
happens once per strategy. The user:
  1. Picks a trading family from ~15 choices.
  2. Describes how they trade, in their own words.
That family + text is sent to the model with a system prompt that demands
STRICT JSON matching the rules.py schema -- no prose, no backticks. The
response is parsed and handed back as an editable review form (an
AIProposal). Nothing runs until the user approves: approving validates the
(possibly hand-edited) rules, locks them in as the active rule set, and
saves them into the named-strategy library. The free-text playbook is kept
verbatim alongside the rules so a later coaching feature has context.

If there's no AI key at all (see ai_client.ai_available()), this whole path
should be greyed out by the caller/UI -- presets and manual rule editing in
rules.py keep working regardless.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import ai_client
import rules as rules_mod

AI_STEP_LABEL = "AI"        # UI badge text for this step
AI_STEP_COLOR = "purple"    # UI badge color for this step

TRADE_FAMILIES = [
    "Momentum",
    "Mean Reversion",
    "Breakout",
    "Trend Following",
    "Scalp",
    "Swing",
    "ICT / Price Action",
    "Pullback / Continuation",
    "Range Trading",
    "Fade",
    "Opening Range Breakout",
    "VWAP Trading",
    "Gap Trading",
    "News / Catalyst Trading",
    "Statistical / Systematic",
]

_SCHEMA_DESCRIPTION = """{
  "direction": "long_only" | "short_only" | "both",
  "stop_pct": <positive number, percent risked per trade>,
  "target_pct": <positive number, percent target> | null,
  "session_start": "HH:MM",   // 24h, US/Eastern
  "session_end": "HH:MM",     // 24h, US/Eastern
  "max_size": <positive number, position size cap>,
  "exit_style": "fixed" | "breakeven" | "trailing" | "time",
  "slippage_bps": <non-negative number, default 2 if not implied by the text>
}"""

SYSTEM_PROMPT = f"""You convert a trader's plain-English description of how they trade into a \
single strict JSON object that a rules engine can apply mechanically.

Output ONLY a single JSON object matching exactly this schema -- no prose, no \
explanation, no markdown code fences, nothing before or after it:

{_SCHEMA_DESCRIPTION}

Rules for filling it in:
- Infer every field as best you can from the trader's family + description.
- If the text doesn't specify a target, set target_pct to null rather than guessing.
- session_start/session_end must be plain "HH:MM" 24-hour strings in US/Eastern time; \
default to "09:30"/"16:00" (regular session) if the text gives no session info.
- If nothing in the text implies otherwise, use slippage_bps: 2.
- Never include any field not in the schema. Never include comments. Output valid JSON only."""


class StrategyAIError(RuntimeError):
    """Raised for any plain-English-path failure (no key, bad response, parse failure). One clear line."""
    pass


@dataclass
class AIProposal:
    family: str
    free_text: str
    proposed_rules: Optional[dict]      # None if parsing/validation failed
    validation_error: Optional[str]     # set if proposed_rules failed schema validation
    raw_response: str
    is_ai_generated: bool = True
    approved: bool = False


def _strip_code_fences(text):
    """Defensive cleanup in case the model wraps JSON in backticks despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _parse_strict_json(raw_response):
    cleaned = _strip_code_fences(raw_response)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise StrategyAIError(f"Model response wasn't valid JSON: {e}")


def propose_rules_from_text(family, free_text, timeout=30):
    """
    Step 1 of the AI (purple) path. Sends family + free text to the model and
    returns an AIProposal for review -- it does NOT save or activate anything.

    Raises StrategyAIError immediately (before any network call) if there's
    no AI key configured, so the caller can grey out the button instead of
    hitting this at all.
    """
    if family not in TRADE_FAMILIES:
        raise StrategyAIError(f"Unknown trading family '{family}'. Choose from: {TRADE_FAMILIES}.")
    if not free_text or not free_text.strip():
        raise StrategyAIError("Describe how you trade before generating rules.")
    if not ai_client.ai_available():
        raise StrategyAIError(
            "No AI key configured -- the plain-English path is unavailable. "
            "Use a preset or build rules manually instead."
        )

    user_message = f"Trading family: {family}\n\nTrader's own description:\n{free_text.strip()}"

    raw = ai_client.call_ai(SYSTEM_PROMPT, user_message, timeout=timeout)
    parsed = _parse_strict_json(raw)

    try:
        normalized = rules_mod.validate_rules(parsed)
        return AIProposal(
            family=family, free_text=free_text.strip(), proposed_rules=normalized,
            validation_error=None, raw_response=raw,
        )
    except rules_mod.RulesValidationError as e:
        # Hand back the raw parse + error so the review form can show what's wrong
        # and let the user fix it by hand rather than just failing outright.
        return AIProposal(
            family=family, free_text=free_text.strip(), proposed_rules=parsed if isinstance(parsed, dict) else None,
            validation_error=str(e), raw_response=raw,
        )


def approve_proposal(proposal, strategy_name, edited_rules=None,
                      rules_path=rules_mod.RULES_PATH_DEFAULT,
                      strategies_path=rules_mod.STRATEGIES_PATH_DEFAULT):
    """
    Step 2 of the AI (purple) path -- called only when the user clicks
    Approve in the review form. `edited_rules` is whatever the user ended up
    with in the editable form (may differ from proposal.proposed_rules if
    they changed anything, including fixing a validation error).

    Validates the final rules, locks them in as the active rule set
    (rules.json), and saves them into the named-strategy library alongside
    the original free-text playbook and family, tagged origin="ai".
    """
    final_rules = edited_rules if edited_rules is not None else proposal.proposed_rules
    if final_rules is None:
        raise StrategyAIError("No rules to approve -- fix the review form fields first.")

    normalized = rules_mod.save_rules(final_rules, path=rules_path)
    rules_mod.save_strategy(
        strategy_name, normalized,
        playbook=proposal.free_text, family=proposal.family, origin="ai",
        path=strategies_path,
    )
    proposal.approved = True
    proposal.proposed_rules = normalized
    return normalized


if __name__ == "__main__":
    print("Trading families:", TRADE_FAMILIES)
    print("AI available:", ai_client.ai_available())
