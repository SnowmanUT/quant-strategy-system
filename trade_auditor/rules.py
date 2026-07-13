"""
Layer 2: rules.py

The strategy rules engine. A single JSON object describes how the auditor
should mechanically apply a strategy:

{
  "direction":     "long_only" | "short_only" | "both",
  "stop_pct":      float,        # required, > 0
  "target_pct":    float | null, # null = no fixed target
  "session_start": "HH:MM",      # Eastern
  "session_end":   "HH:MM",      # Eastern
  "max_size":      float,        # position size cap (shares/contracts)
  "exit_style":    "fixed" | "breakeven" | "trailing" | "time",
  "slippage_bps":  float         # default 2
}

- The active rule set is persisted to rules.json (save_rules / load_rules).
- A named-strategy library is persisted to strategies.json: save the current
  rules under a name, and reload/apply them with one call (save_strategy /
  apply_strategy). Each library entry can also carry the free-text playbook
  and family it came from, so a later "coach" feature has context.
- PRESETS: a dozen one-click templates covering popular styles. A preset
  only pre-fills these same schema fields -- everything stays editable.
- compute_readout(rules): reward-to-risk from stop/target, and the
  breakeven win rate 1/(1+R), phrased in plain English. Cheap to call on
  every keystroke while a form is being edited.

This module has NO dependency on the AI client -- presets, manual editing,
and the strategy library all work with zero API key, by design.
"""

import json
import os
from datetime import datetime, timezone

DIRECTIONS = {"long_only", "short_only", "both"}
EXIT_STYLES = {"fixed", "breakeven", "trailing", "time"}

DEFAULT_SLIPPAGE_BPS = 2

RULES_SCHEMA_FIELDS = [
    "direction", "stop_pct", "target_pct", "session_start", "session_end",
    "max_size", "exit_style", "slippage_bps",
]

DEFAULT_RULES = {
    "direction": "both",
    "stop_pct": 0.5,
    "target_pct": 1.0,
    "session_start": "09:30",
    "session_end": "16:00",
    "max_size": 100,
    "exit_style": "fixed",
    "slippage_bps": DEFAULT_SLIPPAGE_BPS,
}

RULES_PATH_DEFAULT = "rules.json"
STRATEGIES_PATH_DEFAULT = "strategies.json"


class RulesValidationError(ValueError):
    """Raised when a rules dict doesn't match the schema. Message is one clear line."""
    pass


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _valid_hhmm(s):
    if not isinstance(s, str):
        return False
    try:
        datetime.strptime(s, "%H:%M")
        return True
    except ValueError:
        return False


def validate_rules(rules):
    """
    Validate a rules dict against the schema. Fills in slippage_bps with the
    default if omitted. Returns a normalized copy. Raises RulesValidationError
    with one clear line on the first problem found.
    """
    if not isinstance(rules, dict):
        raise RulesValidationError("Rules must be a JSON object.")

    missing = [f for f in RULES_SCHEMA_FIELDS if f not in rules and f != "slippage_bps"]
    if missing:
        raise RulesValidationError(f"Rules are missing required field(s): {', '.join(missing)}.")

    r = dict(rules)
    r.setdefault("slippage_bps", DEFAULT_SLIPPAGE_BPS)

    if r["direction"] not in DIRECTIONS:
        raise RulesValidationError(f"direction must be one of {sorted(DIRECTIONS)}, got {r['direction']!r}.")

    if not _is_number(r["stop_pct"]) or r["stop_pct"] <= 0:
        raise RulesValidationError("stop_pct must be a positive number.")

    if r["target_pct"] is not None and (not _is_number(r["target_pct"]) or r["target_pct"] <= 0):
        raise RulesValidationError("target_pct must be a positive number or null (no fixed target).")

    if not _valid_hhmm(r["session_start"]):
        raise RulesValidationError("session_start must be an 'HH:MM' string (Eastern).")
    if not _valid_hhmm(r["session_end"]):
        raise RulesValidationError("session_end must be an 'HH:MM' string (Eastern).")
    if r["session_start"] >= r["session_end"]:
        raise RulesValidationError("session_start must be earlier than session_end.")

    if not _is_number(r["max_size"]) or r["max_size"] <= 0:
        raise RulesValidationError("max_size must be a positive number.")

    if r["exit_style"] not in EXIT_STYLES:
        raise RulesValidationError(f"exit_style must be one of {sorted(EXIT_STYLES)}, got {r['exit_style']!r}.")

    if not _is_number(r["slippage_bps"]) or r["slippage_bps"] < 0:
        raise RulesValidationError("slippage_bps must be a non-negative number.")

    return {field: r[field] for field in RULES_SCHEMA_FIELDS}


# --------------------------------------------------------------------------
# Active rule set persistence (rules.json)
# --------------------------------------------------------------------------

def save_rules(rules, path=RULES_PATH_DEFAULT):
    """Validate and persist the active rule set to rules.json."""
    normalized = validate_rules(rules)
    with open(path, "w") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def load_rules(path=RULES_PATH_DEFAULT):
    """Load the active rule set. Returns DEFAULT_RULES if no file exists yet."""
    if not os.path.exists(path):
        return dict(DEFAULT_RULES)
    with open(path, "r") as f:
        raw = json.load(f)
    return validate_rules(raw)


# --------------------------------------------------------------------------
# Named-strategy library (strategies.json)
# --------------------------------------------------------------------------

def _load_library(path=STRATEGIES_PATH_DEFAULT):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _write_library(library, path=STRATEGIES_PATH_DEFAULT):
    with open(path, "w") as f:
        json.dump(library, f, indent=2)


def save_strategy(name, rules, playbook=None, family=None, origin="manual", path=STRATEGIES_PATH_DEFAULT):
    """
    Save the current rules into the named-strategy library under `name`,
    overwriting any existing entry with that name.

    playbook: optional free-text description of how the trader trades
              (kept verbatim so a later coaching feature has context).
    family:   optional trading family label (e.g. "Momentum"), set when the
              strategy came from the AI plain-English path.
    origin:   "manual" | "ai" -- how this entry's rules were produced.
    """
    if not name or not name.strip():
        raise RulesValidationError("Strategy name cannot be empty.")
    normalized = validate_rules(rules)
    library = _load_library(path)
    library[name.strip()] = {
        "rules": normalized,
        "playbook": playbook,
        "family": family,
        "origin": origin,
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_library(library, path)
    return library[name.strip()]


def load_strategy(name, path=STRATEGIES_PATH_DEFAULT):
    """Return the full library entry ({rules, playbook, family, origin, saved_at}) for `name`."""
    library = _load_library(path)
    if name not in library:
        raise RulesValidationError(f"No saved strategy named '{name}'.")
    return library[name]


def list_strategies(path=STRATEGIES_PATH_DEFAULT):
    """Return the list of saved strategy names."""
    return sorted(_load_library(path).keys())


def delete_strategy(name, path=STRATEGIES_PATH_DEFAULT):
    library = _load_library(path)
    if name not in library:
        raise RulesValidationError(f"No saved strategy named '{name}'.")
    del library[name]
    _write_library(library, path)


def apply_strategy(name, strategies_path=STRATEGIES_PATH_DEFAULT, rules_path=RULES_PATH_DEFAULT):
    """One-click reload: load a saved strategy's rules and make them the active rule set."""
    entry = load_strategy(name, path=strategies_path)
    return save_rules(entry["rules"], path=rules_path)


# --------------------------------------------------------------------------
# PRESETS -- a dozen one-click starting templates. Only pre-fill the schema;
# every field stays editable in the form afterward.
# --------------------------------------------------------------------------

PRESETS = {
    "Mean Reversion": {
        "direction": "both", "stop_pct": 0.6, "target_pct": 0.6,
        "session_start": "09:45", "session_end": "15:45",
        "max_size": 100, "exit_style": "fixed", "slippage_bps": 2,
    },
    "Breakout": {
        "direction": "long_only", "stop_pct": 0.8, "target_pct": 2.0,
        "session_start": "09:30", "session_end": "16:00",
        "max_size": 100, "exit_style": "trailing", "slippage_bps": 3,
    },
    "Momentum": {
        "direction": "both", "stop_pct": 0.5, "target_pct": 1.5,
        "session_start": "09:30", "session_end": "15:30",
        "max_size": 100, "exit_style": "trailing", "slippage_bps": 3,
    },
    "Trend Following": {
        "direction": "both", "stop_pct": 1.5, "target_pct": None,
        "session_start": "09:30", "session_end": "16:00",
        "max_size": 100, "exit_style": "trailing", "slippage_bps": 2,
    },
    "Scalp": {
        "direction": "both", "stop_pct": 0.15, "target_pct": 0.2,
        "session_start": "09:30", "session_end": "11:30",
        "max_size": 200, "exit_style": "fixed", "slippage_bps": 4,
    },
    "Pullback": {
        "direction": "both", "stop_pct": 0.5, "target_pct": 1.0,
        "session_start": "09:30", "session_end": "16:00",
        "max_size": 100, "exit_style": "breakeven", "slippage_bps": 2,
    },
    "Fade": {
        "direction": "both", "stop_pct": 0.4, "target_pct": 0.8,
        "session_start": "09:30", "session_end": "10:30",
        "max_size": 100, "exit_style": "fixed", "slippage_bps": 3,
    },
    "Opening Range": {
        "direction": "both", "stop_pct": 0.5, "target_pct": 1.0,
        "session_start": "09:30", "session_end": "10:30",
        "max_size": 100, "exit_style": "breakeven", "slippage_bps": 3,
    },
    "VWAP Reversion": {
        "direction": "both", "stop_pct": 0.3, "target_pct": 0.3,
        "session_start": "09:45", "session_end": "15:45",
        "max_size": 150, "exit_style": "fixed", "slippage_bps": 2,
    },
    "Gap And Go": {
        "direction": "long_only", "stop_pct": 1.0, "target_pct": 2.5,
        "session_start": "09:30", "session_end": "10:15",
        "max_size": 100, "exit_style": "trailing", "slippage_bps": 4,
    },
    "Range Reversion": {
        "direction": "both", "stop_pct": 0.4, "target_pct": 0.6,
        "session_start": "10:00", "session_end": "15:30",
        "max_size": 100, "exit_style": "fixed", "slippage_bps": 2,
    },
    "Momentum Ignition": {
        "direction": "both", "stop_pct": 0.3, "target_pct": 0.9,
        "session_start": "09:30", "session_end": "16:00",
        "max_size": 100, "exit_style": "time", "slippage_bps": 4,
    },
}


def apply_preset(name):
    """Return a fresh, validated copy of a preset's rules by name."""
    if name not in PRESETS:
        raise RulesValidationError(f"No preset named '{name}'. Available: {sorted(PRESETS.keys())}.")
    return validate_rules(PRESETS[name])


# --------------------------------------------------------------------------
# Shared plan-eligibility checks -- used by Layer 3 (auditor_engine.py and
# discipline.py) so "does this trade match the plan" is defined exactly once.
# --------------------------------------------------------------------------

def direction_matches(market_pos, direction):
    """True if a trade's Market pos. ('Long'/'Short') is allowed by the rules' direction."""
    if direction == "both":
        return True
    wanted = "Long" if direction == "long_only" else "Short"
    return str(market_pos).strip().title() == wanted


def in_session(ts, session_start, session_end):
    """True if a timestamp's HH:MM falls within [session_start, session_end], inclusive."""
    hhmm = ts.strftime("%H:%M")
    return session_start <= hhmm <= session_end


def size_within_plan(qty, max_size):
    """True if a trade's Qty doesn't exceed the plan's max_size."""
    return qty <= max_size


def plan_eligible(row, rules):
    """
    True if a trade row (with 'Market pos.', 'Entry time', 'Qty') is inside
    the plan on all three axes: direction, session window, and size.
    Trades that fail this are "outside your plan, not taken" downstream.
    """
    return (
        direction_matches(row["Market pos."], rules["direction"])
        and in_session(row["Entry time"], rules["session_start"], rules["session_end"])
        and size_within_plan(row["Qty"], rules["max_size"])
    )


# --------------------------------------------------------------------------
# LIVE READOUT -- reward:risk and breakeven win rate, recomputed as the
# user types. Cheap, synchronous, no I/O.
# --------------------------------------------------------------------------

def compute_readout(rules):
    """
    Given a (possibly not-yet-fully-valid) rules dict, compute:
      - reward_to_risk: target_pct / stop_pct, or None if no fixed target
      - breakeven_win_rate: 1 / (1 + R), or None if reward_to_risk is None
      - plain_english: a one/two sentence human-readable summary

    Tolerant of partial/in-progress input from a live form: returns a
    plain_english note instead of raising if stop_pct isn't usable yet.
    """
    stop_pct = rules.get("stop_pct")
    target_pct = rules.get("target_pct")
    exit_style = rules.get("exit_style")

    if not _is_number(stop_pct) or stop_pct <= 0:
        return {
            "reward_to_risk": None,
            "breakeven_win_rate": None,
            "plain_english": "Enter a stop % greater than 0 to see reward:risk and breakeven win rate.",
        }

    if target_pct is None:
        style_note = {
            "trailing": "your trailing stop will decide how much you actually make on winners.",
            "time": "your time-based exit will decide how much you actually make on winners.",
            "breakeven": "reward is open-ended until you move to breakeven and let it run.",
            "fixed": "there's no target set, so reward is effectively open-ended.",
        }.get(exit_style, "reward is open-ended since there's no fixed target.")
        return {
            "reward_to_risk": None,
            "breakeven_win_rate": None,
            "plain_english": (
                f"No fixed target -- you're risking {stop_pct:g}% per trade, and {style_note} "
                "Reward:risk isn't defined until there's a target or a closed-out sample of trades to measure."
            ),
        }

    if not _is_number(target_pct) or target_pct <= 0:
        return {
            "reward_to_risk": None,
            "breakeven_win_rate": None,
            "plain_english": "Enter a target % greater than 0, or leave it blank for no fixed target.",
        }

    r = target_pct / stop_pct
    breakeven = 1 / (1 + r)

    plain_english = (
        f"You're risking {stop_pct:g}% to make {target_pct:g}%, a {r:.2g}:1 reward-to-risk. "
        f"At that ratio you need to win more than {breakeven:.0%} of your trades just to break even, "
        f"before commissions and slippage."
    )

    return {
        "reward_to_risk": round(r, 4),
        "breakeven_win_rate": round(breakeven, 4),
        "plain_english": plain_english,
    }


if __name__ == "__main__":
    rules = apply_preset("Breakout")
    print("Preset rules:", rules)
    print("Readout:", compute_readout(rules)["plain_english"])
    save_rules(rules)
    save_strategy("My Breakout v1", rules, playbook="Trade opening-range breaks with volume confirmation.", origin="manual")
    print("Saved strategies:", list_strategies())
