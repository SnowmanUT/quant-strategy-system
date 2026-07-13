# Prop-Firm Consult Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent, multi-turn AI consult chat panel to the dashboard where the trader can ask about trading methods/models suited to prop-firm evaluation trading, grounded in their own audited findings.

**Architecture:** Widen `ai_client.py`'s single-turn `call_ai()` into a proper multi-turn `call_ai_conversation()` (both Anthropic and OpenAI-shaped providers already accept a `messages` array natively). Add a new `consult_layer.py` module (mirrors the existing `coaching_layer.py`) that builds a consult-specific system prompt — reusing `coaching_layer.build_distilled_findings()` plus a new daily P&L breakdown — and calls the widened client. Add three routes to `app.py` (`GET/POST /consult`, `POST /consult/clear`) backed by a JSON file (`data/consult_history.json`), serialized by a single process-wide lock shared with `/run`'s history-wipe-on-new-audit step. Add a chat panel to `dashboard_template.html` next to the existing Coach's Read panel, reusing its `mdToHtml()` renderer.

**Tech Stack:** Python 3 / Flask (existing), stdlib `threading` for the lock, stdlib `json`, no new dependencies. This repo has no test framework (`grep -rn "unittest\|pytest" *.py` finds nothing, no `tests/` dir) — its established verification pattern is ad-hoc scripts (see `ai_client.py`'s own `__main__` block) plus live browser/HTTP-driven end-to-end checks, which is what this plan's "test" steps use instead of introducing pytest as a new dependency.

## Global Constraints

- No new pip dependencies (spec: reuse stdlib + existing `requests`/`flask`).
- `call_ai()`'s existing signature and behavior must not change — Coach's Read (`coaching_layer.py`) and the plain-English strategy translator (`strategy_ai.py`) both call it unchanged.
- Consult history persists to `data/consult_history.json`, auto-wipes on every successful `/run`, and caps at the last 20 messages sent to the API per turn (spec: cost/context-limit control) while the full history still persists to disk and renders in the UI.
- `POST /consult` message input: reject empty/whitespace-only and anything over 4000 characters, both with HTTP 400, before any AI call or disk write.
- Route status-code conventions must match existing routes exactly: 400 for malformed input or missing prerequisite (`Run an audit first.`, matching `/coach`'s existing check), 200 with `{"ok": false, ...}` for the no-API-key case (matching `/coach`'s existing `"set a key to enable coaching"` string verbatim).
- `ai_available` must be computed live per-request (`ai_client.ai_available()`), never baked into the statically-rendered `dashboard.html`.

---

### Task 1: Widen `ai_client.py` for multi-turn conversations

**Files:**
- Modify: `ai_client.py:134-239` (the three `_call_*` functions and `call_ai`)

**Interfaces:**
- Consumes: nothing new (uses existing `resolve_key_and_provider()`, `get_model()`, `AIClientError`, `requests`).
- Produces: `call_ai_conversation(system_prompt: str, messages: list[dict], timeout=DEFAULT_TIMEOUT_SECONDS, max_tokens=DEFAULT_MAX_TOKENS) -> str`, where `messages` is `[{"role": "user"|"assistant", "content": str}, ...]` (caller's responsibility to strip any other keys like `ts` first). Raises `AIClientError` on no-key/bad-key/timeout/HTTP-error, same as today. `call_ai(system_prompt, user_message, timeout=..., max_tokens=...) -> str` keeps its exact current signature and behavior, now implemented as a one-message call to `call_ai_conversation`.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_ai_client.py`:

```python
"""Verify call_ai_conversation exists and both providers build correct
multi-turn request bodies, using unittest.mock so no network/API key is
needed. Run with: .venv/bin/python verify_ai_client.py"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/home/snowwhite/trade_auditor")

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
import ai_client

messages = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "follow-up question"},
]


def fake_anthropic_response(*args, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"content": [{"type": "text", "text": "reply text"}]}
    return resp


with patch("ai_client.requests.post", side_effect=fake_anthropic_response) as mock_post:
    result = ai_client.call_ai_conversation("system prompt here", messages)
    assert result == "reply text", f"expected 'reply text', got {result!r}"
    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["system"] == "system prompt here"
    assert sent_body["messages"] == messages, f"messages not passed through verbatim: {sent_body['messages']}"
    print("PASS: call_ai_conversation (anthropic) sends full message history")

# call_ai must still work unchanged: single-turn, same signature/behavior.
with patch("ai_client.requests.post", side_effect=fake_anthropic_response) as mock_post:
    result = ai_client.call_ai("system prompt", "single user message")
    assert result == "reply text"
    sent_body = mock_post.call_args.kwargs["json"]
    assert sent_body["messages"] == [{"role": "user", "content": "single user message"}]
    print("PASS: call_ai (single-turn) still works, delegates correctly")

os.environ["OPENAI_API_KEY"] = "sk-test-openai-key"
del os.environ["ANTHROPIC_API_KEY"]


def fake_openai_response(*args, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": "openai reply"}}]}
    return resp


with patch("ai_client.requests.post", side_effect=fake_openai_response) as mock_post:
    result = ai_client.call_ai_conversation("sys", messages)
    assert result == "openai reply"
    sent_body = mock_post.call_args.kwargs["json"]
    expected = [{"role": "system", "content": "sys"}] + messages
    assert sent_body["messages"] == expected, f"openai messages wrong: {sent_body['messages']}"
    print("PASS: call_ai_conversation (openai) prepends system + sends full history")

print("ALL PASS")
```

- [ ] **Step 2: Run it, confirm it fails with `AttributeError` (function doesn't exist yet)**

Run: `.venv/bin/python /tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_ai_client.py`
Expected: `AttributeError: module 'ai_client' has no attribute 'call_ai_conversation'`

- [ ] **Step 3: Implement `call_ai_conversation`, refactor the three `_call_*` functions and `call_ai`**

Replace `ai_client.py` lines 134-239 (from `def _call_anthropic` through the end of `call_ai`) with:

```python
def _call_anthropic(api_key, model, system_prompt, messages, timeout, max_tokens):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    resp = requests.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise AIClientError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    text = "".join(parts).strip()
    if not text:
        raise AIClientError("Anthropic API returned no text content.")
    return text


def _call_openai(api_key, model, system_prompt, messages, timeout, max_tokens):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }
    resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise AIClientError(f"OpenAI API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        raise AIClientError("OpenAI API returned an unexpected response shape.")
    if not text:
        raise AIClientError("OpenAI API returned no text content.")
    return text


def _call_deepseek(api_key, model, system_prompt, messages, timeout, max_tokens):
    # DeepSeek's chat completions API mirrors OpenAI's request/response shape.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
    }
    resp = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise AIClientError(f"DeepSeek API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        raise AIClientError("DeepSeek API returned an unexpected response shape.")
    if not text:
        raise AIClientError("DeepSeek API returned no text content.")
    return text


def call_ai_conversation(system_prompt, messages, timeout=DEFAULT_TIMEOUT_SECONDS, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Send a system prompt + a list of {"role": "user"|"assistant", "content": str}
    turns to whichever provider the resolved API key belongs to. Returns the
    raw text response. Raises AIClientError with one clear line on any
    failure (no key, bad key format, timeout, HTTP error).
    """
    api_key, provider = resolve_key_and_provider()
    if api_key is None:
        raise AIClientError(
            "No AI API key found (checked ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, "
            "OPENAI_API_KEY, AI_API_KEY, and ai_config.py). The plain-English path "
            "needs a key; presets and manual rule editing don't."
        )

    model = get_model(provider)

    try:
        if provider == "anthropic":
            return _call_anthropic(api_key, model, system_prompt, messages, timeout, max_tokens)
        if provider == "deepseek":
            return _call_deepseek(api_key, model, system_prompt, messages, timeout, max_tokens)
        return _call_openai(api_key, model, system_prompt, messages, timeout, max_tokens)
    except requests.exceptions.Timeout:
        raise AIClientError(f"{provider.title()} API request timed out after {timeout}s.")
    except requests.exceptions.ConnectionError as e:
        raise AIClientError(f"Could not reach the {provider.title()} API: {e}")
    except AIClientError:
        raise
    except Exception as e:
        raise AIClientError(f"Unexpected {provider.title()} API client error: {e}")


def call_ai(system_prompt, user_message, timeout=DEFAULT_TIMEOUT_SECONDS, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Send a system + single user message to whichever provider the resolved
    API key belongs to. Returns the raw text response. Raises AIClientError
    with one clear line on any failure (no key, bad key format, timeout,
    HTTP error).
    """
    return call_ai_conversation(
        system_prompt, [{"role": "user", "content": user_message}], timeout=timeout, max_tokens=max_tokens
    )
```

- [ ] **Step 4: Run the verification script again, confirm all four PASS lines print**

Run: `.venv/bin/python /tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_ai_client.py`
Expected:
```
PASS: call_ai_conversation (anthropic) sends full message history
PASS: call_ai (single-turn) still works, delegates correctly
PASS: call_ai_conversation (openai) prepends system + sends full history
ALL PASS
```

- [ ] **Step 5: Sanity-check the real DeepSeek key still works end to end (this repo has one configured in `.env`)**

Run: `.venv/bin/python ai_client.py`
Expected: `Provider: deepseek, model: deepseek-chat` (unchanged from before this task — confirms `resolve_key_and_provider`/`get_model` weren't touched).

- [ ] **Step 6: Commit**

```bash
cd /home/snowwhite && git add trade_auditor/ai_client.py
git commit -m "$(cat <<'EOF'
Widen ai_client.py for multi-turn conversations

call_ai_conversation() sends a full messages array to whichever
provider the resolved key belongs to; call_ai() now delegates to it
with a one-message list instead of duplicating the request-building
logic. Signature and behavior of call_ai() are unchanged -- Coach's
Read and the plain-English strategy translator keep working as-is.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Create `consult_layer.py`

**Files:**
- Create: `consult_layer.py`

**Interfaces:**
- Consumes: `ai_client.call_ai_conversation(system_prompt, messages, timeout, max_tokens) -> str`, `ai_client.AIClientError`, `ai_client.ai_available() -> bool` (all from Task 1); `coaching_layer.build_distilled_findings(dash_data: dict) -> dict` (existing, unmodified).
- Produces: `consult_layer.ConsultError(RuntimeError)`; `consult_layer.MAX_HISTORY_MESSAGES = 20`; `consult_layer.build_daily_pnl_summary(dash_data: dict) -> dict`; `consult_layer.build_consult_system_prompt(dash_data: dict) -> str`; `consult_layer.get_reply(message: str, history: list[dict], dash_data: dict) -> str` where `history` is `[{"role": ..., "content": ..., "ts": ...}, ...]` (the `ts` key is stripped internally) and the last element must be the current user turn. Raises `ConsultError` on no-key or AI failure.

- [ ] **Step 1: Write the failing verification script**

Create `/tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_consult_layer.py`:

```python
"""Verify consult_layer's pure-logic pieces (no network). Run with:
.venv/bin/python verify_consult_layer.py"""
import sys
sys.path.insert(0, "/home/snowwhite/trade_auditor")

import consult_layer

# --- build_daily_pnl_summary ---
fake_dash_data = {
    "meta": {"n_trades": 4, "declared_family": None, "strategy_playbook": None, "generated_at": "2026-07-12T00:00:00Z"},
    "quant_stats": {
        "expectancy_r": {"mean": 0.1, "se": 0.05, "distinguishable_from_zero": False},
        "sqn": {"value": 1.0}, "sharpe_per_trade": {"value": 0.1},
        "payoff_ratio": {"value": 1.5}, "max_drawdown_dollars": {"value": -100},
        "kelly_fraction": {"value": 0.05},
    },
    "behavioral": {"luck": {"concentration_top_decile_pct_of_gross_winning": 50, "pnl_without_top_n": {}},
                   "plan_adherence": {"adherence_rate": 0.8}},
    "discipline_cost": {"aggregate": {}},
    "leaks": [],
    "verdict": {"sentence": "test verdict"},
    "style_panel": {"declared_family": None, "observed_majority_family": None, "verdict": None},
    "trade_replay": {
        "days": {
            "MNQ=F|2026-07-01": {"date": "2026-07-01", "trades": [{"actual_profit": 100}, {"actual_profit": -20}]},
            "MNQ=F|2026-07-02": {"date": "2026-07-02", "trades": [{"actual_profit": -50}]},
        }
    },
}

summary = consult_layer.build_daily_pnl_summary(fake_dash_data)
assert summary["n_days"] == 2, summary
assert summary["best_day"]["net_pnl"] == 80.0, summary  # 100 - 20
assert summary["worst_day"]["net_pnl"] == -50.0, summary
# gross profit = 100 (only positive trades), best day net 80 -> 80/100 = 80%
assert summary["best_day_pct_of_gross_profit"] == 80.0, summary
print("PASS: build_daily_pnl_summary computes best/worst day and consistency %")

# --- build_consult_system_prompt ---
prompt = consult_layer.build_consult_system_prompt(fake_dash_data)
assert "daily loss" in prompt.lower(), "system prompt missing daily-loss domain knowledge"
assert "trailing" in prompt.lower(), "system prompt missing trailing-drawdown domain knowledge"
assert "consistency" in prompt.lower(), "system prompt missing consistency-rule domain knowledge"
assert '"n_days": 2' in prompt, "findings JSON (with daily summary) not embedded in system prompt"
print("PASS: build_consult_system_prompt embeds domain knowledge + findings JSON")

# --- get_reply: no-key path (no network call should happen) ---
import os
for var in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "AI_API_KEY"):
    os.environ.pop(var, None)
import ai_client
# also blank out ai_config fallback for this check
import ai_config
ai_config.API_KEY = ""
try:
    consult_layer.get_reply("hello", [{"role": "user", "content": "hello", "ts": "x"}], fake_dash_data)
    raise SystemExit("FAIL: expected ConsultError with no key configured")
except consult_layer.ConsultError as e:
    assert "key" in str(e).lower()
    print("PASS: get_reply raises ConsultError with no key configured")

print("ALL PASS")
```

- [ ] **Step 2: Run it, confirm it fails with `ModuleNotFoundError`**

Run: `.venv/bin/python /tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_consult_layer.py`
Expected: `ModuleNotFoundError: No module named 'consult_layer'`

- [ ] **Step 3: Implement `consult_layer.py`**

Create `consult_layer.py`:

```python
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
```

- [ ] **Step 4: Run the verification script again, confirm all PASS lines print**

Run: `.venv/bin/python /tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_consult_layer.py`
Expected:
```
PASS: build_daily_pnl_summary computes best/worst day and consistency %
PASS: build_consult_system_prompt embeds domain knowledge + findings JSON
PASS: get_reply raises ConsultError with no key configured
ALL PASS
```

- [ ] **Step 5: Commit**

```bash
cd /home/snowwhite && git add trade_auditor/consult_layer.py
git commit -m "$(cat <<'EOF'
Add consult_layer.py for the prop-firm consult chat

Mirrors coaching_layer.py's structure but is consultative rather than
verdict-style: free to suggest any trading method, grounded in the
same distilled findings plus a new daily P&L breakdown (daily loss
limits, trailing drawdown, and consistency rules are all daily
concepts the coach's existing distillation didn't cover).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `/consult` routes to `app.py`, wire the auto-wipe into `/run`

**Files:**
- Modify: `app.py` (imports at top, new routes, `/run` handler)

**Interfaces:**
- Consumes: `consult_layer.get_reply(message, history, dash_data, timeout) -> str`, `consult_layer.ConsultError` (Task 2); `ai_client.ai_available() -> bool` (Task 1); existing `DATA_DIR`, `DASH_DATA_PATH` module constants.
- Produces: `GET /consult` -> `{"ok": true, "history": [...], "ai_available": bool}`. `POST /consult {"message": str}` -> `{"ok": true, "reply": str, "history": [...]}` on success, `{"ok": false, "error": str}` on failure (400 for bad input / no audit yet, 200 for no-key / AI failure / stale-run). `POST /consult/clear` -> `{"ok": true}`. `CONSULT_HISTORY_PATH` module constant (for Task 4's manual testing reference only, not imported by the frontend).

- [ ] **Step 1: Write the failing verification script**

This one drives the real Flask app over HTTP (matches how every route in this app has been verified all session — no route in `app.py` has a unit test today, they're all verified live). Create `/tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_consult_routes.py`:

```python
"""Verify the /consult routes end-to-end against a running trade_auditor
server (start it first: .venv/bin/python run.py, in another terminal / the
Claude Preview tool). Assumes an audit has already been run (dash_data.json
exists) and an API key is configured (.env's DEEPSEEK_API_KEY in this repo).
Run with: .venv/bin/python verify_consult_routes.py"""
import json
import urllib.request

BASE = "http://127.0.0.1:5050"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# 1. Clear any prior history so this run is deterministic.
status, data = call("POST", "/consult/clear")
assert status == 200 and data["ok"], data
print("PASS: /consult/clear")

# 2. GET with empty history.
status, data = call("GET", "/consult")
assert status == 200 and data["ok"] and data["history"] == [], data
assert "ai_available" in data, data
print(f"PASS: GET /consult empty history, ai_available={data['ai_available']}")

# 3. Empty message rejected with 400.
status, data = call("POST", "/consult", {"message": "   "})
assert status == 400 and data["ok"] is False, (status, data)
print("PASS: empty message rejected with 400")

# 4. Oversized message rejected with 400.
status, data = call("POST", "/consult", {"message": "x" * 4001})
assert status == 400 and data["ok"] is False, (status, data)
print("PASS: oversized message rejected with 400")

# 5. Real message -- requires dash_data.json to exist and a key configured.
status, data = call("POST", "/consult", {"message": "What's one method worth testing given my numbers?"})
assert status == 200, (status, data)
if not data["ok"]:
    print(f"SKIP (expected if no audit run yet or no key): {data['error']}")
else:
    assert "reply" in data and len(data["reply"]) > 0, data
    assert len(data["history"]) == 2, data["history"]
    assert data["history"][0]["role"] == "user"
    assert data["history"][1]["role"] == "assistant"
    print("PASS: POST /consult returns a reply and persists both turns")

    # 6. GET again -- history should still be there.
    status, data = call("GET", "/consult")
    assert status == 200 and len(data["history"]) == 2, data
    print("PASS: history persisted, visible on GET")

    # 7. Clear -- history empties.
    status, data = call("POST", "/consult/clear")
    assert status == 200 and data["ok"], data
    status, data = call("GET", "/consult")
    assert data["history"] == [], data
    print("PASS: clear empties history")

print("ALL PASS (or SKIP if no audit/key -- rerun after /run and with a key configured)")
```

- [ ] **Step 2: Start the server and run it, confirm 404s (routes don't exist yet)**

Start the server (Claude Preview `preview_start` on the `trade-auditor` launch config, or `.venv/bin/python run.py` directly), then:
Run: `.venv/bin/python /tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_consult_routes.py`
Expected: fails on the first call with a 404 (`/consult/clear` doesn't exist yet).

- [ ] **Step 3: Implement the routes**

In `app.py`, add to the imports block (after the existing `import trades_io` at line 37):

```python
import json
import threading
from datetime import datetime, timezone

import consult_layer
```

After the existing `DASHBOARD_TEMPLATE_PATH = ...` / `SETUP_HTML_PATH = ...` constants (around line 51), add:

```python
CONSULT_HISTORY_PATH = os.path.join(DATA_DIR, "consult_history.json")
CONSULT_MAX_MESSAGE_CHARS = 4000
_consult_lock = threading.Lock()
```

After `app = Flask(__name__)` (line 53), add the helper functions:

```python
def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_consult_history():
    if not os.path.exists(CONSULT_HISTORY_PATH):
        return {"run_id": None, "history": []}
    try:
        with open(CONSULT_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"run_id": None, "history": []}


def _save_consult_history(data):
    with open(CONSULT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _read_dash_data():
    """None if missing or mid-write (rare, since build_dashboard.py writes it directly)."""
    if not os.path.exists(DASH_DATA_PATH):
        return None
    try:
        with open(DASH_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
```

Add the three new routes right after the existing `/coach` route (after line 317, before the `# DASHBOARD` section comment):

```python
# --------------------------------------------------------------------------
# CONSULT -- optional, AI (purple). Multi-turn, persisted, grounded in the
# current run's audited findings. Auto-clears whenever a fresh /run lands.
# --------------------------------------------------------------------------

@app.route("/consult", methods=["GET"])
def consult_get():
    data = _load_consult_history()
    return jsonify({"ok": True, "history": data["history"], "ai_available": ai_client.ai_available()})


@app.route("/consult", methods=["POST"])
def consult_post():
    payload = request.get_json(force=True, silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Message cannot be empty."}), 400
    if len(message) > CONSULT_MAX_MESSAGE_CHARS:
        return jsonify({"ok": False, "error": f"Message too long (max {CONSULT_MAX_MESSAGE_CHARS} characters)."}), 400

    dash_data = _read_dash_data()
    if dash_data is None:
        return jsonify({"ok": False, "error": "Run an audit first."}), 400

    if not ai_client.ai_available():
        return jsonify({"ok": False, "error": "set a key to enable coaching"}), 200

    run_id = dash_data["meta"]["generated_at"]

    # Single lock serializes every /consult call and /run's history wipe --
    # for a local single-user tool, holding it across the AI call (up to
    # ~45s) is a simpler and safer guarantee than fine-grained locking: it
    # makes "a /run lands mid-conversation" and "two tabs post at once"
    # both structurally impossible instead of merely detected-and-handled.
    with _consult_lock:
        data = _load_consult_history()
        if data.get("run_id") != run_id:
            data = {"run_id": run_id, "history": []}
        working_history = data["history"] + [{"role": "user", "content": message, "ts": _now_iso()}]

        try:
            reply = consult_layer.get_reply(message, working_history, dash_data)
        except consult_layer.ConsultError as e:
            return jsonify({"ok": False, "error": str(e)}), 200

        # Belt-and-suspenders: re-check in case a future change narrows the
        # lock's scope around the AI call. Currently unreachable since /run
        # can't wipe while this lock is held.
        current_dash_data = _read_dash_data()
        current_run_id = current_dash_data["meta"]["generated_at"] if current_dash_data else None
        if current_run_id != run_id:
            return jsonify({"ok": False, "error": "Audit was rerun while waiting for a reply -- ask again."}), 200

        data["run_id"] = run_id
        data["history"] = working_history + [{"role": "assistant", "content": reply, "ts": _now_iso()}]
        _save_consult_history(data)
        final_history = data["history"]

    return jsonify({"ok": True, "reply": reply, "history": final_history})


@app.route("/consult/clear", methods=["POST"])
def consult_clear():
    dash_data = _read_dash_data()
    run_id = dash_data["meta"]["generated_at"] if dash_data else None
    with _consult_lock:
        _save_consult_history({"run_id": run_id, "history": []})
    return jsonify({"ok": True})
```

In `/run` (`run_pipeline`), after the pipeline's `try`/`except` block succeeds -- i.e. right after the line `dash_data = build_dashboard.build_dashboard(...)` inside the `try`, but the wipe itself goes **after** the `except` blocks (so a failed run never wipes a valid conversation). Find this existing code (around line 257-262):

```python
    except trades_io.TradesFileError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Pipeline failed: {e}", "trace": traceback.format_exc()}), 500

    headline = _print_headline(dash_data, warnings)
```

Insert one line between them:

```python
    except trades_io.TradesFileError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Pipeline failed: {e}", "trace": traceback.format_exc()}), 500

    with _consult_lock:
        _save_consult_history({"run_id": dash_data["meta"]["generated_at"], "history": []})

    headline = _print_headline(dash_data, warnings)
```

Also update the module docstring's route list at the top of `app.py` (lines 8-18) to add:

```python
  GET  /consult              current consult chat history + live ai_available flag
  POST /consult               send a message, get a reply, persist both turns
  POST /consult/clear          wipe the consult history (also happens automatically on /run)
```

- [ ] **Step 4: Restart the server, run the verification script again**

Restart (stop + `preview_start` again, or re-run `run.py` -- Flask's dev server isn't in debug/reload mode, matches how every prior change in this app was verified this session).
Run: `.venv/bin/python /tmp/claude-1000/-home-snowwhite-trade-auditor/2af8e123-856b-4311-aca7-e2500acb7cae/scratchpad/verify_consult_routes.py`
Expected: all `PASS` lines (or `SKIP` with a clear reason if no audit has been run yet against this server instance -- in that case, run an audit first via the wizard or a direct `/run` call, matching this session's established pattern, then rerun this script).

- [ ] **Step 5: Manually verify the auto-wipe-on-/run behavior**

With a conversation already in `data/consult_history.json` (from Step 4), trigger a fresh `/run` (through the wizard, or `POST /run` with valid `data_source`/`rules` payload as used earlier this session), then:
Run: `.venv/bin/python -c "import json; print(json.load(open('/home/snowwhite/trade_auditor/data/consult_history.json'))['history'])"`
Expected: `[]` (wiped).

- [ ] **Step 6: Commit**

```bash
cd /home/snowwhite && git add trade_auditor/app.py
git commit -m "$(cat <<'EOF'
Add /consult routes: GET, POST, POST /consult/clear

Multi-turn chat backed by data/consult_history.json, serialized by a
single process-wide lock shared with /run's auto-wipe (a fresh audit
clears any conversation grounded in the previous one's numbers). The
lock is held across the AI call itself -- for a local single-user
tool this makes races structurally impossible rather than merely
detected, at the cost of a /run request blocking behind an in-flight
consult reply in the rare case both happen at once.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add the consult chat panel to `dashboard_template.html`

**Files:**
- Modify: `dashboard_template.html` (CSS block, HTML after the Coach's Read panel, JS at the bottom)

**Interfaces:**
- Consumes: `GET /consult`, `POST /consult`, `POST /consult/clear` JSON contracts from Task 3; existing global `mdToHtml(md: str) -> str` function (already defined, used by the Coach's Read panel).
- Produces: `#consultPanel` DOM section; `initConsult()` JS function, called once from the existing `(function init(){...})()` IIFE at the bottom of the file.

- [ ] **Step 1: Manual "failing" check -- confirm the panel doesn't exist yet**

With the server running and a dashboard already generated, open `/dashboard` in a browser (or via the Claude Preview tool) and confirm there's no consult panel/chat UI anywhere on the page -- only the existing Coach's Read panel.

- [ ] **Step 2: Add CSS**

In `dashboard_template.html`, after the existing `/* ---- coach panel ---- */` CSS block (ends around line 95, right before `/* ---- KPI row ---- */`), add:

```css
  /* ---- consult panel ---- */
  #consultMessages{
    max-height:340px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
    margin-bottom:12px;padding-right:4px;
  }
  #consultMessages.empty{color:var(--muted);font-style:italic;}
  .consult-msg{padding:8px 12px;border-radius:6px;font-size:12.5px;max-width:85%;}
  .consult-msg.user{align-self:flex-end;background:var(--cyan-dim);color:var(--text);}
  .consult-msg.assistant{align-self:flex-start;background:var(--purple-dim);color:var(--text);}
  .consult-msg p{margin:0 0 6px 0;}
  .consult-msg p:last-child{margin-bottom:0;}
  #consultInputRow{display:flex;gap:8px;}
  #consultInput{
    flex:1;background:var(--panel2);border:1px solid var(--border);color:var(--text);
    font-family:inherit;font-size:12.5px;padding:8px 10px;border-radius:4px;resize:vertical;min-height:38px;
  }
  #consultSendBtn{
    background:var(--purple-dim);color:var(--purple);border:1px solid var(--purple);
    padding:7px 14px;border-radius:4px;font-family:inherit;font-size:11px;cursor:pointer;
    letter-spacing:.05em;text-transform:uppercase;
  }
  #consultSendBtn:hover{background:var(--purple);color:#0a0510;}
  #consultSendBtn:disabled{opacity:.5;cursor:default;}
  #consultClearBtn{
    background:none;color:var(--muted);border:1px solid var(--border);
    padding:7px 14px;border-radius:4px;font-family:inherit;font-size:11px;cursor:pointer;
  }
  #consultClearBtn:hover{color:var(--text);border-color:var(--muted);}
  #consultStatus{font-size:11px;color:var(--muted);margin-top:8px;}
```

- [ ] **Step 3: Add the panel HTML**

Right after the existing Coach's Read panel's closing `</div>` (line 214, immediately before `<div class="kpi-row" id="kpiRow"></div>`), add:

```html
<div class="panel">
  <h2>Consult <span class="tag ai">AI Coaching</span></h2>
  <div id="consultMessages" class="empty">No messages yet -- ask about trading methods or models suited to your prop-firm rules.</div>
  <div id="consultInputRow">
    <textarea id="consultInput" placeholder="e.g. I'm on an Apex 50k combine, 2500 trailing drawdown -- what should I test?"></textarea>
    <button id="consultSendBtn">Send</button>
    <button id="consultClearBtn">Clear</button>
  </div>
  <div id="consultStatus"></div>
</div>
```

- [ ] **Step 4: Add the JS**

Right before the closing `(function init(){...})()` IIFE at the bottom of the file (before line 820's `(function init(){`), add:

```javascript
// ============================================================================
// consult panel: multi-turn, persisted via GET/POST /consult
// ============================================================================
function renderConsultMessages(history){
  const box = document.getElementById("consultMessages");
  if(!history || history.length === 0){
    box.className = "empty";
    box.textContent = "No messages yet -- ask about trading methods or models suited to your prop-firm rules.";
    return;
  }
  box.className = "";
  box.innerHTML = history.map(m =>
    `<div class="consult-msg ${m.role}">${mdToHtml(m.content)}</div>`
  ).join("");
  box.scrollTop = box.scrollHeight;
}

async function initConsult(){
  const input = document.getElementById("consultInput");
  const sendBtn = document.getElementById("consultSendBtn");
  const clearBtn = document.getElementById("consultClearBtn");
  const status = document.getElementById("consultStatus");

  async function loadHistory(){
    try{
      const resp = await fetch("/consult");
      const data = await resp.json();
      renderConsultMessages(data.history || []);
      if(!data.ai_available){
        input.disabled = true; sendBtn.disabled = true;
        status.innerHTML = `<span class="amber">set a key to enable coaching</span>`;
      }
    }catch(err){
      status.innerHTML = `<span class="red">Could not reach the server.</span>`;
    }
  }

  sendBtn.addEventListener("click", async ()=>{
    const message = input.value.trim();
    if(!message) return;
    sendBtn.disabled = true; input.disabled = true;
    status.textContent = "Thinking...";
    try{
      const resp = await fetch("/consult", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message}),
      });
      const data = await resp.json().catch(()=>({}));
      if(!resp.ok || data.ok === false){
        status.innerHTML = `<span class="amber">${data.error || ("HTTP "+resp.status)}</span>`;
        return;  // leave the typed message in the input for retry -- not persisted server-side
      }
      renderConsultMessages(data.history);
      input.value = "";
      status.textContent = "";
    }catch(err){
      status.innerHTML = `<span class="red">Consult endpoint isn't reachable</span> (expects POST /consult from run.py's Flask app -- make sure it's running).`;
    }finally{
      sendBtn.disabled = false; input.disabled = false;
    }
  });

  clearBtn.addEventListener("click", async ()=>{
    clearBtn.disabled = true;
    try{
      await fetch("/consult/clear", {method: "POST"});
      renderConsultMessages([]);
      status.textContent = "";
    }catch(err){
      status.innerHTML = `<span class="red">Could not clear.</span>`;
    }finally{
      clearBtn.disabled = false;
    }
  });

  await loadHistory();
}
```

Then add `initConsult();` as a new line inside the existing `(function init(){...})()` IIFE, after `renderDiscipline();`:

```javascript
(function init(){
  if(!DASH){ document.body.innerHTML = '<p style="padding:40px;color:#f87171;">No dashboard data embedded -- run build_dashboard.py to generate dashboard.html.</p>'; return; }
  renderMeta();
  renderVerdict();
  renderKpis();
  renderQuantStats();
  renderReplay();
  renderConstellation();
  renderStylePanel();
  renderEquityAndHour();
  renderLeaks();
  renderDiscipline();
  initConsult();
})();
```

- [ ] **Step 5: Regenerate the dashboard and verify visually**

The template is only rendered into `data/dashboard.html` at `/run` time, so trigger a fresh `/run` (through the wizard, using data already run earlier this session), then load `/dashboard` in the browser.
Expected: a "Consult" panel appears below Coach's Read, matching its visual style, with an empty-state message, a textarea, Send, and Clear buttons.

- [ ] **Step 6: Commit**

```bash
cd /home/snowwhite && git add trade_auditor/dashboard_template.html
git commit -m "$(cat <<'EOF'
Add consult chat panel to the dashboard

Sits next to Coach's Read, same terminal-style theme, reuses its
mdToHtml() renderer. Loads persisted history on page load, greys out
with the existing no-key messaging when ai_available is false.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: End-to-end verification

**Files:** none (verification only, matches this session's established E2E approach: real data already in this repo, driven through the Claude Preview browser tools).

**Interfaces:** none new -- exercises everything from Tasks 1-4 together.

- [ ] **Step 1: Full run**

Start the server, run a fresh audit using the real Tradovate data already converted earlier this session (`data/trades.csv` from the `Performance-acct3/4/5.csv` multi-file upload, or rerun the upload), confirm the dashboard loads with both Coach's Read and the new Consult panel.

- [ ] **Step 2: Multi-turn conversation with real grounding**

In the Consult panel, send at least 3 messages, including one that forces a callback to a specific number (e.g. "what's my current plan adherence rate and does that matter for a prop firm?"). Confirm the reply cites the real number from the findings (matches this session's established anti-hallucination check on Coach's Read).

- [ ] **Step 3: Persistence across refresh**

Refresh the dashboard page. Confirm the full conversation still renders (via `GET /consult` on load).

- [ ] **Step 4: Persistence across restart**

Stop and restart the server. Confirm `GET /consult` still returns the same history (disk persistence, not just in-memory).

- [ ] **Step 5: Auto-clear on rerun**

Trigger a fresh `/run`. Confirm the Consult panel is empty again on the next dashboard load.

- [ ] **Step 6: Manual clear**

Start a new conversation (at least one exchange), then click "Clear". Confirm it empties immediately without needing a page refresh.

- [ ] **Step 7: No-key behavior**

Temporarily rename `.env` (e.g. `mv .env .env.bak`), restart the server, reload the dashboard. Confirm the Consult panel is greyed out with "set a key to enable coaching" and the input is disabled. Restore: `mv .env.bak .env`.

- [ ] **Step 8: Confirm Coach's Read and the plain-English strategy translator still work unchanged**

Click "Ask the coach" on the same dashboard -- confirm it still returns a normal one-shot review (proves `call_ai()`'s refactor in Task 1 didn't break its existing caller). If convenient, also exercise Step 2's plain-English strategy path in the setup wizard for the same reason.
