# Prop-Firm Consult Chat — Design

## Problem

The existing Coach's Read (`/coach`) is a one-shot, retrospective performance
review: it restates numbers already in `dash_data.json` as a blunt verdict.
It doesn't take input from the trader and can't hold a conversation, so it
can't help with a forward-looking question like "given my audited behavior
and this specific prop firm's rules, what trading methods should I actually
run?"

## Goal

A separate, persistent, multi-turn chat panel on the dashboard where the
trader can consult an AI about trading methods/models suited to prop-firm
evaluation trading -- grounded in their own audited findings, not tied to
the app's existing preset library, with baseline domain knowledge about how
prop-firm evaluations work baked into the system prompt.

## Non-goals

- Not replacing or modifying Coach's Read -- both panels coexist.
- Not constraining suggestions to the 12 strategy-family presets in
  `rules.py` -- the AI can suggest anything.
- Not building a generic chat framework -- this is one purpose-built panel.

## Architecture & data flow

```
Dashboard load  --GET /consult-->  render saved history (if any)
User sends msg  --POST /consult {message}-->
    1. load data/consult_history.json (or start empty)
    2. append {role: "user", content: message, ts}
    3. rebuild distilled findings fresh from CURRENT data/dash_data.json
    4. build system prompt = CONSULT_SYSTEM_PROMPT + findings JSON block
    5. call ai_client.call_ai_conversation(system_prompt, history_as_messages)
    6. append {role: "assistant", content: reply, ts}
    7. persist history to disk
    8. return reply to frontend
"Clear conversation" / a completed /run  -->  wipe consult_history.json
```

Findings are rebuilt from disk on every single `/consult` call and folded
into the system prompt (not sent as a chat turn) -- so a mid-conversation
`/run` produces fresh grounding on the very next message, and doesn't
consume turn history with a growing findings block.

`/run` (in `app.py`) wipes `data/consult_history.json` at the point it
regenerates `dash_data.json`, since a stale conversation about a previous
audit's numbers is actively misleading.

## Storage

`data/consult_history.json`:
```json
{
  "history": [
    {"role": "user", "content": "...", "ts": "2026-07-12T18:20:00"},
    {"role": "assistant", "content": "...", "ts": "2026-07-12T18:20:04"}
  ]
}
```
Same directory/pattern as `data/rules.json`, `data/strategies.json`.

## Backend changes

### `ai_client.py`

Add `call_ai_conversation(system_prompt, messages, timeout=..., max_tokens=...)`
where `messages` is `[{"role": "user"|"assistant", "content": "..."}]`.
Both Anthropic's Messages API and OpenAI/DeepSeek's Chat Completions API
accept a `messages` array natively -- this is a straight widening of the
existing `_call_anthropic` / `_call_openai` / `_call_deepseek` request
bodies, not a new code path. Keep `call_ai()` (single-turn) as-is for
Coach's Read and the plain-English strategy translator, which don't need
history.

### New `consult_layer.py` (mirrors `coaching_layer.py`)

- `CONSULT_SYSTEM_PROMPT`: consultative tone (not verdict-report), baseline
  prop-firm evaluation domain knowledge (daily loss limits, trailing vs.
  end-of-day drawdown, consistency rules, profit targets, scaling plans),
  same anti-hallucination rule for citing real numbers from findings, same
  "can't verify discretionary setups from price alone" caveat, explicitly
  free to suggest any trading method/model (not limited to this app's
  preset library), instructed to ask clarifying questions about the
  trader's specific firm/rules when it needs more than the trader has
  given it.
- Reuses `coaching_layer.build_distilled_findings(dash_data)` for the
  findings block (no duplication).
- `get_reply(user_message, history, dash_data)` -> builds system prompt,
  calls `ai_client.call_ai_conversation`, returns reply text. Raises the
  same `CoachingError`-style exception on no-key/failure.

### `app.py` routes

- `GET /consult` -> `{ok: true, history: [...]}` (empty list if no file yet).
- `POST /consult {message}` -> validates `dash_data.json` exists (else
  `{ok:false, error:"Run an audit first."}`), validates key available (else
  same `"set a key to enable coaching"` shape as `/coach`), on success
  appends both turns and persists, returns `{ok:true, reply: "...", history: [...]}`.
  On AI failure: returns `{ok:false, error: "..."}` and does NOT persist
  the failed turn (user's message stays in the input box client-side for
  retry, per frontend behavior below).
- `POST /consult/clear` -> truncates `consult_history.json` to `{"history": []}`,
  returns `{ok:true}`.
- Inside the existing `/run` handler: after `dash_data.json` is written,
  also reset `consult_history.json` to `{"history": []}`.

## Frontend changes

`dashboard_template.html`: new panel next to Coach's Read (same step-3
area), matching the existing terminal-style theme:

- Scrollable message list, user/assistant turns visually distinguished.
- Text input + send button + "Clear conversation" button.
- On dashboard load: `GET /consult`, render existing history.
- On send: disable input, POST, append reply on success; on failure show
  inline error and leave the typed message in the input for retry (not
  cleared, not persisted).
- NO-KEY MODE: same greyed-out treatment as Coach's Read when
  `ai_available()` is false server-side (surfaced via the existing
  `/coach`-style error string, or a lightweight `ai_available` flag added
  to the dashboard's initial payload -- implementation detail for the plan).

## Error handling

| Condition | Behavior |
|---|---|
| No API key configured | `{ok:false, error:"set a key to enable coaching"}`, panel greyed out |
| No `dash_data.json` yet | `{ok:false, error:"Run an audit first."}` |
| AI call fails (timeout/HTTP/bad key) | `{ok:false, error:"..."}`, failed turn not persisted, chat state otherwise untouched |
| Clear conversation | Always succeeds, wipes to empty history |
| New `/run` completes | Auto-wipes consult history as a side effect |

## Testing / verification

After implementation, drive it end-to-end in the browser against real data
already in this repo (`Performance-acct3/4/5.csv` via the Tradovate import
path built earlier this session):
1. Run an audit.
2. Open the consult panel, hold an actual multi-turn conversation (at
   least 3 exchanges, including one that requires the model to reference a
   specific number from the findings).
3. Refresh the page -- confirm history survived.
4. Restart the server -- confirm history survived (disk persistence).
5. Rerun the audit -- confirm consult history auto-cleared.
6. Hit "Clear conversation" manually -- confirm it empties immediately.
7. Confirm behavior with no API key configured (greyed out, clear message).
