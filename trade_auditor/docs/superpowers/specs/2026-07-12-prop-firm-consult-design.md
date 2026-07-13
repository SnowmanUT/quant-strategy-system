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
Dashboard load  --GET /consult-->  render saved history (if any) + ai_available flag
User sends msg  --POST /consult {message}-->
    1. acquire module-level threading.Lock (serializes all /consult and /run
       writers -- app.py already runs Flask's dev server threaded)
    2. load data/consult_history.json (or start empty); read its run_id
    3. append {role: "user", content: message, ts}, capped at 4000 chars,
       rejected with 400 if empty/whitespace-only
    4. rebuild distilled findings fresh from CURRENT data/dash_data.json
       (also read CURRENT run_id from dash_data.json's meta)
    5. build system prompt = CONSULT_SYSTEM_PROMPT + findings JSON block
    6. take the last 20 messages (bare {role, content} only -- strip ts
       before sending), call
       ai_client.call_ai_conversation(system_prompt, capped_messages)
    7. re-check run_id against the CURRENT dash_data.json's run_id; if it
       changed while the AI call was in flight (a /run landed mid-request),
       DISCARD this reply, release the lock, and return
       {ok:false, error:"Audit was rerun while waiting for a reply -- ask again."}
       instead of resurrecting a stale-grounded turn
    8. otherwise append {role: "assistant", content: reply, ts}, persist to
       disk, release the lock, return reply to frontend
"Clear conversation" / a completed /run  -->  wipe consult_history.json
    (also under the same lock)
```

Findings are rebuilt from disk on every single `/consult` call and folded
into the system prompt (not sent as a chat turn) -- so a mid-conversation
`/run` produces fresh grounding on the very next message, and doesn't
consume turn history with a growing findings block. Only the last 20
messages are sent to the API per turn (full history still persists to disk
and renders in the UI) -- caps both cost and the risk of exceeding a
provider's context limit on a long-running conversation.

`/run` (in `app.py`) wipes `data/consult_history.json` immediately after
`dash_data.json` is successfully written (i.e. after the pipeline's `try`
block succeeds, not inside it) -- a failed pipeline run must not wipe an
otherwise-valid conversation. The lock in step 1 above is what prevents
this wipe from racing a `/consult` request that's already mid-flight: both
the wipe and the read-modify-write in `/consult` hold the same lock, and
the run_id check in step 7 catches the case where `/consult` started
before the wipe but would otherwise finish after it.

## Storage

`data/consult_history.json`:
```json
{
  "run_id": "<dash_data.json's meta.run_id, or its mtime if no run_id field exists>",
  "history": [
    {"role": "user", "content": "...", "ts": "2026-07-12T18:20:00"},
    {"role": "assistant", "content": "...", "ts": "2026-07-12T18:20:04"}
  ]
}
```
Same directory/pattern as `data/rules.json`, `data/strategies.json`. Check
during planning whether `dash_data.json`'s `meta` already carries a run
identifier or generation timestamp (build_dashboard.py) to reuse as
`run_id` -- only add a new one if nothing suitable already exists.

## Backend changes

### `ai_client.py`

Add `call_ai_conversation(system_prompt, messages, timeout=..., max_tokens=...)`
where `messages` is `[{"role": "user"|"assistant", "content": "..."}]` --
caller's responsibility to have already stripped any non-`{role,content}`
keys (like `ts`) before passing in, since Anthropic's API rejects unknown
message keys. Both Anthropic's Messages API and OpenAI/DeepSeek's Chat
Completions API accept a `messages` array natively -- this is a straight
widening of the existing `_call_anthropic` / `_call_openai` / `_call_deepseek`
request bodies, not a new code path. Refactor `call_ai()` (single-turn) to
delegate to `call_ai_conversation()` with a one-message list, so there's
one request-building code path instead of two. Coach's Read and the
plain-English strategy translator keep calling `call_ai()` unchanged.

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
  findings block (no duplication), and additionally builds a
  `daily_pnl_summary` (n trading days, best day $, worst day $, best day's
  % of total gross profit) from `dash_data["trade_replay"]["days"]` --
  daily loss limits, trailing drawdown, and consistency/best-day-%  rules
  are all fundamentally daily concepts, and the coach's existing
  distillation has no daily breakdown at all. This summary is
  consult-only; `coaching_layer.py` and Coach's Read are untouched.
- `get_reply(message, history, dash_data)` -> builds system prompt (findings
  + daily_pnl_summary), takes the last 20 entries of `history` and strips
  them to bare `{role, content}`, calls `ai_client.call_ai_conversation`,
  returns reply text. Raises the same `CoachingError`-style exception on
  no-key/failure. Timeout matches Coach's Read (45s).

### `app.py` routes

- `GET /consult` -> `{ok: true, history: [...], ai_available: bool}`.
  `ai_available` is computed live via `ai_client.ai_available()` on every
  call -- NOT baked into the static `dashboard.html` payload, since that
  file is written once at `/run` time (see app.py's `send_file` of a
  pre-rendered dashboard) and would go stale if the user adds/removes a
  key afterward without rerunning.
- `POST /consult {message}` -> 400 if `message` missing/blank/over 4000
  chars (matches `/upload`'s validation-error convention: 400 for
  malformed input). Then: `dash_data.json` missing -> 400
  `{ok:false, error:"Run an audit first."}` (matches `/upload`'s pattern,
  not `/coach`'s 200). No key -> 200 `{ok:false, error:"set a key to
  enable coaching"}` (matches `/coach`'s existing convention exactly, since
  this is a normal/expected state, not a client error). On success:
  acquire the lock, run the flow in "Architecture & data flow" above,
  return `{ok:true, reply: "...", history: [...]}`. On AI failure or a
  run_id mismatch: `{ok:false, error:"..."}`, failed turn not persisted,
  input stays in the box client-side for retry.
- `POST /consult/clear` -> under the same lock, truncates
  `consult_history.json` to `{"run_id": <current>, "history": []}`,
  returns `{ok:true}`.
- Inside the existing `/run` handler: after the pipeline's `try` block
  succeeds and `dash_data.json` is written (not inside the try, so a
  failed run doesn't wipe a valid conversation), under the same lock,
  reset `consult_history.json` to `{"run_id": <new run_id>, "history": []}`.

## Frontend changes

`dashboard_template.html`: new panel next to Coach's Read (same step-3
area), matching the existing terminal-style theme:

- Scrollable message list, user/assistant turns visually distinguished.
- Text input + send button + "Clear conversation" button.
- On dashboard load: `GET /consult`, render existing history.
- On send: disable input, POST, append reply on success; on failure show
  inline error and leave the typed message in the input for retry (not
  cleared, not persisted).
- Assistant replies render through the same markdown renderer Coach's Read
  already uses on the dashboard -- not raw `innerHTML` of AI output.
- NO-KEY MODE: greyed out based on the live `ai_available` flag returned by
  `GET /consult` (see above) -- checked on load and safe to re-check after
  a failed send in case the state changed underneath.

## Error handling

| Condition | Behavior |
|---|---|
| Empty/blank/>4000-char message | 400, rejected before touching the AI or disk |
| No API key configured | 200 `{ok:false, error:"set a key to enable coaching"}`, panel greyed out |
| No `dash_data.json` yet | 400 `{ok:false, error:"Run an audit first."}` |
| AI call fails (timeout/HTTP/bad key) | `{ok:false, error:"..."}`, failed turn not persisted, chat state otherwise untouched |
| `/run` lands while a `/consult` call is in flight | reply discarded via run_id check, `{ok:false, error:"Audit was rerun while waiting for a reply -- ask again."}`, no corrupted/stale turn persisted |
| Clear conversation | Always succeeds, wipes to empty history (new run_id) |
| New `/run` completes | Auto-wipes consult history as a side effect (after the pipeline succeeds, not on failure) |

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
8. Send a message, then trigger `/run` again before the reply lands --
   confirm the stale reply is discarded (run_id mismatch path), not
   persisted, and the fresh conversation state after wipe is intact.
9. Try an empty message and a >4000-char paste -- confirm both are
   rejected with 400 before any AI call is made.
10. Verify `call_ai()` still works unchanged for Coach's Read and the
    plain-English strategy translator after being refactored to delegate
    to `call_ai_conversation()`.
