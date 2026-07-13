# Trade Auditor

A local, five-layer trade auditing tool: real intraday bars, a rules engine,
a deterministic behavioral/discipline/style audit, a terminal-style
dashboard, and an optional AI coach -- all wired together by one Flask app.

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

That's it. `run.py` checks your dependencies, starts a local server at
`http://127.0.0.1:5050`, and opens the setup wizard in your browser
automatically. Every run also prints its headline numbers to this terminal.

## What you'll see

1. **Step 1 -- Data**: click "Use sample data" (generates a realistic
   momentum trader's log from live Yahoo Finance intraday bars), or upload
   your own NinjaTrader-style trade export CSV.
2. **Step 2 -- Strategy**: pick a one-click preset, describe your style in
   plain English for the AI to draft rules (optional, needs an API key --
   see below), or fill in the rules by hand. Everything's editable either way.
3. **Step 3 -- Run**: kicks off the full pipeline and takes you to the
   dashboard.

The dashboard has everything: verdict banner, KPIs, quant stats, a full
candlestick trade replay with day/timeframe navigation, the trade
constellation, style check, equity curve, leaks, P&L by hour, and the
discipline-cost panel with all nine replay assumptions listed.

## No API key needed

Every deterministic feature -- the whole audit, the dashboard, presets,
manual rules -- works with **zero API keys**. Only two optional pieces need
one: the AI plain-English strategy path (Step 2) and the Coach's Read panel
on the dashboard. Without a key, both are clearly disabled rather than
broken.

To enable them, set an environment variable before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or OPENAI_API_KEY=sk-...
python run.py
```

(Or edit `ai_config.py` directly if you'd rather not use an environment
variable -- see the comments in that file.)

## Files

- `run.py` -- start here
- `app.py` -- the Flask app (routes, pipeline orchestration)
- `setup.html` / `dashboard_template.html` -- the two pages
- `bars.py`, `trades_io.py`, `fetch_and_gen.py` -- data layer
- `rules.py`, `strategy_ai.py`, `ai_client.py`, `ai_config.py` -- strategy rules + AI plumbing
- `metrics.py`, `auditor_engine.py`, `discipline.py`, `context.py` -- the deterministic engine
- `build_dashboard.py` -- computes dash_data.json and renders the dashboard
- `coaching_layer.py` -- the AI coach's distilled findings + report

Everything the app writes (trades, rules, the dashboard) lands in a `data/`
folder created next to these files the first time you run it.
