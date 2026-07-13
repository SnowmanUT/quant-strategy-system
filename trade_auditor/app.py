"""
Layer 5: app.py

The Flask app that ties every layer together. Zero required API keys --
every deterministic feature (Layers 1-4) runs fully with no key configured;
only the Coach's Read panel needs one (NO-KEY MODE, see /coach below).

Routes:
  GET  /                serves setup.html (the 3-step wizard)
  POST /upload           validate + stage an uploaded NinjaTrader CSV (Step 1)
  GET  /presets           list Layer 2's preset names (Step 2)
  GET  /families           list Layer 2's AI trading families (Step 2)
  POST /translate         AI (purple) plain-English -> strict JSON rules (Step 2)
  GET  /strategies         list the named-strategy library (Step 2)
  GET  /strategies/<name>  load one saved strategy, for one-click reload (Step 2)
  POST /run                DATA -> rules -> engine -> build_dashboard -> dash_data.json (Step 3)
  POST /coach               distilled findings -> AI -> Coach's Read panel
  GET  /dashboard            serves the Layer 4 page with the fresh payload injected
"""

import os
import traceback
import uuid

from flask import Flask, jsonify, request, send_file, Response

import pandas as pd

import ai_client
import bars as bars_mod
import build_dashboard
import coaching_layer
import fetch_and_gen
import import_tradovate
import rules as rules_mod
import strategy_ai
import trades_io

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

TRADES_PATH = os.path.join(DATA_DIR, "trades.csv")
RULES_PATH = os.path.join(DATA_DIR, "rules.json")
STRATEGIES_PATH = os.path.join(DATA_DIR, "strategies.json")
DASH_DATA_PATH = os.path.join(DATA_DIR, "dash_data.json")
DASHBOARD_HTML_PATH = os.path.join(DATA_DIR, "dashboard.html")
DASHBOARD_TEMPLATE_PATH = os.path.join(APP_DIR, "dashboard_template.html")
SETUP_HTML_PATH = os.path.join(APP_DIR, "setup.html")

app = Flask(__name__)


# --------------------------------------------------------------------------
# GET / -- the setup wizard
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(SETUP_HTML_PATH)


# --------------------------------------------------------------------------
# STEP 1: DATA
# --------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    """
    Validate one or more uploaded NinjaTrader-style or Tradovate Performance
    CSVs, with clear errors. Multiple files (e.g. one per prop-firm account)
    are each auto-detected/converted individually, then merged into a single
    trade log for the run.
    """
    files = [f for f in request.files.getlist("file") if f.filename]
    if not files:
        return jsonify({"ok": False, "error": "No file uploaded."}), 400

    frames = []
    source_formats = set()
    for f in files:
        tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{os.path.basename(f.filename)}")
        f.save(tmp_path)
        try:
            raw_df = pd.read_csv(tmp_path)
        except Exception as e:
            os.remove(tmp_path)
            return jsonify({"ok": False, "error": f"Could not read '{f.filename}': {e}"}), 400

        if import_tradovate.is_tradovate_export(raw_df):
            source_formats.add("tradovate")
            account_label = os.path.splitext(os.path.basename(f.filename))[0]
            try:
                frame = import_tradovate.convert_dataframe(raw_df, account_label=account_label)
            except ValueError as e:
                os.remove(tmp_path)
                return jsonify({"ok": False, "error": f"'{f.filename}': {e}"}), 400
        else:
            source_formats.add("ninjatrader")
            frame = raw_df
        os.remove(tmp_path)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    filename = f"{uuid.uuid4().hex}_merged.csv" if len(files) > 1 else f"{uuid.uuid4().hex}_{os.path.basename(files[0].filename)}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    combined.to_csv(save_path, index=False)

    try:
        trades_df = trades_io.load_trades(save_path)
    except trades_io.TradesFileError as e:
        os.remove(save_path)
        return jsonify({"ok": False, "error": str(e)}), 400

    symbols, start, end = bars_mod.symbols_and_range_from_trades(trades_df)
    return jsonify({
        "ok": True,
        "filename": filename,
        "source_format": "+".join(sorted(source_formats)),
        "n_files": len(files),
        "n_trades": len(trades_df),
        "symbols": symbols,
        "date_range": [start.isoformat(), end.isoformat()] if start is not None else None,
    })


# --------------------------------------------------------------------------
# STEP 2: STRATEGY -- presets, AI plain-English path, manual, library
# --------------------------------------------------------------------------

@app.route("/presets")
def presets():
    return jsonify({name: rules_mod.PRESETS[name] for name in rules_mod.PRESETS})


@app.route("/families")
def families():
    return jsonify({"families": strategy_ai.TRADE_FAMILIES})


@app.route("/translate", methods=["POST"])
def translate():
    """AI (purple): family + free text -> strict JSON rules, for the editable review form."""
    payload = request.get_json(force=True, silent=True) or {}
    family = payload.get("family")
    free_text = payload.get("free_text")
    try:
        proposal = strategy_ai.propose_rules_from_text(family, free_text)
    except strategy_ai.StrategyAIError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Unexpected error: {e}"}), 500

    return jsonify({
        "ok": True,
        "proposed_rules": proposal.proposed_rules,
        "validation_error": proposal.validation_error,
        "family": proposal.family,
        "free_text": proposal.free_text,
    })


@app.route("/strategies")
def list_strategies():
    return jsonify({"names": rules_mod.list_strategies(path=STRATEGIES_PATH)})


@app.route("/strategies/<name>")
def get_strategy(name):
    try:
        entry = rules_mod.load_strategy(name, path=STRATEGIES_PATH)
    except rules_mod.RulesValidationError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    return jsonify({"ok": True, "strategy": entry})


# --------------------------------------------------------------------------
# STEP 3: RUN -- the pipeline
# --------------------------------------------------------------------------

@app.route("/run", methods=["POST"])
def run_pipeline():
    """
    Body: {
      "data_source": "sample" | "upload",
      "upload_filename": "..." (if upload),
      "rules": {...}              -- the final rules chosen in Step 2,
      "strategy_name": "..." (optional -- also saves into the library),
      "playbook": "..." (optional),
      "family": "..." (optional),
      "origin": "manual" | "ai" (optional, default "manual"),
    }

    Locks in the rules, gets the data (sample runs fetch_and_gen fresh from
    live Yahoo; upload validates + fetches bars for exactly the symbols/dates
    found -- any that can't be fetched are skipped and listed, the audit
    still covers 100% of the trades since it works from the log alone), then
    runs Engine -> build_dashboard -> dash_data.json.
    """
    payload = request.get_json(force=True, silent=True) or {}
    data_source = payload.get("data_source", "sample")
    rules_payload = payload.get("rules")
    strategy_name = payload.get("strategy_name")
    playbook = payload.get("playbook")
    family = payload.get("family")
    origin = payload.get("origin", "manual")

    if not rules_payload:
        return jsonify({"ok": False, "error": "No rules provided -- finish Step 2 first."}), 400

    try:
        active_rules = rules_mod.save_rules(rules_payload, path=RULES_PATH)
    except rules_mod.RulesValidationError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    if strategy_name:
        try:
            rules_mod.save_strategy(strategy_name, active_rules, playbook=playbook, family=family,
                                     origin=origin, path=STRATEGIES_PATH)
        except rules_mod.RulesValidationError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    warnings = []
    try:
        if data_source == "sample":
            _, bar_warnings = fetch_and_gen.generate_trades(save_path=TRADES_PATH, bars_save_path=None)
            warnings.extend(bar_warnings)
        elif data_source == "upload":
            upload_filename = payload.get("upload_filename")
            if not upload_filename:
                return jsonify({"ok": False, "error": "No uploaded file specified."}), 400
            upload_path = os.path.join(UPLOAD_DIR, upload_filename)
            if not os.path.exists(upload_path):
                return jsonify({"ok": False, "error": "Uploaded file not found on the server -- upload it again."}), 400

            trades_df = trades_io.load_trades(upload_path)  # already validated at /upload, but re-check is cheap
            trades_df.to_csv(TRADES_PATH, index=False)

            # Fetch bars for exactly the symbols/dates in the file. Any that can't be
            # fetched are SKIPPED (bars.fetch_bars never crashes) and listed -- the
            # behavioral audit (auditor_engine) works from the log alone and still
            # covers 100% of the trades regardless.
            symbols, start, end = bars_mod.symbols_and_range_from_trades(trades_df)
            _, bar_warnings = bars_mod.fetch_bars(symbols, start_date=start, end_date=end, save_path=None)
            warnings.extend(bar_warnings)
        else:
            return jsonify({"ok": False, "error": f"Unknown data_source '{data_source}'."}), 400

        dash_data = build_dashboard.build_dashboard(
            trades_path=TRADES_PATH, rules_path=RULES_PATH, strategies_path=STRATEGIES_PATH,
            strategy_name=strategy_name, output_json=DASH_DATA_PATH, output_html=DASHBOARD_HTML_PATH,
            template_path=DASHBOARD_TEMPLATE_PATH,
        )
    except trades_io.TradesFileError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Pipeline failed: {e}", "trace": traceback.format_exc()}), 500

    headline = _print_headline(dash_data, warnings)

    return jsonify({"ok": True, "redirect": "/dashboard", "warnings": warnings, "headline": headline})


def _print_headline(dash_data, warnings):
    v = dash_data["verdict"]
    q = dash_data["quant_stats"]
    d = dash_data["discipline_cost"]["aggregate"]
    exp_r = q["expectancy_r"]["mean"]
    sqn_val = q["sqn"]["value"]

    lines = [
        "",
        "=" * 60,
        "RUN COMPLETE",
        "=" * 60,
        f"Trades: {dash_data['meta']['n_trades']}",
        v["sentence"],
        f"Expectancy: {exp_r:+.2f}R" if exp_r is not None else "Expectancy: n/a",
        f"SQN: {sqn_val:.2f}" if sqn_val is not None else "SQN: n/a",
        d["plain_english"],
    ]
    if warnings:
        lines.append(f"Bar warnings: {len(warnings)} (see dashboard for detail)")
    lines += ["Dashboard: http://127.0.0.1:5050/dashboard", "=" * 60, ""]

    text = "\n".join(lines)
    print(text)
    return text


# --------------------------------------------------------------------------
# COACH -- optional, AI (purple). NO-KEY MODE: everything else above runs
# fully with zero API keys; this is the only endpoint that needs one.
# --------------------------------------------------------------------------

@app.route("/coach", methods=["POST"])
def coach():
    if not os.path.exists(DASH_DATA_PATH):
        return jsonify({"ok": False, "error": "No dashboard data yet -- run the audit first."}), 400

    if not ai_client.ai_available():
        return jsonify({"ok": False, "error": "set a key to enable coaching"}), 200

    import json
    with open(DASH_DATA_PATH, "r", encoding="utf-8") as f:
        dash_data = json.load(f)

    try:
        markdown = coaching_layer.get_coaching(dash_data)
    except coaching_layer.CoachingError as e:
        return jsonify({"ok": False, "error": str(e)}), 200

    return jsonify({"ok": True, "markdown": markdown})


# --------------------------------------------------------------------------
# DASHBOARD
# --------------------------------------------------------------------------

@app.route("/dashboard")
def dashboard():
    if not os.path.exists(DASHBOARD_HTML_PATH):
        return Response("No dashboard yet -- run the audit from the setup wizard first (go to /).", status=404)
    return send_file(DASHBOARD_HTML_PATH)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
