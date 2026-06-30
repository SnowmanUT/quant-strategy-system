"""
layer2.py — Sweep · Six-Filter Funnel · Funnel Report

Depends on layer1.py (same directory).

Public API:
    COST_SCHEDULE                           dict  — per-asset cost overrides
    FILTERS                                 list  — six configurable filters
    FILTER_THRESHOLDS                       dict  — adjust any threshold here

    backtest_returns(pos, dret, cost_bps)   pd.Series
    compute_metrics(returns, positions)     dict
    walk_forward(df, fn, params, ...)       dict
    run_pipeline(universe, configs, ...)    pd.DataFrame  + writes sweep_results.csv
    apply_funnel(df, filters)               pd.DataFrame  (survivors)
    funnel_report(all_df, survivors_df)     None  (prints report)
    get_oos_returns(df, fn, params, ...)    pd.Series
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from layer1 import download_data, build_configs, TICKERS

# ══════════════════════════════════════════════════════════════════════════════
# COST SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

CRYPTO_TICKERS: frozenset[str] = frozenset({"BTC-USD", "ETH-USD"})

COST_SCHEDULE: dict[str, float] = {
    "default": 1.0,   # 1 bp per side
    "crypto":  5.0,   # 5 bp per side
}


def _cost_bps(ticker: str, schedule: dict = COST_SCHEDULE) -> float:
    return schedule.get(
        "crypto" if ticker in CRYPTO_TICKERS else "default",
        schedule.get("default", 1.0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def backtest_returns(
    positions: pd.Series,
    daily_ret: pd.Series,
    cost_bps: float = 1.0,
) -> pd.Series:
    """
    strat_ret[t] = position[t] × daily_ret[t] − |Δposition[t]| × cost
    positions: {-1, 0, 1}, already lag-shifted (layer1 guarantee).
    """
    cost     = cost_bps / 10_000
    turnover = positions.diff().abs().fillna(0.0)
    return (positions * daily_ret - turnover * cost).rename("strat_ret")


def _sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 10 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _max_dd(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 1:
        return 0.0
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def compute_metrics(
    returns: pd.Series,
    positions: pd.Series | None = None,
) -> dict:
    result = {"sharpe": _sharpe(returns), "max_drawdown": _max_dd(returns)}
    if positions is not None:
        result["trade_count"] = int((positions.diff().abs() > 1e-9).sum())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(
    df: pd.DataFrame,
    fn,
    params: dict,
    n_splits: int = 5,
    is_frac: float = 0.70,
    cost_bps: float = 1.0,
) -> dict:
    """
    5 sequential windows, 70 / 30 IS / OOS each.
    Positions computed once on full history (causal; no per-window warm-up loss).

    Returns dict with:
        oos_returns    pd.Series  — stitched OOS tails (all windows)
        oos_sharpe     float
        oos_drawdown   float
        is_sharpe      float      — mean IS Sharpe across windows
        window_sharpes list[float]— per-window OOS Sharpe
        trade_count    int
        positions      pd.Series
    """
    positions = fn(df, **params)
    daily_ret = df["Close"].pct_change()

    n           = len(df)
    window_size = n // n_splits

    oos_chunks:     list[pd.Series] = []
    is_sharpes:     list[float]     = []
    window_sharpes: list[float]     = []

    for i in range(n_splits):
        start = i * window_size
        end   = (i + 1) * window_size if i < n_splits - 1 else n

        w_pos = positions.iloc[start:end]
        w_ret = daily_ret.iloc[start:end]
        split = int(len(w_pos) * is_frac)

        is_ret  = backtest_returns(w_pos.iloc[:split],  w_ret.iloc[:split],  cost_bps)
        oos_ret = backtest_returns(w_pos.iloc[split:], w_ret.iloc[split:], cost_bps)

        is_sharpes.append(_sharpe(is_ret))
        window_sharpes.append(_sharpe(oos_ret))
        oos_chunks.append(oos_ret)

    oos_combined = pd.concat(oos_chunks).dropna()
    valid_is     = [s for s in is_sharpes if np.isfinite(s)]

    return {
        "oos_returns":     oos_combined,
        "oos_sharpe":      _sharpe(oos_combined),
        "oos_drawdown":    _max_dd(oos_combined),
        "is_sharpe":       float(np.mean(valid_is)) if valid_is else 0.0,
        "window_sharpes":  window_sharpes,
        "trade_count":     int((positions.diff().abs() > 1e-9).sum()),
        "positions":       positions,
    }


def get_oos_returns(
    df: pd.DataFrame,
    fn,
    params: dict,
    n_splits: int = 5,
    is_frac: float = 0.70,
    cost_bps: float = 1.0,
) -> pd.Series:
    return walk_forward(df, fn, params, n_splits, is_frac, cost_bps)["oos_returns"]


# ══════════════════════════════════════════════════════════════════════════════
# SIX-FILTER SURVIVAL FUNNEL
# ══════════════════════════════════════════════════════════════════════════════

FILTER_THRESHOLDS: dict[str, float] = {
    "min_dd":         -0.35,   # OOS MaxDD floor
    "min_oos_sharpe":  0.50,   # minimum OOS Sharpe
    "max_oos_sharpe":  2.50,   # sanity cap (above = asset did the work)
    "oos_is_ratio":    1.30,   # OOS ≤ IS × this  (lucky-OOS / overfit check)
    "min_trades":     30,      # minimum position changes
    # filter 6: is_sharpe > 0  (no threshold param needed)
}

# Each filter: (id, display_label, condition_fn(row) -> bool)
# Row can be dict or pd.Series — both support ["key"] access.
FILTERS: list[tuple] = [
    (
        "max_drawdown",
        f"OOS MaxDD > {FILTER_THRESHOLDS['min_dd']:.0%}",
        lambda r: r["oos_drawdown"] > FILTER_THRESHOLDS["min_dd"],
    ),
    (
        "min_sharpe",
        f"OOS Sharpe > {FILTER_THRESHOLDS['min_oos_sharpe']:.2f}",
        lambda r: r["oos_sharpe"] > FILTER_THRESHOLDS["min_oos_sharpe"],
    ),
    (
        "max_sharpe",
        f"OOS Sharpe < {FILTER_THRESHOLDS['max_oos_sharpe']:.2f}  (sanity cap)",
        lambda r: r["oos_sharpe"] < FILTER_THRESHOLDS["max_oos_sharpe"],
    ),
    (
        "oos_not_above_is",
        f"OOS Sharpe ≤ IS × {FILTER_THRESHOLDS['oos_is_ratio']:.2f}  (no lucky OOS)",
        lambda r: r["oos_sharpe"] <= r["is_sharpe"] * FILTER_THRESHOLDS["oos_is_ratio"],
    ),
    (
        "min_trades",
        f"Trade count ≥ {int(FILTER_THRESHOLDS['min_trades'])}",
        lambda r: r["trade_count"] >= FILTER_THRESHOLDS["min_trades"],
    ),
    (
        "is_positive",
        "IS Sharpe > 0",
        lambda r: r["is_sharpe"] > 0,
    ),
]


def apply_funnel(
    results_df: pd.DataFrame,
    filters: list[tuple] | None = None,
) -> pd.DataFrame:
    """Apply filters sequentially. Returns survivors (passes all six)."""
    if filters is None:
        filters = FILTERS
    surviving = results_df.copy()
    for _, _, cond in filters:
        surviving = surviving[surviving.apply(cond, axis=1)]
    return surviving.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE  (sweep all configs × all assets)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    universe: dict[str, pd.DataFrame],
    configs: list[tuple] | None = None,
    n_splits: int = 5,
    is_frac: float = 0.70,
    cost_schedule: dict | None = None,
    cache: bool = True,
    csv_path: str = "sweep_results.csv",
    pkl_path: str = "data_cache/pipeline_results.pkl",
) -> pd.DataFrame:
    """
    Run walk-forward for every (config, asset) pair.
    Writes sweep_results.csv with all raw metrics.
    Returns DataFrame (no OOS return series stored — use get_oos_returns to regenerate).
    """
    if configs is None:
        configs = build_configs()
    if cost_schedule is None:
        cost_schedule = COST_SCHEDULE

    pkl_file = Path(pkl_path)
    if cache and pkl_file.exists():
        print(f"[pipeline] cache hit → {pkl_file}")
        return pd.read_pickle(pkl_file)

    total   = len(configs) * len(universe)
    records = []
    done    = 0

    print(f"[pipeline] {len(configs)} configs × {len(universe)} assets = {total:,} backtests\n")

    for ticker, df in universe.items():
        cost = _cost_bps(ticker, cost_schedule)
        for name, fn, params, category in configs:
            done += 1
            try:
                wf = walk_forward(df, fn, params, n_splits, is_frac, cost)
                records.append({
                    "name":           name,
                    "family":         name.split("__")[0],
                    "ticker":         ticker,
                    "category":       category,
                    "fn":             fn,
                    "params":         params,
                    "is_sharpe":      round(wf["is_sharpe"],      4),
                    "oos_sharpe":     round(wf["oos_sharpe"],     4),
                    "oos_drawdown":   round(wf["oos_drawdown"],   4),
                    "trade_count":    wf["trade_count"],
                    "window_sharpes": wf["window_sharpes"],
                })
            except Exception as exc:
                print(f"  [warn] {name}/{ticker}: {exc}")

            if done % 500 == 0 or done == total:
                print(f"  {done:>6,}/{total:,}  ({100*done//total:>3}%)")

    results = pd.DataFrame(records)

    # Write CSV (exclude non-serialisable fn/params/window_sharpes columns)
    csv_cols = ["name","family","ticker","category","is_sharpe",
                "oos_sharpe","oos_drawdown","trade_count"]
    results[csv_cols].to_csv(csv_path, index=False)
    print(f"[pipeline] sweep written → {csv_path}  ({len(results):,} rows)")

    if cache:
        pkl_file.parent.mkdir(exist_ok=True)
        results.to_pickle(pkl_path)
        print(f"[pipeline] cache saved → {pkl_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FUNNEL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def funnel_report(
    all_df: pd.DataFrame,
    filters: list[tuple] | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Print the full attrition funnel report. Returns survivors DataFrame.

    Sections:
      1. Attrition — total → positive OOS → OOS > 0.5 → all-six survivors
      2. Survival rate by category
      3. Survival rate by family (with mean OOS Sharpe)
      4. Top survivors table
    """
    if filters is None:
        filters = FILTERS

    total = len(all_df)

    # ── sequential attrition ─────────────────────────────────────────────────
    surviving = all_df.copy()
    steps: list[tuple[str, int]] = [("Total backtests", total)]

    # Intermediate milestones before filters
    pos_oos = (all_df["oos_sharpe"] > 0).sum()
    steps.append(("OOS Sharpe > 0  (any signal at all)", int(pos_oos)))

    cleared_half = (all_df["oos_sharpe"] > 0.5).sum()
    steps.append(("OOS Sharpe > 0.50  (pre-filter view)", int(cleared_half)))

    # Apply filters one at a time, record count after each
    for fid, label, cond in filters:
        surviving = surviving[surviving.apply(cond, axis=1)]
        steps.append((f"+ {label}", len(surviving)))

    survivors = surviving.reset_index(drop=True)

    # ── print attrition ──────────────────────────────────────────────────────
    W = 68
    print()
    print("═" * W)
    print("  SWEEP + SURVIVAL FUNNEL REPORT")
    print("═" * W)
    print(f"\n  {'Stage':<50} {'N':>7}  {'%':>6}")
    print(f"  {'─'*50} {'─'*7}  {'─'*6}")
    for label, n in steps:
        pct = 100 * n / total if total else 0
        indent = "    " if label.startswith("+") else "  "
        print(f"  {indent}{label:<48} {n:>7,}  {pct:>5.1f}%")

    # ── by category ──────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  SURVIVAL BY CATEGORY")
    print(f"{'─'*W}")
    print(f"  {'Category':<14} {'Tested':>8} {'Survived':>9} {'Rate':>7} {'Mean OOS Sharpe':>16}")
    print(f"  {'─'*14} {'─'*8} {'─'*9} {'─'*7} {'─'*16}")

    for cat in sorted(all_df["category"].unique()):
        cat_all  = all_df[all_df["category"] == cat]
        cat_surv = survivors[survivors["category"] == cat] if len(survivors) else pd.DataFrame()
        n_all    = len(cat_all)
        n_surv   = len(cat_surv)
        rate     = 100 * n_surv / n_all if n_all else 0
        mean_s   = cat_surv["oos_sharpe"].mean() if n_surv else float("nan")
        mean_str = f"{mean_s:.3f}" if not np.isnan(mean_s) else "   —  "
        print(f"  {cat:<14} {n_all:>8,} {n_surv:>9,} {rate:>6.1f}% {mean_str:>16}")

    # ── by family ────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  SURVIVAL BY FAMILY  (sorted by survivor count)")
    print(f"{'─'*W}")
    print(f"  {'Family':<26} {'Tested':>8} {'Survived':>9} {'Rate':>7} {'Mean OOS Sharpe':>16}")
    print(f"  {'─'*26} {'─'*8} {'─'*9} {'─'*7} {'─'*16}")

    family_stats = []
    for fam in all_df["family"].unique():
        fam_all  = all_df[all_df["family"] == fam]
        fam_surv = survivors[survivors["family"] == fam] if len(survivors) else pd.DataFrame()
        n_all    = len(fam_all)
        n_surv   = len(fam_surv)
        mean_s   = fam_surv["oos_sharpe"].mean() if n_surv else float("nan")
        family_stats.append((fam, n_all, n_surv, mean_s))

    family_stats.sort(key=lambda x: (-x[2], -x[3] if not np.isnan(x[3]) else 0))
    for fam, n_all, n_surv, mean_s in family_stats:
        rate     = 100 * n_surv / n_all if n_all else 0
        mean_str = f"{mean_s:.3f}" if not np.isnan(mean_s) else "   —  "
        print(f"  {fam:<26} {n_all:>8,} {n_surv:>9,} {rate:>6.1f}% {mean_str:>16}")

    # ── top survivors ────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  TOP {top_n} SURVIVORS  (by OOS Sharpe)")
    print(f"{'─'*W}")

    if len(survivors) == 0:
        print("  (no survivors)")
    else:
        top = survivors.sort_values("oos_sharpe", ascending=False).head(top_n)
        hdr = f"  {'#':>3}  {'Config / Family':<28} {'Ticker':<9} {'OOS SR':>7} {'OOS DD':>7} {'IS SR':>7} {'Trades':>7}"
        print(hdr)
        print(f"  {'─'*3}  {'─'*28} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            # Show family + param snippet (truncate to 28 chars)
            display = row["name"]
            if len(display) > 28:
                display = display[:25] + "…"
            print(
                f"  {rank:>3}  {display:<28} {row['ticker']:<9}"
                f" {row['oos_sharpe']:>7.3f} {row['oos_drawdown']:>6.1%}"
                f" {row['is_sharpe']:>7.3f} {row['trade_count']:>7,}"
            )

    print(f"\n{'═'*W}")
    print(f"  Survivors: {len(survivors):,} / {total:,} "
          f"({100*len(survivors)/total:.2f}%)")
    print(f"{'═'*W}\n")

    return survivors


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python layer2.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 62)
    print("LAYER 2 — Sweep · Six-Filter Funnel · Report")
    print("=" * 62)

    universe = download_data()
    configs  = build_configs()

    print(f"\n  {len(configs)} configs × {len(universe)} assets "
          f"= {len(configs)*len(universe):,} backtests\n")

    t0      = time.perf_counter()
    results = run_pipeline(universe, configs)
    elapsed = time.perf_counter() - t0
    print(f"\n  Wall time: {elapsed/60:.1f} min")

    survivors = funnel_report(results)

    print("Layer 2 complete. Outputs:")
    print("  sweep_results.csv          — all raw metrics")
    print("  data_cache/pipeline_results.pkl  — cached for re-runs")
    print(f"  {len(survivors)} survivors ready for Layer 3 (portfolio)")
