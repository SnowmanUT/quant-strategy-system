"""
layer3.py — Robustness Checks: Parameter Sensitivity + Bootstrap Stress Test

Depends on layer1.py and layer2.py (same directory).

Public API:
    FRAGILE_DD_THRESHOLD      float — worst-case bootstrap DD below this → fragile
    SENSITIVE_STD_THRESH      float — std OOS Sharpe above this → sensitive flag
    SENSITIVE_POS_THRESH      float — pct_positive below this → sensitive flag
    N_BOOTSTRAP               int   — reshuffle count (default 200)

    param_sensitivity(all_df)                    -> pd.DataFrame
    bootstrap_stress(returns, n_boot, seed)      -> dict
    run_robustness(survivors_df, universe, ...)  -> pd.DataFrame
    robustness_report(sens_df, boot_df, all_df) -> None  (prints + writes CSVs)
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from layer2 import (
    get_oos_returns, _sharpe, _max_dd,
    COST_SCHEDULE, _cost_bps,
    walk_forward,
)
from layer1 import download_data, build_configs

# ══════════════════════════════════════════════════════════════════════════════
# TUNEABLE THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

FRAGILE_DD_THRESHOLD  = -0.50   # worst-case bootstrap DD ≤ this → "fragile"
SENSITIVE_STD_THRESH  =  0.30   # family OOS Sharpe std > this → "sensitive"
SENSITIVE_POS_THRESH  =  0.50   # fraction positive configs < this → "sensitive"
N_BOOTSTRAP           = 200     # permutations per survivor


# ══════════════════════════════════════════════════════════════════════════════
# FAST ARRAY-LEVEL METRIC HELPERS  (avoid pd.Series overhead in tight loops)
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe_arr(arr: np.ndarray) -> float:
    if len(arr) < 10:
        return 0.0
    std = arr.std()
    return 0.0 if std == 0 else float(arr.mean() / std * np.sqrt(252))


def _max_dd_arr(arr: np.ndarray) -> float:
    if len(arr) < 1:
        return 0.0
    cum = np.cumprod(1.0 + arr)
    return float(np.min(cum / np.maximum.accumulate(cum) - 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════

def param_sensitivity(
    all_results_df: pd.DataFrame,
    std_thresh: float = SENSITIVE_STD_THRESH,
    pos_thresh: float = SENSITIVE_POS_THRESH,
) -> pd.DataFrame:
    """
    Group all sweep rows by strategy family. For each family compute:
        n_configs        unique parameter configurations (config names)
        n_total          total (config × asset) rows in sweep
        mean_oos_sharpe  mean OOS Sharpe across all configs and assets
        std_oos_sharpe   std of OOS Sharpe  — tight = robust, wide = curve-fit risk
        pct_positive     fraction of rows with OOS Sharpe > 0
        flag             'robust' or 'sensitive'

    Logic: if a family only works on one exact setting and collapses on all others,
    std is high and pct_positive is low — that edge is probably illusory.
    A tight spread with high positive fraction signals a real, parameter-stable edge.
    """
    agg = (
        all_results_df
        .groupby("family")
        .agg(
            n_configs       = ("name",        "nunique"),
            n_total         = ("oos_sharpe",  "count"),
            mean_oos_sharpe = ("oos_sharpe",  "mean"),
            std_oos_sharpe  = ("oos_sharpe",  "std"),
            pct_positive    = ("oos_sharpe",  lambda x: (x > 0).mean()),
        )
        .round(4)
    )

    # Attach category (first value per family — they're all the same)
    agg["category"] = all_results_df.groupby("family")["category"].first()

    # Flag: sensitive if spread too wide OR too few positive configs
    agg["flag"] = np.where(
        (agg["std_oos_sharpe"] > std_thresh) | (agg["pct_positive"] < pos_thresh),
        "sensitive",
        "robust",
    )

    return (
        agg
        .reset_index()
        .sort_values("mean_oos_sharpe", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_stress(
    returns: pd.Series,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
    fragile_threshold: float = FRAGILE_DD_THRESHOLD,
) -> dict:
    """
    Permute (reshuffle) the OOS return series n_boot times.
    Each permutation gives a different equity path from the same pool of daily returns.
    This isolates whether the edge comes from the DISTRIBUTION of returns (real alpha)
    or the exact SEQUENCE they happened to occur in (luck / regime dependency).

    Returns:
        p05_sharpe, p50_sharpe, p95_sharpe  — Sharpe percentiles across reshuffles
        worst_dd                             — worst drawdown seen across all reshuffles
        mean_sharpe                          — mean bootstrap Sharpe
        flag                                 — 'solid' or 'fragile'
        boot_sharpes, boot_dds               — full arrays for charting
    """
    rng  = np.random.default_rng(seed)
    vals = returns.dropna().values
    n    = len(vals)

    if n < 20:
        return {
            "p05_sharpe": 0.0, "p50_sharpe": 0.0, "p95_sharpe": 0.0,
            "mean_sharpe": 0.0, "worst_dd": 0.0,
            "flag": "insufficient_data",
            "boot_sharpes": [], "boot_dds": [],
        }

    boot_sharpes = np.empty(n_boot)
    boot_dds     = np.empty(n_boot)

    for i in range(n_boot):
        shuffled           = rng.permutation(vals)   # permutation, not resample
        boot_sharpes[i]    = _sharpe_arr(shuffled)
        boot_dds[i]        = _max_dd_arr(shuffled)

    worst_dd = float(boot_dds.min())
    flag     = "solid" if worst_dd > fragile_threshold else "fragile"

    return {
        "p05_sharpe":  round(float(np.percentile(boot_sharpes,  5)), 4),
        "p50_sharpe":  round(float(np.percentile(boot_sharpes, 50)), 4),
        "p95_sharpe":  round(float(np.percentile(boot_sharpes, 95)), 4),
        "mean_sharpe": round(float(boot_sharpes.mean()),              4),
        "worst_dd":    round(worst_dd,                                4),
        "flag":        flag,
        "boot_sharpes": boot_sharpes.tolist(),
        "boot_dds":     boot_dds.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS RUNNER  (parameter sensitivity + bootstrap for all top survivors)
# ══════════════════════════════════════════════════════════════════════════════

def run_robustness(
    all_results_df: pd.DataFrame,
    survivors_df: pd.DataFrame,
    universe: dict[str, pd.DataFrame],
    top_n: int = 50,
    n_boot: int = N_BOOTSTRAP,
    cost_schedule: dict | None = None,
    n_splits: int = 5,
    is_frac: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run both robustness checks.

    Returns:
        sensitivity_df  — per-family parameter sensitivity table
        bootstrap_df    — per-survivor bootstrap stress results
    """
    if cost_schedule is None:
        cost_schedule = COST_SCHEDULE

    # ── parameter sensitivity ────────────────────────────────────────────────
    print("[robustness] computing parameter sensitivity …")
    sens_df = param_sensitivity(all_results_df)
    print(f"  {len(sens_df)} families analysed  "
          f"({(sens_df['flag']=='robust').sum()} robust, "
          f"{(sens_df['flag']=='sensitive').sum()} sensitive)")

    # ── bootstrap stress on top N survivors ─────────────────────────────────
    top = (
        survivors_df
        .sort_values("oos_sharpe", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    print(f"\n[robustness] bootstrap {n_boot}× permutations on {len(top)} survivors …")

    boot_records: list[dict] = []

    for i, (_, row) in enumerate(top.iterrows(), 1):
        ticker = row["ticker"]
        fn     = row["fn"]
        params = row["params"]
        cost   = _cost_bps(ticker, cost_schedule)
        df     = universe.get(ticker)

        if df is None:
            print(f"  [warn] {ticker} not in universe — skipping")
            continue

        try:
            oos_ret = get_oos_returns(
                df, fn, params,
                n_splits=n_splits, is_frac=is_frac, cost_bps=cost,
            )
            bs = bootstrap_stress(oos_ret, n_boot=n_boot)

            boot_records.append({
                "name":              row["name"],
                "family":            row["family"],
                "ticker":            ticker,
                "category":          row["category"],
                "actual_oos_sharpe": row["oos_sharpe"],
                "actual_oos_dd":     row["oos_drawdown"],
                "p05_sharpe":        bs["p05_sharpe"],
                "p50_sharpe":        bs["p50_sharpe"],
                "p95_sharpe":        bs["p95_sharpe"],
                "mean_boot_sharpe":  bs["mean_sharpe"],
                "worst_dd":          bs["worst_dd"],
                "flag":              bs["flag"],
                "boot_sharpes":      bs["boot_sharpes"],   # for charting
                "boot_dds":          bs["boot_dds"],
            })
        except Exception as exc:
            print(f"  [warn] {row['name']}/{ticker}: {exc}")

        if i % 10 == 0 or i == len(top):
            print(f"  {i:>3}/{len(top)}")

    boot_df = pd.DataFrame(boot_records)
    return sens_df, boot_df


# ══════════════════════════════════════════════════════════════════════════════
# ROBUSTNESS REPORT  (print + write CSVs)
# ══════════════════════════════════════════════════════════════════════════════

def robustness_report(
    sens_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    sensitivity_csv: str = "param_sensitivity.csv",
    bootstrap_csv: str   = "bootstrap_results.csv",
    bootstrap_pkl: str   = "data_cache/bootstrap_raw.pkl",
) -> None:
    """
    Print the full robustness report and write result CSVs.

    Sections:
        1. Parameter Sensitivity — family-level spread analysis
        2. Bootstrap Stress Test — survivor-level solid / fragile breakdown
    """
    W = 72

    # ── section 1: parameter sensitivity ────────────────────────────────────
    print()
    print("═" * W)
    print("  ROBUSTNESS CHECK 1 — PARAMETER SENSITIVITY")
    print("  (high std = fragile edge; high pct_positive = stable across params)")
    print("═" * W)

    n_robust    = (sens_df["flag"] == "robust").sum()
    n_sensitive = (sens_df["flag"] == "sensitive").sum()
    print(f"\n  {len(sens_df)} families  |  {n_robust} robust  |  {n_sensitive} sensitive\n")

    hdr = (
        f"  {'Flag':<10} {'Family':<26} {'Cat':<11}"
        f" {'N cfg':>5} {'N tot':>6}"
        f" {'Mean SR':>8} {'Std SR':>7} {'Pct+':>6}"
    )
    div = f"  {'─'*10} {'─'*26} {'─'*11} {'─'*5} {'─'*6} {'─'*8} {'─'*7} {'─'*6}"
    print(hdr)
    print(div)

    for cat in sorted(sens_df["category"].unique()):
        sub = sens_df[sens_df["category"] == cat].sort_values("mean_oos_sharpe", ascending=False)
        for _, r in sub.iterrows():
            flag_str = f"[{r['flag'].upper()[:3]}]"
            print(
                f"  {flag_str:<10} {r['family']:<26} {r['category']:<11}"
                f" {int(r['n_configs']):>5} {int(r['n_total']):>6}"
                f" {r['mean_oos_sharpe']:>8.3f} {r['std_oos_sharpe']:>7.3f}"
                f" {r['pct_positive']:>5.0%}"
            )
        print(div)

    # ── section 2: bootstrap stress test ────────────────────────────────────
    print()
    print("═" * W)
    print("  ROBUSTNESS CHECK 2 — BOOTSTRAP STRESS TEST  "
          f"({len(boot_df)} survivors, {N_BOOTSTRAP} permutations each)")
    print("  (solid = worst-case DD > "
          f"{FRAGILE_DD_THRESHOLD:.0%};  fragile = below that threshold)")
    print("═" * W)

    if len(boot_df) == 0:
        print("\n  (no bootstrap results)\n")
    else:
        n_solid   = (boot_df["flag"] == "solid").sum()
        n_fragile = (boot_df["flag"] == "fragile").sum()
        print(f"\n  {n_solid} solid  |  {n_fragile} fragile\n")

        hdr2 = (
            f"  {'Flag':<9} {'Config / Family':<28} {'Ticker':<8}"
            f" {'Actual SR':>9} {'P05 SR':>7} {'P50 SR':>7} {'P95 SR':>7} {'Worst DD':>9}"
        )
        div2 = (
            f"  {'─'*9} {'─'*28} {'─'*8}"
            f" {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9}"
        )
        print(hdr2)
        print(div2)

        for _, r in boot_df.sort_values("actual_oos_sharpe", ascending=False).iterrows():
            flag_str = f"[{r['flag'].upper()[:3]}]"
            display  = r["name"][:27] + "…" if len(r["name"]) > 28 else r["name"]
            print(
                f"  {flag_str:<9} {display:<28} {r['ticker']:<8}"
                f" {r['actual_oos_sharpe']:>9.3f}"
                f" {r['p05_sharpe']:>7.3f}"
                f" {r['p50_sharpe']:>7.3f}"
                f" {r['p95_sharpe']:>7.3f}"
                f" {r['worst_dd']:>8.1%}"
            )

    # ── write CSVs ───────────────────────────────────────────────────────────
    sens_cols = [
        "family", "category", "flag",
        "n_configs", "n_total",
        "mean_oos_sharpe", "std_oos_sharpe", "pct_positive",
    ]
    sens_df[sens_cols].to_csv(sensitivity_csv, index=False)
    print(f"\n  Sensitivity table → {sensitivity_csv}")

    if len(boot_df):
        boot_cols = [
            "name", "family", "ticker", "category", "flag",
            "actual_oos_sharpe", "actual_oos_dd",
            "p05_sharpe", "p50_sharpe", "p95_sharpe",
            "mean_boot_sharpe", "worst_dd",
        ]
        boot_df[boot_cols].to_csv(bootstrap_csv, index=False)
        print(f"  Bootstrap flat    → {bootstrap_csv}")

        Path(bootstrap_pkl).parent.mkdir(exist_ok=True)
        boot_df.to_pickle(bootstrap_pkl)
        print(f"  Bootstrap raw     → {bootstrap_pkl}  (includes full distributions)")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python layer3.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    from layer2 import run_pipeline, funnel_report, FILTERS

    print("=" * 62)
    print("LAYER 3 — Robustness Checks")
    print("=" * 62)

    # ── load data and run sweep (uses cache if available) ───────────────────
    universe = download_data()
    configs  = build_configs()

    print("\n[1/3] Running sweep (or loading cache) …")
    all_results = run_pipeline(universe, configs)

    print("\n[2/3] Applying survival funnel …")
    survivors = funnel_report(all_results)

    if len(survivors) == 0:
        print("\n  No survivors — cannot run robustness checks.")
        raise SystemExit(0)

    # ── robustness checks ────────────────────────────────────────────────────
    print("\n[3/3] Running robustness checks …\n")
    t0 = time.perf_counter()
    sens_df, boot_df = run_robustness(
        all_results_df = all_results,
        survivors_df   = survivors,
        universe       = universe,
        top_n          = min(50, len(survivors)),
        n_boot         = N_BOOTSTRAP,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Robustness wall time: {elapsed:.1f}s")

    robustness_report(sens_df, boot_df)

    print("Layer 3 outputs:")
    print("  param_sensitivity.csv         — family-level parameter stability")
    print("  bootstrap_results.csv         — per-survivor solid/fragile flags")
    print("  data_cache/bootstrap_raw.pkl  — full distributions for charting")
