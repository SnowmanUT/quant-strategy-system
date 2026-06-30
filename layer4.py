"""
layer4.py — Cross-Sectional Momentum vs Time-Series Momentum
Standalone module — only depends on layer1.py for download_data().

Cross-sectional momentum (XSM):
    Every 21 trading days, rank all assets by trailing return.
    Long top 1/3, short bottom 1/3, equal weight, hold to next rebalance.

Time-series momentum (TSM):
    Each asset trades its own sign(trailing return), equal-weighted portfolio.

Lookbacks tested: 3 months (63), 6 months (126), 12-1 months (252, skip 21).
Both validated with the same 5-fold walk-forward used in the main sweep.
Results written to xsm_vs_tsm.csv.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from layer1 import download_data, MIN_BARS, TICKERS

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

CRYPTO_TICKERS: frozenset[str] = frozenset({"BTC-USD", "ETH-USD"})

COST_BPS: dict[str, float] = {
    "default": 1.0,   # bp per side
    "crypto":  5.0,
}

# Lookback specs: (bars_lookback, bars_skip, display_label)
LOOKBACKS: list[tuple[int, int, str]] = [
    (63,  0,  "3-month"),
    (126, 0,  "6-month"),
    (252, 21, "12-1-month"),   # skip 21 days avoids short-term reversal
]

REBAL_FREQ:  int   = 21     # rebalance every ~month
TOP_FRAC:    float = 1/3    # long/short each 1/3 of ranked universe
N_SPLITS:    int   = 5
IS_FRAC:     float = 0.70


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def build_price_matrix(universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Align all assets to a common date index.
    Returns DataFrame (dates × tickers) of Close prices.
    Columns with fewer than MIN_BARS valid closes are dropped.
    """
    closes = pd.DataFrame(
        {tkr: df["Close"] for tkr, df in universe.items()}
    )
    # Drop assets that fall below minimum history after alignment
    closes = closes.loc[:, closes.count() >= MIN_BARS]
    # Don't drop rows — assets may have NaN for dates before they listed.
    # Strategy functions ignore NaN assets at each rebalance.
    return closes


def _ticker_cost(ticker: str) -> float:
    """One-way cost in decimal (not bp)."""
    return COST_BPS.get(
        "crypto" if ticker in CRYPTO_TICKERS else "default", 1.0
    ) / 10_000


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RETURN SERIES
# ══════════════════════════════════════════════════════════════════════════════

def xsm_returns(
    closes: pd.DataFrame,
    lookback: int,
    skip: int = 0,
    rebal: int = REBAL_FREQ,
    top_frac: float = TOP_FRAC,
) -> pd.Series:
    """
    Cross-sectional momentum portfolio.

    At each rebalance date:
        signal_i = closes[t - skip] / closes[t - skip - lookback] - 1
        rank assets by signal; long top top_frac, short bottom top_frac.
        weight = ±1/n_q  (each bucket equal-weight, sums to ±1.0)
    Gross exposure ≈ 2.0  (100% long + 100% short).
    """
    daily_ret = closes.pct_change()
    tickers   = closes.columns.tolist()

    # Position matrix: NaN until first signal, then forward-filled
    pos = pd.DataFrame(np.nan, index=closes.index, columns=tickers)

    first_i = lookback + skip + rebal
    for i in range(first_i, len(closes), rebal):
        end_i   = i - skip         # most recent bar used in signal
        start_i = end_i - lookback  # lookback-start bar
        if start_i < 0:
            continue

        sig   = closes.iloc[end_i] / closes.iloc[start_i] - 1.0
        valid = sig.dropna()
        n     = len(valid)
        if n < 3:
            continue

        n_q    = max(1, int(n * top_frac))
        ranked = valid.rank()

        new_pos                                       = pd.Series(0.0, index=tickers)
        new_pos.loc[ranked[ranked > n - n_q].index]  =  1.0 / n_q   # long top n_q
        new_pos.loc[ranked[ranked <= n_q].index]     = -1.0 / n_q   # short bottom n_q
        pos.iloc[i]                                   = new_pos.values

    pos = pos.ffill().fillna(0.0)

    # Portfolio return
    port = (pos * daily_ret).sum(axis=1)

    # Transaction costs: applied at every position change (rebalance dates)
    delta = pos.diff().abs()
    for tkr in tickers:
        port -= delta[tkr] * _ticker_cost(tkr)

    return port.rename("xsm")


def tsm_returns(
    closes: pd.DataFrame,
    lookback: int,
    skip: int = 0,
) -> pd.Series:
    """
    Time-series momentum portfolio: equal-weighted across all assets.

    position_i = sign(closes[t-skip] / closes[t-skip-lookback] - 1)
    weight_i   = ±1/n_valid  (n_valid = # assets with a signal on that day)
    Gross exposure ≈ 1.0.

    Signal is lagged 1 day (position enters next bar) — no lookahead.
    """
    daily_ret = closes.pct_change()
    tickers   = closes.columns.tolist()

    # Momentum signal, skip most recent 'skip' bars
    sig      = closes.shift(skip) / closes.shift(lookback + skip) - 1.0
    pos_raw  = np.sign(sig).shift(1)   # lag 1 → no lookahead

    # Equal-weight by valid-asset count on each day
    n_valid  = pos_raw.notna().sum(axis=1).replace(0, np.nan)
    port     = (pos_raw.fillna(0.0) * daily_ret).sum(axis=1) / n_valid

    # Transaction costs: per-asset turnover
    delta = pos_raw.fillna(0.0).diff().abs()
    for tkr in tickers:
        port -= (delta[tkr] / n_valid) * _ticker_cost(tkr)

    return port.rename("tsm")


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION  (mirrors layer2 logic, applied to portfolio series)
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 10 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _max_dd(ret: pd.Series) -> float:
    r = ret.dropna()
    if not len(r):
        return 0.0
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def wf_portfolio(
    port_ret: pd.Series,
    n_splits: int = N_SPLITS,
    is_frac: float = IS_FRAC,
) -> dict:
    """
    Walk-forward validation on a portfolio return series.
    Mirrors the single-asset walk_forward in layer2:
        5 sequential windows, 70/30 IS/OOS split each.
        Stitch 5 OOS tails → final OOS Sharpe + drawdown.
    """
    r = port_ret.dropna()
    n = len(r)
    ws = n // n_splits

    oos_chunks:     list[pd.Series] = []
    is_sharpes:     list[float]     = []
    window_sharpes: list[float]     = []

    for i in range(n_splits):
        start = i * ws
        end   = (i + 1) * ws if i < n_splits - 1 else n
        w     = r.iloc[start:end]
        split = int(len(w) * is_frac)

        is_sharpes.append(_sharpe(w.iloc[:split]))
        oos_ret = w.iloc[split:]
        window_sharpes.append(_sharpe(oos_ret))
        oos_chunks.append(oos_ret)

    oos = pd.concat(oos_chunks)
    return {
        "oos_sharpe":     _sharpe(oos),
        "oos_drawdown":   _max_dd(oos),
        "is_sharpe":      float(np.mean([s for s in is_sharpes if np.isfinite(s)])),
        "window_sharpes": window_sharpes,
        "oos_returns":    oos,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REGIME ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

# Named calendar regimes across the test window 2010-2025.
# Each is a (start, end, label) — semi-inclusive [start, end).
REGIMES: list[tuple[str, str, str]] = [
    ("2010-01-01", "2012-01-01", "post-GFC recovery"),
    ("2012-01-01", "2016-01-01", "low-vol QE bull"),
    ("2016-01-01", "2018-07-01", "global sync growth"),
    ("2018-07-01", "2020-03-01", "late-cycle chop"),
    ("2020-03-01", "2020-12-01", "COVID crash + rebound"),
    ("2020-12-01", "2022-01-01", "reflation / crypto boom"),
    ("2022-01-01", "2023-01-01", "rate-hike bear"),
    ("2023-01-01", "2025-01-01", "AI bull / soft landing"),
]


def regime_sharpes(oos_ret: pd.Series) -> list[tuple[str, float, str]]:
    """Return (label, sharpe, dates) for each regime, only where OOS data exists."""
    rows = []
    for start, end, label in REGIMES:
        slice_ = oos_ret.loc[start:end].dropna()
        if len(slice_) < 21:
            continue
        rows.append((label, _sharpe(slice_), f"{slice_.index[0].date()}–{slice_.index[-1].date()}"))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(universe: dict[str, pd.DataFrame]) -> dict:
    """
    Run XSM and TSM for all three lookbacks.
    Returns nested dict of results keyed by strategy × lookback.
    """
    closes  = build_price_matrix(universe)
    n_dates = len(closes)
    n_assets = len(closes.columns)
    print(f"\n  Price matrix: {n_dates:,} days × {n_assets} assets"
          f"  ({closes.index[0].date()} → {closes.index[-1].date()})")

    results: dict[str, dict] = {}

    for lookback, skip, label in LOOKBACKS:
        print(f"\n  [{label}]  lookback={lookback}d  skip={skip}d")

        # ── XSM ─────────────────────────────────────────────────────────────
        xsm_ret = xsm_returns(closes, lookback=lookback, skip=skip)
        xsm_wf  = wf_portfolio(xsm_ret)
        xsm_reg = regime_sharpes(xsm_wf["oos_returns"])
        print(f"    XSM  OOS Sharpe={xsm_wf['oos_sharpe']:+.3f}  "
              f"OOS MaxDD={xsm_wf['oos_drawdown']:.1%}  "
              f"IS Sharpe={xsm_wf['is_sharpe']:+.3f}")

        # ── TSM ─────────────────────────────────────────────────────────────
        tsm_ret = tsm_returns(closes, lookback=lookback, skip=skip)
        tsm_wf  = wf_portfolio(tsm_ret)
        tsm_reg = regime_sharpes(tsm_wf["oos_returns"])
        print(f"    TSM  OOS Sharpe={tsm_wf['oos_sharpe']:+.3f}  "
              f"OOS MaxDD={tsm_wf['oos_drawdown']:.1%}  "
              f"IS Sharpe={tsm_wf['is_sharpe']:+.3f}")

        results[label] = {
            "xsm": {**xsm_wf, "regime_sharpes": xsm_reg, "full_ret": xsm_ret},
            "tsm": {**tsm_wf, "regime_sharpes": tsm_reg, "full_ret": tsm_ret},
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# REPORT + CSV
# ══════════════════════════════════════════════════════════════════════════════

def print_report(results: dict, csv_path: str = "xsm_vs_tsm.csv") -> None:
    W = 76
    print()
    print("═" * W)
    print("  CROSS-SECTIONAL vs TIME-SERIES MOMENTUM  —  COMPARISON")
    print("  (Results reported as they came out, no tuning)")
    print("═" * W)

    # ── gross exposure note ──────────────────────────────────────────────────
    print("""
  Gross exposure:
    XSM  ≈ 2.0  (100% long top third + 100% short bottom third)
    TSM  ≈ 1.0  (equal-weight ±1/n per asset, roughly half each side)
  A higher Sharpe on XSM partly reflects higher leverage. Read the DD too.
""")

    # ── main comparison table ────────────────────────────────────────────────
    print(f"  {'Lookback':<14} {'Strategy':<6}"
          f" {'OOS Sharpe':>11} {'OOS MaxDD':>10} {'IS Sharpe':>10}"
          f" {'Win ≥ 3/5':>10}")
    print(f"  {'─'*14} {'─'*6} {'─'*11} {'─'*10} {'─'*10} {'─'*10}")

    csv_rows: list[dict] = []
    verdict: list[str] = []

    for lookback, skip, label in LOOKBACKS:
        for strat in ("xsm", "tsm"):
            wf  = results[label][strat]
            pos_windows = sum(1 for s in wf["window_sharpes"] if s > 0)
            print(
                f"  {label:<14} {strat.upper():<6}"
                f" {wf['oos_sharpe']:>+11.3f}"
                f" {wf['oos_drawdown']:>10.1%}"
                f" {wf['is_sharpe']:>+10.3f}"
                f"   {pos_windows}/5"
            )
            csv_rows.append({
                "lookback":     label,
                "strategy":     strat.upper(),
                "oos_sharpe":   round(wf["oos_sharpe"],   4),
                "oos_drawdown": round(wf["oos_drawdown"],  4),
                "is_sharpe":    round(wf["is_sharpe"],    4),
                "windows_positive": pos_windows,
                "window_sharpes": ";".join(f"{s:.3f}" for s in wf["window_sharpes"]),
            })
        print()

    # ── per-lookback verdict ────────────────────────────────────────────────
    print(f"  {'─'*W}")
    print(f"  VERDICT BY LOOKBACK:")
    for lookback, skip, label in LOOKBACKS:
        xsm_s = results[label]["xsm"]["oos_sharpe"]
        tsm_s = results[label]["tsm"]["oos_sharpe"]
        xsm_dd = results[label]["xsm"]["oos_drawdown"]
        tsm_dd = results[label]["tsm"]["oos_drawdown"]
        if xsm_s > tsm_s:
            call = (f"XSM wins on Sharpe (+{xsm_s - tsm_s:.3f}), "
                    f"drawdown XSM {xsm_dd:.1%} vs TSM {tsm_dd:.1%}")
        else:
            call = (f"TSM wins on Sharpe (+{tsm_s - xsm_s:.3f}), "
                    f"drawdown XSM {xsm_dd:.1%} vs TSM {tsm_dd:.1%}")
        print(f"  {label:<14}: {call}")
    print()

    # ── walk-forward window breakdown ───────────────────────────────────────
    print(f"  {'─'*W}")
    print(f"  PER-WINDOW OOS SHARPE (shows regime dependency):")
    print(f"  Each of the 5 windows covers ~{100//N_SPLITS}% of history × "
          f"{int(100*(1-IS_FRAC))}% OOS slice.\n")

    hdw = f"  {'Lookback':<14} {'Strat':<6}  " + "  ".join(f"{'Win'+str(i+1):>7}" for i in range(N_SPLITS))
    print(hdw)
    print(f"  {'─'*14} {'─'*6}  " + "  ".join(["─"*7]*N_SPLITS))
    for lookback, skip, label in LOOKBACKS:
        for strat in ("xsm", "tsm"):
            ws = results[label][strat]["window_sharpes"]
            ws_str = "  ".join(f"{s:>+7.3f}" for s in ws)
            print(f"  {label:<14} {strat.upper():<6}  {ws_str}")
        print()

    # ── regime analysis ────────────────────────────────────────────────────
    print(f"  {'─'*W}")
    print(f"  REGIME SHARPES (12-1-month lookback, OOS bars only):")
    label_12 = "12-1-month"
    for strat in ("xsm", "tsm"):
        regs = results[label_12][strat]["regime_sharpes"]
        print(f"\n  {strat.upper()}:")
        for reg_label, sr, dates in regs:
            bar = "▓" * max(0, int(sr * 10)) if sr > 0 else "░" * max(0, int(-sr * 10))
            print(f"    {reg_label:<28} {sr:>+6.2f}  {bar}  ({dates})")

    # ── bottom-line plain statement ─────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  BOTTOM LINE:")
    best_lb = max(LOOKBACKS, key=lambda x: results[x[2]]["xsm"]["oos_sharpe"])
    xsm_best = results[best_lb[2]]["xsm"]["oos_sharpe"]
    tsm_best_same = results[best_lb[2]]["tsm"]["oos_sharpe"]
    overall_xsm = np.mean([results[lb[2]]["xsm"]["oos_sharpe"] for lb in LOOKBACKS])
    overall_tsm = np.mean([results[lb[2]]["tsm"]["oos_sharpe"] for lb in LOOKBACKS])

    print(f"""
  Average OOS Sharpe across all three lookbacks:
    XSM  {overall_xsm:+.3f}
    TSM  {overall_tsm:+.3f}

  {'XSM outperformed TSM on average across lookbacks.' if overall_xsm > overall_tsm else 'TSM outperformed or matched XSM on average across lookbacks.'}
  Best XSM lookback: {best_lb[2]}  (OOS Sharpe {xsm_best:+.3f} vs TSM {tsm_best_same:+.3f} on same lookback).
  Note: XSM runs at ~2× gross exposure vs TSM. Drawdowns on XSM
  tend to be deeper in proportion. Both strategies can have severe
  drawdowns in trending single-factor bear markets (2022).
""")
    print("═" * W)

    # ── write CSV ────────────────────────────────────────────────────────────
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  Results written → {csv_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python layer4.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 62)
    print("LAYER 4 — Cross-Sectional vs Time-Series Momentum")
    print("=" * 62)

    print("\nLoading market data …")
    universe = download_data()

    t0      = time.perf_counter()
    results = run_comparison(universe)
    elapsed = time.perf_counter() - t0

    print(f"\n  Computed in {elapsed:.1f}s")

    print_report(results)
