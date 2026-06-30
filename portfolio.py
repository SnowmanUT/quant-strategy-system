"""
portfolio.py — Layer 5: Portfolio construction from solid survivors.

Loads the 43 bootstrap-solid survivors, deduplicates to 1 config per
(family, ticker) pair, then builds three portfolio variants:

  EW    — equal weight, 1/N each slot
  SR    — IS Sharpe-weighted per walk-forward window (no lookahead)
  IVOL  — inverse in-sample volatility weighted per walk-forward window

Gross exposure capped at 1.0 per ticker (no doubling up on same asset).
Walk-forward validated on same 5-window 70/30 split as main tester.
Writes portfolio_results.csv.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from layer1 import download_data, build_configs, TICKERS
from layer2 import backtest_returns, COST_SCHEDULE, CRYPTO_TICKERS

CACHE_DIR   = Path("data_cache")
N_SPLITS    = 5
IS_FRAC     = 0.70
MAX_PER_TICKER = 1.0   # cap gross exposure per underlying asset


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SOLID SURVIVORS + DEDUP
# ══════════════════════════════════════════════════════════════════════════════

def load_survivors(csv: str = "bootstrap_results.csv") -> pd.DataFrame:
    df    = pd.read_csv(csv)
    solid = df[df["flag"] == "solid"].copy()
    # keep best OOS Sharpe config per (family, ticker)
    best  = (solid
             .sort_values("actual_oos_sharpe", ascending=False)
             .drop_duplicates(["family", "ticker"])
             .reset_index(drop=True))
    return best


def find_config(name: str, configs: list) -> tuple | None:
    for cfg in configs:
        if cfg[0] == name:
            return cfg
    return None


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCT POSITION SERIES
# ══════════════════════════════════════════════════════════════════════════════

def build_positions(survivors: pd.DataFrame,
                    universe: dict,
                    configs: list) -> dict[str, pd.Series]:
    """
    Returns {slot_id → position Series} where slot_id = "family|ticker".
    """
    slots: dict[str, pd.Series] = {}

    for _, row in survivors.iterrows():
        tkr  = row["ticker"]
        name = row["name"]
        cfg  = find_config(name, configs)
        if cfg is None or tkr not in universe:
            print(f"  [skip] {name} — config or ticker missing")
            continue
        _, fn, params, _ = cfg
        df = universe[tkr]
        try:
            pos = fn(df, **params)
        except Exception as e:
            print(f"  [error] {name} on {tkr}: {e}")
            continue
        slot_id        = f"{row['family']}|{tkr}"
        slots[slot_id] = pos.rename(slot_id)

    print(f"  {len(slots)} position series built")
    return slots


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET DAILY RETURN SERIES (with transaction costs)
# ══════════════════════════════════════════════════════════════════════════════

def slot_returns(slots: dict, universe: dict,
                 survivors: pd.DataFrame) -> dict[str, pd.Series]:
    """Net daily return for each slot (after transaction costs)."""
    rets: dict[str, pd.Series] = {}
    ticker_map = {f"{r['family']}|{r['ticker']}": r["ticker"]
                  for _, r in survivors.iterrows()}

    for slot_id, pos in slots.items():
        tkr      = ticker_map.get(slot_id, slot_id.split("|")[1])
        cost_bps = COST_SCHEDULE.get("crypto" if tkr in CRYPTO_TICKERS else "default", 1.0)
        df       = universe[tkr]
        daily_r  = df["Close"].pct_change()
        ret      = backtest_returns(pos, daily_r, cost_bps=cost_bps)
        rets[slot_id] = ret

    return rets


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 10 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _max_dd(r: pd.Series) -> float:
    r = r.dropna()
    if not len(r):
        return 0.0
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _annual_ret(r: pd.Series) -> float:
    r = r.dropna()
    if not len(r):
        return 0.0
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def _calmar(r: pd.Series) -> float:
    dd = _max_dd(r)
    return _annual_ret(r) / abs(dd) if dd != 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RETURN BUILDER (one weighting scheme)
# ══════════════════════════════════════════════════════════════════════════════

def _ticker_of(slot_id: str) -> str:
    return slot_id.split("|")[1]


def build_portfolio(rets: dict[str, pd.Series],
                    scheme: str = "EW",
                    n_splits: int = N_SPLITS,
                    is_frac: float = IS_FRAC,
                    max_per_ticker: float = MAX_PER_TICKER) -> pd.Series:
    """
    Combine slot returns into a single portfolio return series.

    scheme: "EW" | "SR" | "IVOL"
      EW   — equal weight (fixed 1/N)
      SR   — IS Sharpe weight recalculated each WF window
      IVOL — 1/IS_vol weight recalculated each WF window

    Per-ticker gross exposure capped at max_per_ticker across all windows.
    """
    # Align all return series to common index
    port_df = pd.DataFrame(rets).sort_index()
    slots   = list(port_df.columns)
    n       = len(port_df)
    ws      = n // n_splits

    port_ret = pd.Series(np.nan, index=port_df.index)

    for i in range(n_splits):
        start = i * ws
        end   = (i + 1) * ws if i < n_splits - 1 else n
        is_end = start + int((end - start) * is_frac)

        is_r  = port_df.iloc[start:is_end]
        oos_r = port_df.iloc[is_end:end]

        if scheme == "EW":
            raw_w = {s: 1.0 for s in slots}
        elif scheme == "SR":
            raw_w = {}
            for s in slots:
                sr = _sharpe(is_r[s])
                raw_w[s] = max(sr, 0.0)   # drop negative-IS slots
        elif scheme == "IVOL":
            raw_w = {}
            for s in slots:
                vol = is_r[s].dropna().std()
                raw_w[s] = 1.0 / vol if vol > 1e-8 else 0.0

        # Apply per-ticker exposure cap
        ticker_exposure: dict[str, float] = {}
        capped_w: dict[str, float] = {}
        for s in slots:
            t = _ticker_of(s)
            ticker_exposure[t] = ticker_exposure.get(t, 0.0) + raw_w[s]

        for s in slots:
            t    = _ticker_of(s)
            te   = ticker_exposure[t]
            cap  = max_per_ticker
            if te > cap and te > 0:
                capped_w[s] = raw_w[s] * cap / te
            else:
                capped_w[s] = raw_w[s]

        # Normalize weights so portfolio sums to 1.0 gross
        total_w = sum(abs(w) for w in capped_w.values())
        if total_w < 1e-10:
            continue
        norm_w = {s: capped_w[s] / total_w for s in slots}

        # OOS portfolio return
        oos_port = sum(norm_w[s] * oos_r[s].fillna(0.0) for s in slots)
        port_ret.iloc[is_end:end] = oos_port.values

    return port_ret.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# STATS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_stats(name: str, r: pd.Series) -> dict:
    return {
        "scheme":      name,
        "oos_sharpe":  round(_sharpe(r),     3),
        "oos_ret":     round(_annual_ret(r),  3),
        "oos_dd":      round(_max_dd(r),      3),
        "calmar":      round(_calmar(r),      3),
        "n_days":      int(r.dropna().shape[0]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def contribution_table(rets: dict[str, pd.Series],
                       port_ret: pd.Series) -> pd.DataFrame:
    """Correlation of each slot's return with portfolio return, plus standalone SR."""
    rows = []
    for slot_id, r in rets.items():
        aligned = r.reindex(port_ret.index).dropna()
        port_al = port_ret.reindex(aligned.index).dropna()
        corr    = aligned.corr(port_al)
        fam, tkr = slot_id.split("|")
        rows.append({
            "family":   fam,
            "ticker":   tkr,
            "slot_sr":  round(_sharpe(r), 3),
            "corr_to_port": round(corr, 3),
        })
    return pd.DataFrame(rows).sort_values("corr_to_port", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(results: list[dict], contrib: pd.DataFrame,
                 survivors: pd.DataFrame) -> None:
    W = 72
    print()
    print("═" * W)
    print("  PORTFOLIO RESULTS — 43 SOLID SURVIVORS → 32 STRATEGY-ASSET SLOTS")
    print("═" * W)

    print(f"\n  {'Scheme':<8} {'OOS Sharpe':>11} {'Ann Ret':>9} {'Max DD':>8} {'Calmar':>8} {'Days':>6}")
    print(f"  {'─'*8} {'─'*11} {'─'*9} {'─'*8} {'─'*8} {'─'*6}")
    for r in results:
        print(f"  {r['scheme']:<8} {r['oos_sharpe']:>+11.3f} "
              f"{r['oos_ret']:>9.1%} {r['oos_dd']:>8.1%} "
              f"{r['calmar']:>8.2f} {r['n_days']:>6,}")

    print(f"\n{'─'*W}")
    print("  SLOT CONTRIBUTION  (EW portfolio, sorted by correlation)")
    print(f"  {'Family':<28} {'Ticker':<9} {'Slot SR':>8} {'Corr→Port':>10}")
    print(f"  {'─'*28} {'─'*9} {'─'*8} {'─'*10}")
    for _, row in contrib.iterrows():
        print(f"  {row['family']:<28} {row['ticker']:<9} "
              f"{row['slot_sr']:>+8.3f} {row['corr_to_port']:>10.3f}")

    print(f"\n{'─'*W}")
    print("  SURVIVOR SUMMARY BY TICKER  (slots in final portfolio)")
    tkr_group = (survivors
                 .sort_values("actual_oos_sharpe", ascending=False)
                 .groupby("ticker", sort=False)
                 .agg(n_slots=("family","count"),
                      mean_sr=("actual_oos_sharpe","mean"))
                 .sort_values("n_slots", ascending=False))
    for tkr, row in tkr_group.iterrows():
        bar = "▓" * int(row["mean_sr"] * 8)
        print(f"  {tkr:<9} {int(row['n_slots'])} slot(s)  mean SR {row['mean_sr']:+.3f}  {bar}")

    print(f"\n{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 62)
    print("PORTFOLIO — 43 solid survivors → optimised combination")
    print("=" * 62)

    print("\n[1/4] Loading data …")
    universe = download_data()
    configs  = build_configs()

    print("[2/4] Loading survivors + building positions …")
    survivors = load_survivors()
    print(f"  {len(survivors)} unique family-ticker pairs")

    slots = build_positions(survivors, universe, configs)
    rets  = slot_returns(slots, universe, survivors)

    print("[3/4] Building portfolios (EW · SR · IVOL) …")
    t0     = time.perf_counter()
    ew_ret = build_portfolio(rets, scheme="EW")
    sr_ret = build_portfolio(rets, scheme="SR")
    iv_ret = build_portfolio(rets, scheme="IVOL")
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    results = [
        portfolio_stats("EW",   ew_ret),
        portfolio_stats("SR",   sr_ret),
        portfolio_stats("IVOL", iv_ret),
    ]

    print("[4/4] Contribution analysis …")
    contrib = contribution_table(rets, ew_ret)

    print_report(results, contrib, survivors)

    # ── per-window breakdown ──────────────────────────────────────────────────
    print("  PER-WINDOW OOS SHARPE  (EW portfolio):")
    n   = len(ew_ret)
    ws  = n // N_SPLITS
    for i in range(N_SPLITS):
        start = i * ws
        end   = (i + 1) * ws if i < N_SPLITS - 1 else n
        w_sr  = _sharpe(ew_ret.iloc[start:end])
        bar   = "▓" * max(0, int(w_sr * 8)) if w_sr > 0 else "░" * max(0, int(-w_sr * 8))
        print(f"    Win {i+1}: {w_sr:>+6.3f}  {bar}")

    # ── write CSV ─────────────────────────────────────────────────────────────
    out = pd.DataFrame(results)
    out.to_csv("portfolio_results.csv", index=False)

    # also write daily portfolio returns for each scheme
    pd.DataFrame({"EW": ew_ret, "SR": sr_ret, "IVOL": iv_ret}).to_csv(
        "portfolio_returns.csv")

    print("\n  portfolio_results.csv   — summary stats")
    print("  portfolio_returns.csv   — daily OOS returns (EW · SR · IVOL)")
