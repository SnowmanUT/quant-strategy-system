"""
Layer 1: bars.py

fetch_bars(symbols, ...): pulls REAL intraday bars from Yahoo Finance and
returns one dataframe with: instrument, datetime, open, high, low, close, volume.

- Hybrid resolution: last ~60 days of 5-minute bars PLUS the last ~7 days again
  at 1-minute (Yahoo caps 1m history to ~7d). Finer (1m) bars win downstream
  wherever both resolutions exist for the same timestamp.
- Regular session only, 09:30-16:00 US/Eastern. Timestamps are converted to
  naive Eastern, deduplicated, and sorted.
- Multi-symbol: accepts a list. When a user uploads trades, use
  symbols_and_range_from_trades() to pull exactly the distinct symbols +
  date range their CSV needs.
- Respects Yahoo's ~59-day window for 5-minute data: if a symbol or date
  range can't be fetched, it is SKIPPED and a plain warning string is
  returned. This function never crashes on a data gap.
- Saves the combined result to bars.csv.
"""

import warnings as _pywarnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["instrument", "datetime", "open", "high", "low", "close", "volume"]

SESSION_START = "09:30"
SESSION_END = "16:00"
EASTERN_TZ = "US/Eastern"

# Yahoo's practical intraday history limits
FIVE_MIN_MAX_DAYS = 59  # Yahoo caps 5m history at ~60 days; stay a day under
ONE_MIN_MAX_DAYS = 7    # Yahoo caps 1m history at ~7 days

_pywarnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")


def _download_interval(symbol, period_days, interval):
    """Download one symbol at one interval. Returns None (never raises) on failure."""
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=period_days)
        df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _normalize_frame(df, symbol):
    """Convert a raw yfinance frame into our schema: Eastern, regular-session-only, sorted."""
    if df is None or df.empty:
        return None

    df = df.copy()

    # yfinance sometimes returns MultiIndex columns (e.g. ('Open','AAPL')) - flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        dt_col: "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    keep = ["datetime", "open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in keep):
        return None
    df = df[keep]

    # Convert to Eastern, then drop tz info to leave naive Eastern timestamps.
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(EASTERN_TZ)
    else:
        df["datetime"] = df["datetime"].dt.tz_convert(EASTERN_TZ)
    df["datetime"] = df["datetime"].dt.tz_localize(None)

    # Regular session only: 09:30-16:00 Eastern
    df = df.set_index("datetime").between_time(SESSION_START, SESSION_END).reset_index()

    df["instrument"] = symbol
    df = df[REQUIRED_COLUMNS]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["instrument", "datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def fetch_bars(symbols, start_date=None, end_date=None, save_path="bars.csv"):
    """
    Pull real intraday bars for one or more symbols from Yahoo Finance.

    Parameters
    ----------
    symbols : str or list[str]
    start_date, end_date : optional bounds. If given, results are trimmed to this
        range after fetching (the fetch windows themselves stay governed by Yahoo's
        ~59d / ~7d limits regardless of what's requested).
    save_path : where to write the combined result (bars.csv by default). Pass
        None to skip saving.

    Returns
    -------
    (dataframe, warnings) : combined dataframe across all symbols/resolutions,
        columns = instrument, datetime, open, high, low, close, volume; and a
        plain list[str] of warnings for anything skipped. Never raises on a
        per-symbol data gap.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})

    warnings_list = []
    all_frames = []

    for sym in symbols:
        raw_5m = _download_interval(sym, FIVE_MIN_MAX_DAYS, "5m")
        five_min = _normalize_frame(raw_5m, sym)
        if five_min is None or five_min.empty:
            warnings_list.append(f"SKIPPED {sym}: could not fetch 5-minute bars (last {FIVE_MIN_MAX_DAYS}d).")
            five_min = pd.DataFrame(columns=REQUIRED_COLUMNS)

        raw_1m = _download_interval(sym, ONE_MIN_MAX_DAYS, "1m")
        one_min = _normalize_frame(raw_1m, sym)
        if one_min is None or one_min.empty:
            warnings_list.append(f"SKIPPED {sym}: could not fetch 1-minute bars (last {ONE_MIN_MAX_DAYS}d).")
            one_min = pd.DataFrame(columns=REQUIRED_COLUMNS)

        if five_min.empty and one_min.empty:
            warnings_list.append(f"SKIPPED {sym}: no data available at any resolution.")
            continue

        # 1m rows go first so drop_duplicates(keep="first") lets finer bars win
        # over 5m bars wherever both cover the same timestamp.
        combined = pd.concat([one_min, five_min], ignore_index=True)
        combined = combined.drop_duplicates(subset=["instrument", "datetime"], keep="first")
        combined = combined.sort_values("datetime").reset_index(drop=True)

        if start_date is not None:
            combined = combined[combined["datetime"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            combined = combined[combined["datetime"] <= pd.to_datetime(end_date)]

        if combined.empty:
            warnings_list.append(f"SKIPPED {sym}: no bars left after trimming to requested date range.")
            continue

        all_frames.append(combined)

    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result = result.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=REQUIRED_COLUMNS)

    if save_path:
        result.to_csv(save_path, index=False)

    return result, warnings_list


def symbols_and_range_from_trades(trades_df):
    """
    Given a trades dataframe (as returned by trades_io.load_trades), return
    (symbols_list, min_datetime, max_datetime) spanning all entry/exit times.
    Lets the caller fetch exactly the bars a user's uploaded trade log needs.
    """
    symbols = sorted(trades_df["Instrument"].dropna().unique().tolist())
    all_times = pd.concat([trades_df["Entry time"], trades_df["Exit time"]]).dropna()
    if all_times.empty:
        return symbols, None, None
    return symbols, all_times.min(), all_times.max()


if __name__ == "__main__":
    df, warns = fetch_bars(["AAPL", "MSFT"])
    print(df.head())
    n_symbols = df["instrument"].nunique() if not df.empty else 0
    print(f"\n{len(df)} bars fetched across {n_symbols} symbol(s).")
    if warns:
        print("\nWarnings:")
        for w in warns:
            print(" -", w)
