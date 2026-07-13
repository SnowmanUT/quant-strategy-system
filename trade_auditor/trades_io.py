"""
Layer 1: trades_io.py

load_trades(path): load a NinjaTrader-style trade export (CSV) into a clean
dataframe.

Required columns (must be present or the file is rejected): Instrument,
Market pos., Qty, Entry price, Exit price, Entry time, Exit time, Profit.

Optional columns (backfilled with sane defaults if missing): Trade number,
Account, Strategy, Commission, MAE, MFE, ETD, Bars, Exit name,
Cum. net profit.

Timestamps are parsed with pd.to_datetime so any sane format works. A bad
file raises TradesFileError with one clear line describing the problem.
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "Instrument", "Market pos.", "Qty", "Entry price", "Exit price",
    "Entry time", "Exit time", "Profit",
]

OPTIONAL_COLUMNS = [
    "Trade number", "Account", "Strategy", "Commission",
    "MAE", "MFE", "ETD", "Bars", "Exit name", "Cum. net profit",
]

OPTIONAL_DEFAULTS = {
    "Trade number": None,      # backfilled sequentially if entirely missing
    "Account": "Sim101",
    "Strategy": "Unknown",
    "Commission": 0.0,
    "MAE": None,
    "MFE": None,
    "ETD": None,
    "Bars": None,
    "Exit name": "Unknown",
    "Cum. net profit": None,   # backfilled as cumsum(Profit) if entirely missing
}

NUMERIC_COLUMNS = ["Qty", "Entry price", "Exit price", "Profit", "Commission", "MAE", "MFE", "ETD", "Bars"]


class TradesFileError(ValueError):
    """Raised when an uploaded trades file can't be read as a NinjaTrader-style export."""
    pass


def load_trades(path_or_buffer):
    """
    Load a NinjaTrader-style trade export and return a cleaned dataframe.
    Raises TradesFileError (one clear line) if the file can't be parsed or
    is missing a required column.
    """
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception as e:
        raise TradesFileError(f"Could not read trades file: {e}") from e

    if df.empty:
        raise TradesFileError("Trades file is empty.")

    df.columns = [c.strip() for c in df.columns]

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise TradesFileError(f"Trades file is missing required column(s): {', '.join(missing_required)}.")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = OPTIONAL_DEFAULTS[col]

    for col in ["Entry time", "Exit time"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad_count = int(parsed.isna().sum() - df[col].isna().sum())
        if bad_count > 0:
            raise TradesFileError(f"Could not parse {bad_count} value(s) in '{col}' as timestamps.")
        df[col] = parsed

    if df["Entry time"].isna().any() or df["Exit time"].isna().any():
        raise TradesFileError("Trades file has missing Entry time / Exit time value(s).")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["Qty", "Entry price", "Exit price", "Profit"]].isna().any().any():
        raise TradesFileError("Trades file has non-numeric or missing values in a required numeric column.")

    if df["Trade number"].isna().all():
        df["Trade number"] = range(1, len(df) + 1)

    if df["Cum. net profit"].isna().all():
        df = df.sort_values("Exit time")
        df["Cum. net profit"] = df["Profit"].cumsum()

    if (df["Qty"] <= 0).any():
        raise TradesFileError("Trades file has non-positive Qty value(s); expected positive contract/share counts.")

    valid_pos = df["Market pos."].astype(str).str.strip().str.title().isin(["Long", "Short"])
    if not valid_pos.all():
        bad_vals = sorted(df.loc[~valid_pos, "Market pos."].astype(str).unique().tolist())
        raise TradesFileError(f"Trades file has invalid 'Market pos.' value(s): {bad_vals}. Expected 'Long' or 'Short'.")
    df["Market pos."] = df["Market pos."].astype(str).str.strip().str.title()

    df = df.sort_values("Entry time").reset_index(drop=True)

    ordered_cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c not in REQUIRED_COLUMNS]
    return df[ordered_cols]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "trades.csv"
    try:
        trades = load_trades(path)
        print(trades.head())
        print(f"\nLoaded {len(trades)} trades.")
    except TradesFileError as e:
        print(f"ERROR: {e}")
