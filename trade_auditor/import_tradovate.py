"""
import_tradovate.py

Converts one or more Tradovate "Performance" CSV exports (symbol,
_priceFormat, _priceFormatType, _tickSize, buyFillId, sellFillId, qty,
buyPrice, sellPrice, pnl, boughtTimestamp, soldTimestamp, duration) into
the NinjaTrader-style schema trades_io.load_trades() expects.

Tradovate's export already pairs each round trip's buy and sell fill, so
no fill-matching is needed -- just direction and column renaming:
  - Long:  bought before sold  -> Entry = buy, Exit = sell
  - Short: sold before bought  -> Entry = sell, Exit = buy
  - pnl uses accounting format ("$81.00", "$(58.00)" for negative).

Usage:
    python import_tradovate.py "Performance-acct3.csv" "Performance-acct4.csv" ... -o data/trades.csv

Each input file's stem becomes that block of trades' Account value (e.g.
"Performance-acct3.csv" -> "acct3"), so multi-account exports stay
distinguishable in the dashboard.
"""

import argparse
import re
import sys

import pandas as pd

PNL_RE = re.compile(r"^\$\(?([\d,]+\.?\d*)\)?$")

# Standard futures contract code: ROOT + month letter (F G H J K M N Q U V X Z)
# + 1-2 digit year, e.g. "MNQU6" = MNQ, September, 2026. Yahoo Finance only
# knows the continuous-contract ticker ("MNQ=F"), not the dated contract code,
# so bars.py's yfinance lookup needs the root with "=F" appended.
CONTRACT_CODE_RE = re.compile(r"^([A-Z0-9]{1,3})([FGHJKMNQUVXZ])(\d{1,2})$")


def _yahoo_symbol(symbol):
    m = CONTRACT_CODE_RE.match(str(symbol).strip())
    if not m:
        return symbol  # not a recognized dated-contract code -- pass through unchanged
    return f"{m.group(1)}=F"


def _parse_pnl(raw):
    raw = str(raw).strip()
    m = PNL_RE.match(raw)
    if not m:
        raise ValueError(f"Could not parse pnl value: {raw!r}")
    value = float(m.group(1).replace(",", ""))
    return -value if raw.startswith("$(") else value


def _account_label(path):
    stem = path.rsplit("/", 1)[-1]
    stem = re.sub(r"\.csv$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^Performance-?", "", stem, flags=re.IGNORECASE)
    return stem or stem


TRADOVATE_COLUMNS = ["symbol", "qty", "buyPrice", "sellPrice", "pnl", "boughtTimestamp", "soldTimestamp"]


def is_tradovate_export(df):
    """True if df's columns match a Tradovate Performance CSV export."""
    return all(c in df.columns for c in TRADOVATE_COLUMNS)


def convert_dataframe(df, account_label="Tradovate"):
    """Convert an already-loaded Tradovate Performance dataframe to trade_auditor's schema."""
    missing = [c for c in TRADOVATE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected Tradovate column(s): {', '.join(missing)}")

    bought = pd.to_datetime(df["boughtTimestamp"])
    sold = pd.to_datetime(df["soldTimestamp"])
    is_long = bought < sold

    out = pd.DataFrame({
        "Instrument": df["symbol"].map(_yahoo_symbol),
        "Market pos.": is_long.map({True: "Long", False: "Short"}),
        "Qty": df["qty"],
        "Entry price": df["buyPrice"].where(is_long, df["sellPrice"]),
        "Exit price": df["sellPrice"].where(is_long, df["buyPrice"]),
        "Entry time": bought.where(is_long, sold),
        "Exit time": sold.where(is_long, bought),
        "Profit": df["pnl"].map(_parse_pnl),
        "Account": account_label,
    })
    return out.sort_values("Entry time").reset_index(drop=True)


def convert_file(path):
    df = pd.read_csv(path)
    try:
        return convert_dataframe(df, account_label=_account_label(path))
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Convert Tradovate Performance CSV(s) to trade_auditor's schema.")
    parser.add_argument("inputs", nargs="+", help="Tradovate Performance-*.csv file(s)")
    parser.add_argument("-o", "--output", default="data/trades.csv", help="output CSV path (default: data/trades.csv)")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        try:
            frames.append(convert_file(path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Entry time").reset_index(drop=True)
    combined["Trade number"] = range(1, len(combined) + 1)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined)} trades from {len(args.inputs)} file(s) -> {args.output}")


if __name__ == "__main__":
    main()
