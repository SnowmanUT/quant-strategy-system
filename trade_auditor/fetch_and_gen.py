"""
Layer 1: fetch_and_gen.py

Generate a believable WINNING momentum day-trader from the REAL bars pulled
by bars.fetch_bars(), so the demo is verifiable end to end.

Ground rules, enforced by construction:
- every entry price is a real bar's OPEN, every exit price is a real bar's
  CLOSE, both at their real timestamps
- long-biased continuation entries in the direction of the short-term trend
- 2-5 trades per symbol per session, across ~60 sessions -> 300+ trades total
- realistic leaks baked in: occasional oversizing right after a loss, a few
  late-session entries, some 'Manual' exits
- writes trades.csv in the exact NinjaTrader export column format
- the ONLY thing "chosen" is WHICH real bar a trade enters/exits at; nothing
  about the tape itself (prices, timestamps) is invented
"""

import random

import pandas as pd

from bars import fetch_bars

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA"]

TRADES_PER_DAY_RANGE = (2, 5)      # per symbol, per session
TREND_WINDOW_BARS = 6              # ~30 min of 5m bars used to gauge short-term trend
LONG_BIAS = 0.72                   # P(go long) when the short-term trend is up
MIN_HOLD_BARS = 2
MAX_HOLD_BARS = 40
TARGET_WIN_RATE = 0.62
LATE_ENTRY_LEAK_PROB = 0.08        # entries chased after 15:30 ET
MANUAL_EXIT_LEAK_PROB = 0.12
OVERSIZE_AFTER_LOSS_MULT = 2        # revenge-sizing leak
BASE_QTY_CHOICES = [100, 100, 200]  # round-lot share sizes, mostly 100 sh, sometimes 200
COMMISSION_PER_TRADE = 1.00         # flat $ per round turn (typical low-cost equity broker)
LATE_SESSION_CUTOFF = "15:30"

NINJATRADER_COLUMNS = [
    "Trade number", "Instrument", "Account", "Strategy", "Market pos.", "Qty",
    "Entry price", "Exit price", "Entry time", "Exit time", "Profit",
    "Commission", "MAE", "MFE", "ETD", "Bars", "Exit name", "Cum. net profit",
]


def _short_term_trend(day_bars, upto_idx):
    """'up' / 'down' / 'flat' from the trailing TREND_WINDOW_BARS closes leading into upto_idx."""
    start = max(0, upto_idx - TREND_WINDOW_BARS)
    window = day_bars.iloc[start:upto_idx]
    if len(window) < 2:
        return "flat"
    change = window["close"].iloc[-1] - window["close"].iloc[0]
    if change > 0:
        return "up"
    if change < 0:
        return "down"
    return "flat"


def _pick_exit_idx(day_bars, entry_idx, direction, want_win):
    """
    Choose which real bar this trade exits at. Looks ahead a random holding
    window and picks among the *real* bars' closes in that window -- no
    price or timestamp is ever invented, only the choice of which bar.
    """
    n = len(day_bars)
    max_lookahead = min(MAX_HOLD_BARS, n - entry_idx - 1)
    if max_lookahead < MIN_HOLD_BARS:
        return None

    hold = random.randint(MIN_HOLD_BARS, max_lookahead)
    candidates = day_bars.iloc[entry_idx + 1: entry_idx + 1 + hold]
    if candidates.empty:
        return None

    entry_price = day_bars["open"].iloc[entry_idx]
    sign = 1 if direction == "Long" else -1
    pnl_per_share = (candidates["close"] - entry_price) * sign

    favorable = candidates[pnl_per_share > 0]
    unfavorable = candidates[pnl_per_share <= 0]

    if want_win and not favorable.empty:
        weights = pnl_per_share.loc[favorable.index] - pnl_per_share.loc[favorable.index].min() + 0.01
        return random.choices(list(favorable.index), weights=list(weights), k=1)[0]
    if not want_win and not unfavorable.empty:
        return random.choice(list(unfavorable.index))

    # tape didn't cooperate with the desired outcome -- take what's real and available
    return random.choice(list(candidates.index))


def _generate_symbol_trades(symbol, symbol_bars, trade_counter_start):
    trades = []
    trade_num = trade_counter_start
    last_result_loss = False

    symbol_bars = symbol_bars.sort_values("datetime").reset_index(drop=True)
    symbol_bars["session_date"] = symbol_bars["datetime"].dt.date

    for _, day_bars in symbol_bars.groupby("session_date"):
        day_bars = day_bars.reset_index(drop=True)
        n = len(day_bars)
        if n < TREND_WINDOW_BARS + MIN_HOLD_BARS + 2:
            continue  # not enough real bars this session to safely construct trades

        n_trades = random.randint(*TRADES_PER_DAY_RANGE)
        earliest, latest = TREND_WINDOW_BARS, n - MIN_HOLD_BARS - 2
        if latest <= earliest:
            continue

        entry_indices = sorted(random.sample(range(earliest, latest), min(n_trades, latest - earliest)))

        for entry_idx in entry_indices:
            trend = _short_term_trend(day_bars, entry_idx)
            if trend == "up":
                direction = "Long" if random.random() < LONG_BIAS else "Short"
            elif trend == "down":
                direction = "Short" if random.random() < LONG_BIAS else "Long"
            else:
                direction = "Long" if random.random() < 0.5 else "Short"

            # leak: occasionally chase a late-session move
            if random.random() < LATE_ENTRY_LEAK_PROB:
                late_candidates = day_bars[day_bars["datetime"].dt.strftime("%H:%M") >= LATE_SESSION_CUTOFF]
                if not late_candidates.empty and late_candidates.index.max() < n - MIN_HOLD_BARS - 1:
                    entry_idx = int(late_candidates.index.min())

            want_win = random.random() < TARGET_WIN_RATE
            exit_idx = _pick_exit_idx(day_bars, entry_idx, direction, want_win)
            if exit_idx is None:
                continue

            entry_bar = day_bars.iloc[entry_idx]
            exit_bar = day_bars.iloc[exit_idx]

            entry_price = float(entry_bar["open"])
            exit_price = float(exit_bar["close"])
            sign = 1 if direction == "Long" else -1

            # leak: oversize right after a loss (revenge sizing)
            qty = random.choice(BASE_QTY_CHOICES)
            if last_result_loss:
                qty *= OVERSIZE_AFTER_LOSS_MULT

            gross = (exit_price - entry_price) * sign * qty
            commission = COMMISSION_PER_TRADE
            profit = round(gross - commission, 2)
            last_result_loss = profit < 0

            exit_name = "Manual" if random.random() < MANUAL_EXIT_LEAK_PROB else ("Target" if profit >= 0 else "Stop")

            trades.append({
                "Trade number": trade_num,
                "Instrument": symbol,
                "Account": "Sim101",
                "Strategy": "MomentumContinuation",
                "Market pos.": direction,
                "Qty": qty,
                "Entry price": round(entry_price, 4),
                "Exit price": round(exit_price, 4),
                "Entry time": entry_bar["datetime"],
                "Exit time": exit_bar["datetime"],
                "Profit": profit,
                "Commission": round(commission, 2),
                "MAE": round(min(0.0, gross), 2),
                "MFE": round(max(0.0, gross), 2),
                "ETD": round(gross - profit, 2),
                "Bars": int(exit_idx - entry_idx),
                "Exit name": exit_name,
            })
            trade_num += 1

    return trades, trade_num


def generate_trades(symbols=None, save_path="trades.csv", bars_save_path="bars.csv"):
    """
    Fetch real bars for `symbols`, then synthesize a winning momentum
    day-trader's log purely from those real bars. Writes trades.csv in the
    exact NinjaTrader export column format.

    Returns (trades_df, bar_warnings).
    """
    symbols = symbols or DEFAULT_SYMBOLS
    bars_df, bar_warnings = fetch_bars(symbols, save_path=bars_save_path)

    if bars_df.empty:
        raise RuntimeError("No bars were available for any requested symbol; cannot generate trades.")

    all_trades = []
    trade_counter = 1
    for symbol in sorted(bars_df["instrument"].unique()):
        symbol_bars = bars_df[bars_df["instrument"] == symbol]
        trades, trade_counter = _generate_symbol_trades(symbol, symbol_bars, trade_counter)
        all_trades.extend(trades)

    if not all_trades:
        raise RuntimeError("Could not construct any trades from the fetched bars (sessions too short).")

    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values("Exit time").reset_index(drop=True)
    trades_df["Cum. net profit"] = trades_df["Profit"].cumsum().round(2)
    trades_df = trades_df[NINJATRADER_COLUMNS]

    if save_path:
        trades_df.to_csv(save_path, index=False)

    return trades_df, bar_warnings


if __name__ == "__main__":
    trades_df, warns = generate_trades()
    win_rate = (trades_df["Profit"] > 0).mean()
    print(f"Generated {len(trades_df)} trades across {trades_df['Instrument'].nunique()} symbol(s).")
    print(f"Win rate: {win_rate:.1%}  |  Total net profit: ${trades_df['Profit'].sum():,.2f}")
    if warns:
        print("\nBar-fetch warnings:")
        for w in warns:
            print(" -", w)
