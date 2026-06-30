# Quant Strategy Testing System

5-layer pipeline for systematic strategy research on daily OHLCV data.
10,701 backtests across 90 strategy families and 29 assets (2010–2025).

## Results

| Layer | Output |
|---|---|
| Full sweep | 10,701 backtests → 191 survivors (1.78%) |
| Bootstrap stress | 191 → 43 solid survivors |
| Portfolio (EW) | OOS Sharpe **1.35** · Max DD **-4.8%** · Calmar 1.46 |

Top signals: `vol_scaled_mom` BTC (SR 1.09), `turn_of_month` HYG (SR 0.94, DD -4.3%), `frog_in_pan` SPY (SR 0.90, DD -9.9%)

## Architecture

```
layer1.py     Data + strategy library (90 families, 369 configs)
layer2.py     Backtest engine + walk-forward + six-filter funnel
layer3.py     Parameter sensitivity + bootstrap stress test
layer4.py     Cross-sectional vs time-series momentum comparison
portfolio.py  Portfolio construction from solid survivors
```

## Setup

```bash
pip install yfinance pandas numpy
python layer2.py    # ~8 min first run, cached after
python layer3.py    # ~20s (uses layer2 cache)
python layer4.py    # ~1s
python portfolio.py # <1s
```

## Layer 1 — Data + Strategy Library

**Universe:** 29 assets — sector ETFs (SPY, QQQ, XLK, XLF, XLE…), commodities (GLD, USO), bonds (TLT, HYG), international (EFA, EEM, EWZ), crypto (BTC-USD, ETH-USD), large-cap equities (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, JPM)

**Strategy families (90 total):**

| Category | Families | Notable |
|---|---|---|
| Trend | 38 | dual_momentum, frog_in_pan, vol_scaled_mom, ma_regime_momentum, intermediate_mom |
| Mean Reversion | 24 | rsi_reversion, stochastic, percent_b, keltner_reversion, stoch_rsi |
| Pattern | 7 | turn_of_month, seasonality, three_bar_reversal |
| Volatility | 6 | atr_breakout, low_vol_regime, hv_ratio |
| Volume | 9 | mfi, overnight_gap, vol_mom |
| Composite | 6 | elder_impulse, macd_rsi_combo |

Academic sources: George & Hwang (2004), Novy-Marx (2012), Daniel & Moskowitz (2016), Da/Gurun/Warachka (2014), Blitz/Huij/Martens (2011), Baz et al (2015/AQR), Heston & Sadka (2008), Moskowitz/Ooi/Pedersen (2012), Ang et al (2006), Crabel (1990), Lou/Polk/Skouras (2019), Coppock (1962), Brock/Lakonishok/LeBaron (1992)

**No-lookahead:** Every strategy calls `_lag(signal)` before returning. Walk-forward positions computed once on full history then sliced per window — no warm-up loss.

**Transaction costs:** 1bp/side equities, 5bp/side crypto, applied on `|Δposition|`

## Layer 2 — Six-Filter Survival Funnel

| Filter | Threshold | Purpose |
|---|---|---|
| OOS MaxDD | > -35% | Eliminates blow-ups |
| OOS Sharpe | > 0.50 | Minimum edge |
| OOS Sharpe | < 2.50 | Sanity cap (too good = data artifact) |
| OOS/IS ratio | ≤ 1.30 | Catches overfit |
| Trade count | ≥ 30 | Minimum statistical sample |
| IS Sharpe | > 0 | Signal must exist in training too |

## Layer 3 — Robustness

**Parameter sensitivity:** Family-level OOS Sharpe std and pct_positive across all configs. Flags sensitive families (std > 0.30 or pct_positive < 50%).

**Bootstrap stress:** 200 permutation reshuffles per survivor (not resampling — tests sequence dependence). Flags fragile if worst-case DD < -50%.

## Layer 4 — Cross-Sectional vs Time-Series Momentum

Tests 3m/6m/12-1m lookbacks. **TSM wins Sharpe** (0.543 avg vs 0.212 XSM). XSM wins drawdown despite 2× gross exposure — the ranking filter acts as a signal-quality gate.

## Layer 5 — Portfolio Construction

32 unique strategy-asset slots (deduped from 43 solid survivors by best config per family-ticker pair). Per-ticker gross exposure capped at 1.0. Three weighting schemes (EW/SR/IVOL) all converge to ~1.35 OOS Sharpe — diversification dominates optimization.

**Dead families** (zero survivors across all configs): MA crossover, MACD, supertrend, Parabolic SAR, ADX, Ichimoku, Hull MA, KAMA, OBV, Chaikin oscillators, volume surge, force index, squeeze breakout, BLL filter, coppock curve, multi-lookback TSMOM, Baz multi-EWMA, signed range momentum

## File Outputs

| File | Contents |
|---|---|
| `sweep_results.csv` | All 10,701 backtests (name, family, ticker, IS/OOS Sharpe, DD, trades) |
| `param_sensitivity.csv` | Family-level sensitivity flags |
| `bootstrap_results.csv` | Per-survivor solid/fragile flags + P05/P50/P95 Sharpe |
| `xsm_vs_tsm.csv` | Cross-sectional vs time-series momentum comparison |
| `portfolio_results.csv` | EW/SR/IVOL summary stats |
| `portfolio_returns.csv` | Daily OOS returns for each portfolio scheme |
| `portfolio_equity_curve.png` | Equity curve + drawdown + rolling Sharpe chart |
