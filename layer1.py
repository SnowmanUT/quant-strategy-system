"""
layer1.py — Layer 1: Data + Strategy Library

Public API:
    download_data(**kwargs)       -> dict[str, pd.DataFrame]
    build_configs()               -> list[(name, fn, params, category)]
    get_positions(df, name, fn, params) -> pd.Series in {-1, 0, 1}
    run_all_strategies(df)        -> pd.DataFrame (default params per family)
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE & SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

TICKERS: list[str] = [
    "SPY","QQQ","IWM","DIA",
    "XLK","XLF","XLE","XLV","XLI","XLU","XLY","XLP",
    "GLD","USO","TLT","HYG","EFA","EEM","EWZ",
    "BTC-USD","ETH-USD",
    "AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","JPM",
]
START     = "2010-01-01"
END       = "2025-01-01"
MIN_BARS  = 500
CACHE_DIR = Path("data_cache")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_data(
    tickers: list[str] = TICKERS,
    start: str = START,
    end: str = END,
    min_bars: int = MIN_BARS,
    cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download adjusted daily OHLCV. Returns {ticker: DataFrame[O,H,L,C,V]}."""
    CACHE_DIR.mkdir(exist_ok=True)
    slug = f"{start[:7]}_{end[:7]}".replace("-","")
    cache_path = CACHE_DIR / f"ohlcv_{slug}.pkl"

    if cache and cache_path.exists():
        print(f"[data] cache hit → {cache_path}")
        return pd.read_pickle(cache_path)

    universe: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    for tkr in tickers:
        try:
            raw  = yf.download(tkr, start=start, end=end,
                               auto_adjust=True, progress=False, threads=False)
            raw  = _flatten(raw)
            keep = [c for c in ("Open","High","Low","Close","Volume") if c in raw.columns]
            df   = raw[keep].dropna(subset=["Close"])
            if len(df) < min_bars:
                print(f"  SKIP {tkr:<12} {len(df)} bars < {min_bars}")
                skipped.append(tkr); continue
            universe[tkr] = df.astype({c: float for c in keep})
            print(f"  OK   {tkr:<12} {len(df):>5,} bars  "
                  f"{df.index[0].date()} → {df.index[-1].date()}")
        except Exception as exc:
            print(f"  ERR  {tkr:<12} {exc}")
            skipped.append(tkr)

    print(f"\n[data] {len(universe)} loaded"
          + (f", skipped {skipped}" if skipped else ""))
    if cache:
        pd.to_pickle(universe, cache_path)
        print(f"[data] cached → {cache_path}")
    return universe


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: (x * w).sum() / w.sum(), raw=True)

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hi, lo, cp = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([hi-lo, (hi-cp).abs(), (lo-cp).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _bb(s: pd.Series, n: int = 20, k: float = 2.0):
    m = s.rolling(n).mean(); σ = s.rolling(n).std()
    return m - k*σ, m, m + k*σ

def _macd(s: pd.Series, fast=12, slow=26, sig=9):
    line = _ema(s, fast) - _ema(s, slow)
    return line, _ema(line, sig), line - _ema(line, sig)

def _tp(df: pd.DataFrame) -> pd.Series:
    return (df["High"] + df["Low"] + df["Close"]) / 3

def _lag(s: pd.Series) -> pd.Series:
    return s.shift(1)

def _pos(arr, index) -> pd.Series:
    return pd.Series(np.clip(np.asarray(arr, float), -1, 1), index=index, dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — TREND
# ══════════════════════════════════════════════════════════════════════════════

def ma_crossover(df, fast=50, slow=200, kind="sma"):
    c = df["Close"]
    f = _ema(c, fast) if kind == "ema" else c.rolling(fast).mean()
    s = _ema(c, slow) if kind == "ema" else c.rolling(slow).mean()
    return _lag(_pos(np.sign(f - s), df.index)).rename("ma_crossover")

def ts_momentum(df, period=252):
    """Time-series momentum: sign of N-bar return."""
    raw = np.sign(df["Close"].pct_change(period).values)
    return _lag(_pos(raw, df.index)).rename("ts_momentum")

def roc_momentum(df, period=20):
    """Rate-of-change momentum (shorter horizons than ts_momentum)."""
    raw = np.sign(df["Close"].pct_change(period).values)
    return _lag(_pos(raw, df.index)).rename("roc_momentum")

def macd_signal(df, fast=12, slow=26, signal=9):
    _, _, hist = _macd(df["Close"], fast, slow, signal)
    return _lag(_pos(np.sign(hist.values), df.index)).rename("macd_signal")

def donchian_breakout(df, period=20):
    """Breakout above/below prior N-bar high/low."""
    hi = df["High"].shift(1).rolling(period).max()
    lo = df["Low"].shift(1).rolling(period).min()
    c  = df["Close"]
    raw = np.where(c > hi, 1., np.where(c < lo, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("donchian_breakout")

def bb_breakout(df, period=20, std=2.0):
    """Trend: go WITH the band break (opposite of bb_reversion)."""
    c = df["Close"]
    lo, _, hi = _bb(c, period, std)
    raw = np.where(c > hi, 1., np.where(c < lo, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("bb_breakout")

def supertrend(df, period=7, mult=3.0):
    hi, lo, c = df["High"].values, df["Low"].values, df["Close"].values
    atr_v = _atr(df, period).values
    hl2   = (hi + lo) / 2
    up_raw = hl2 + mult * atr_v
    dn_raw = hl2 - mult * atr_v

    fu = up_raw.copy(); fd = dn_raw.copy()
    trend = np.ones(len(c))

    for i in range(1, len(c)):
        fu[i] = up_raw[i] if up_raw[i] < fu[i-1] or c[i-1] > fu[i-1] else fu[i-1]
        fd[i] = dn_raw[i] if dn_raw[i] > fd[i-1] or c[i-1] < fd[i-1] else fd[i-1]
        if trend[i-1] == -1 and c[i] > fu[i-1]:
            trend[i] = 1.
        elif trend[i-1] == 1 and c[i] < fd[i-1]:
            trend[i] = -1.
        else:
            trend[i] = trend[i-1]

    return _lag(_pos(trend, df.index)).rename("supertrend")

def parabolic_sar(df, step=0.02, max_af=0.2):
    hi, lo, c = df["High"].values, df["Low"].values, df["Close"].values
    n   = len(c)
    sar = np.zeros(n); ep = np.zeros(n); af = np.zeros(n)
    bull = np.ones(n, dtype=bool)
    sar[0] = lo[0]; ep[0] = hi[0]; af[0] = step; bull[0] = True

    for i in range(1, n):
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
        if bull[i-1]:
            sar[i] = min(sar[i], lo[i-1], lo[i-2] if i > 1 else lo[0])
            if lo[i] < sar[i]:
                bull[i] = False; sar[i] = ep[i-1]; ep[i] = lo[i]; af[i] = step
            else:
                bull[i] = True
                ep[i] = max(ep[i-1], hi[i])
                af[i] = min(af[i-1] + step, max_af) if hi[i] > ep[i-1] else af[i-1]
        else:
            sar[i] = max(sar[i], hi[i-1], hi[i-2] if i > 1 else hi[0])
            if hi[i] > sar[i]:
                bull[i] = True;  sar[i] = ep[i-1]; ep[i] = hi[i]; af[i] = step
            else:
                bull[i] = False
                ep[i] = min(ep[i-1], lo[i])
                af[i] = min(af[i-1] + step, max_af) if lo[i] < ep[i-1] else af[i-1]

    raw = np.where(bull, 1., -1.)
    return _lag(_pos(raw, df.index)).rename("parabolic_sar")

def adx_trend(df, period=14, threshold=25.):
    hi, lo = df["High"], df["Low"]
    up   = hi.diff(); down = -lo.diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.)
    ndm  = np.where((down > up) & (down > 0), down, 0.)
    atr_v = _atr(df, period)
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_v
    ndi  = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_v
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx  = dx.ewm(alpha=1/period, adjust=False).mean()
    raw  = np.where(adx > threshold, np.where(pdi > ndi, 1., -1.), 0.)
    return _lag(_pos(raw, df.index)).rename("adx_trend")

def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    hi, lo, c = df["High"], df["Low"], df["Close"]
    tk = (hi.rolling(tenkan).max() + lo.rolling(tenkan).min()) / 2
    kj = (hi.rolling(kijun).max()  + lo.rolling(kijun).min())  / 2
    # Bull: price above kijun AND tenkan > kijun; Bear: inverse
    bull = (c > kj) & (tk > kj)
    bear = (c < kj) & (tk < kj)
    raw  = np.where(bull, 1., np.where(bear, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("ichimoku")

def linreg_slope(df, period=20):
    c  = df["Close"]
    x  = np.arange(period, dtype=float); xc = x - x.mean()
    norm = (xc**2).sum()
    slope = c.rolling(period).apply(
        lambda y: (xc * (y - y.mean())).sum() / norm, raw=True)
    return _lag(_pos(np.sign(slope.values), df.index)).rename("linreg_slope")

def aroon(df, period=25):
    hi, lo = df["High"], df["Low"]
    aup = hi.rolling(period+1).apply(lambda x: np.argmax(x) / period * 100, raw=True)
    adn = lo.rolling(period+1).apply(lambda x: np.argmin(x) / period * 100, raw=True)
    raw = np.sign((aup - adn).values)
    return _lag(_pos(raw, df.index)).rename("aroon")

def vortex(df, period=14):
    hi, lo, cp = df["High"], df["Low"], df["Close"].shift(1)
    vp  = (hi - lo.shift(1)).abs()
    vm  = (lo - hi.shift(1)).abs()
    tr  = pd.concat([hi-lo, (hi-cp).abs(), (lo-cp).abs()], axis=1).max(axis=1)
    trs = tr.rolling(period).sum().replace(0, np.nan)
    raw = np.sign((vp.rolling(period).sum() - vm.rolling(period).sum()).values)
    return _lag(_pos(raw, df.index)).rename("vortex")

def trix(df, period=12):
    c  = df["Close"]
    e3 = _ema(_ema(_ema(c, period), period), period)
    raw = np.sign(e3.pct_change().values)
    return _lag(_pos(raw, df.index)).rename("trix")

def hull_ma(df, period=20):
    c    = df["Close"]
    half = max(period // 2, 1); sq = max(int(period**0.5), 1)
    hma  = _wma(2 * _wma(c, half) - _wma(c, period), sq)
    raw  = np.sign((hma - hma.shift(1)).values)
    return _lag(_pos(raw, df.index)).rename("hull_ma")

def kama(df, period=10, fast_n=2, slow_n=30):
    c     = df["Close"].values
    er_n  = np.abs(np.diff(c, period, prepend=[np.nan]*period))
    noise = pd.Series(np.abs(np.diff(c, prepend=[np.nan]))).rolling(period).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        er = np.where(noise > 0, er_n / noise, 0.)
    fsc = 2/(fast_n+1); ssc = 2/(slow_n+1)
    sc  = (er * (fsc - ssc) + ssc) ** 2

    k = np.full(len(c), np.nan)
    k[period] = c[period]
    for i in range(period+1, len(c)):
        if not np.isnan(k[i-1]):
            k[i] = k[i-1] + sc[i] * (c[i] - k[i-1])

    raw = np.sign(np.diff(k, prepend=[np.nan]))
    return _lag(_pos(raw, df.index)).rename("kama")

def turtle(df, entry=20, exit_p=10):
    """Turtle breakout: long above N-bar high, short below N-bar low."""
    hi_e = df["High"].shift(1).rolling(entry).max()
    lo_e = df["Low"].shift(1).rolling(entry).min()
    c    = df["Close"]
    raw  = np.where(c > hi_e, 1., np.where(c < lo_e, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("turtle")

def dual_momentum(df, period=252):
    """Absolute momentum: long when N-bar return > 0, flat otherwise."""
    raw = np.where(df["Close"].pct_change(period).values > 0, 1., 0.)
    return _lag(_pos(raw, df.index)).rename("dual_momentum")

def elder_ray(df, period=13):
    ema_v = _ema(df["Close"], period)
    bull  = df["High"] - ema_v
    bear  = df["Low"]  - ema_v
    trend = np.sign((ema_v - ema_v.shift(1)).values)
    raw   = np.where((bull.values > 0) & (trend > 0), 1.,
            np.where((bear.values < 0) & (trend < 0), -1., 0.))
    return _lag(_pos(raw, df.index)).rename("elder_ray")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — MEAN REVERSION
# ══════════════════════════════════════════════════════════════════════════════

def rsi_reversion(df, period=14, lo=30., hi=70.):
    r   = _rsi(df["Close"], period)
    raw = np.where(r < lo, 1., np.where(r > hi, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("rsi_reversion")

def bb_reversion(df, period=20, std=2.0):
    c = df["Close"]; lb, _, ub = _bb(c, period, std)
    raw = np.where(c < lb, 1., np.where(c > ub, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("bb_reversion")

def zscore_reversion(df, period=30, threshold=1.5):
    c   = df["Close"]
    z   = (c - c.rolling(period).mean()) / c.rolling(period).std()
    raw = np.where(z < -threshold, 1., np.where(z > threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("zscore_reversion")

def stochastic(df, k=14, d=3, lo=20., hi=80.):
    c   = df["Close"]
    kv  = 100*(c - df["Low"].rolling(k).min()) / (
          df["High"].rolling(k).max() - df["Low"].rolling(k).min()).replace(0, np.nan)
    dv  = kv.rolling(d).mean()
    raw = np.where(dv < lo, 1., np.where(dv > hi, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("stochastic")

def cci_reversion(df, period=20, threshold=100.):
    tp  = _tp(df); ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - ma) / (0.015 * mad.replace(0, np.nan))
    raw = np.where(cci < -threshold, 1., np.where(cci > threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("cci_reversion")

def williams_r(df, period=14, lo=-80., hi=-20.):
    c   = df["Close"]
    hh  = df["High"].rolling(period).max()
    ll  = df["Low"].rolling(period).min()
    wr  = -100 * (hh - c) / (hh - ll).replace(0, np.nan)
    raw = np.where(wr < lo, 1., np.where(wr > hi, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("williams_r")

def keltner_reversion(df, period=20, mult=2.0):
    mid  = _ema(df["Close"], period)
    band = mult * _atr(df, period)
    c    = df["Close"]
    raw  = np.where(c < mid - band, 1., np.where(c > mid + band, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("keltner_reversion")

def vwap_reversion(df, period=20, z_thresh=1.0):
    tp   = _tp(df)
    vwap = (tp * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    std  = (tp - vwap).rolling(period).std()
    z    = (df["Close"] - vwap) / std.replace(0, np.nan)
    raw  = np.where(z < -z_thresh, 1., np.where(z > z_thresh, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("vwap_reversion")

def percent_b(df, period=20, std=2.0, lo=0.2, hi=0.8):
    c = df["Close"]; lb, _, ub = _bb(c, period, std)
    pb  = (c - lb) / (ub - lb).replace(0, np.nan)
    raw = np.where(pb < lo, 1., np.where(pb > hi, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("percent_b")

def connors_rsi(df, rsi_period=3, streak_period=2, rank_period=100):
    c   = df["Close"]
    rsi_v = _rsi(c, rsi_period)

    # streak (consecutive up/down bars)
    dv = np.sign(c.diff().fillna(0).values)
    streak = np.zeros(len(dv))
    for i in range(1, len(dv)):
        if dv[i] > 0:   streak[i] = max(streak[i-1]+1,  1)
        elif dv[i] < 0: streak[i] = min(streak[i-1]-1, -1)

    streak_rsi = _rsi(pd.Series(streak, index=df.index), streak_period)
    pct_rank = c.pct_change().rolling(rank_period).apply(
        lambda x: (x[:-1] < x[-1]).sum() / (len(x)-1) * 100, raw=True)
    crsi = (rsi_v + streak_rsi + pct_rank) / 3
    raw  = np.where(crsi < 20, 1., np.where(crsi > 80, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("connors_rsi")

def ultimate_osc(df, p1=7, p2=14, p3=28):
    c, hi, lo = df["Close"], df["High"], df["Low"]
    pc  = c.shift(1)
    bp  = c - pd.concat([lo, pc], axis=1).min(axis=1)
    tr  = pd.concat([hi, pc], axis=1).max(axis=1) - pd.concat([lo, pc], axis=1).min(axis=1)
    a1  = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    a2  = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    a3  = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    uo  = 100 * (4*a1 + 2*a2 + a3) / 7
    raw = np.where(uo < 30, 1., np.where(uo > 70, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("ultimate_osc")

def gap_fade(df, threshold=0.01):
    """Fade overnight gaps: gap-up → short, gap-down → long."""
    gap = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    raw = np.where(gap > threshold, -1., np.where(gap < -threshold, 1., 0.))
    return _lag(_pos(raw, df.index)).rename("gap_fade")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — VOLUME
# ══════════════════════════════════════════════════════════════════════════════

def obv_trend(df, period=20):
    d   = df["Close"].diff()
    v   = df["Volume"]
    obv = pd.Series(np.where(d>0, v, np.where(d<0, -v, 0.)), index=df.index).cumsum()
    raw = np.sign((obv - obv.rolling(period).mean()).values)
    return _lag(_pos(raw, df.index)).rename("obv_trend")

def chaikin_mf(df, period=20):
    hl  = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    cmf = (mfm * df["Volume"]).rolling(period).sum() / df["Volume"].rolling(period).sum()
    return _lag(_pos(np.sign(cmf.values), df.index)).rename("chaikin_mf")

def mfi(df, period=14):
    tp  = _tp(df)
    rmf = tp * df["Volume"]
    d   = tp.diff()
    pmf = pd.Series(np.where(d >= 0, rmf, 0.), index=df.index)
    nmf = pd.Series(np.where(d <  0, rmf, 0.), index=df.index)
    mfr = pmf.rolling(period).sum() / nmf.rolling(period).sum().replace(0, np.nan)
    mfi_v = 100 - 100 / (1 + mfr)
    raw = np.where(mfi_v < 20, 1., np.where(mfi_v > 80, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("mfi")

def volume_surge(df, period=20, mult=2.0):
    """Signal in direction of bar when volume > mult × avg."""
    surge = df["Volume"] > mult * df["Volume"].rolling(period).mean()
    direction = np.sign(df["Close"].diff().values)
    raw = np.where(surge, direction, 0.)
    return _lag(_pos(raw, df.index)).rename("volume_surge")

def force_index(df, period=13):
    fi  = df["Close"].diff() * df["Volume"]
    raw = np.sign(_ema(fi, period).values)
    return _lag(_pos(raw, df.index)).rename("force_index")

def chaikin_osc(df, fast=3, slow=10):
    tp  = _tp(df)
    hl  = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    adl = (clv * df["Volume"]).cumsum()
    osc = _ema(adl, fast) - _ema(adl, slow)
    return _lag(_pos(np.sign(osc.values), df.index)).rename("chaikin_osc")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def atr_breakout(df, period=14, mult=2.0):
    c    = df["Close"]
    mid  = c.rolling(period).mean()
    band = mult * _atr(df, period)
    raw  = np.where(c > mid + band, 1., np.where(c < mid - band, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("atr_breakout")

def vol_breakout(df, period=20, mult=1.0):
    """Long/short when today's abs return > mult × avg abs return."""
    ret = df["Close"].pct_change()
    avg = ret.abs().rolling(period).mean()
    raw = np.where(ret.abs() > mult * avg, np.sign(ret.values), 0.)
    return _lag(_pos(raw, df.index)).rename("vol_breakout")

def squeeze_breakout(df, bb_period=20, kc_period=20, bb_std=2.0, kc_mult=1.5):
    """BB squeeze release: trade momentum direction when BB exits Keltner."""
    c             = df["Close"]
    lb, _, ub     = _bb(c, bb_period, bb_std)
    kc_mid        = _ema(c, kc_period)
    kc_band       = kc_mult * _atr(df, kc_period)
    in_squeeze    = (lb > kc_mid - kc_band) & (ub < kc_mid + kc_band)
    hi_n = df["High"].rolling(kc_period).max()
    lo_n = df["Low"].rolling(kc_period).min()
    mom  = _ema(c - (hi_n + lo_n)/2, kc_period)
    raw  = np.where(~in_squeeze, np.sign(mom.values), 0.)
    return _lag(_pos(raw, df.index)).rename("squeeze_breakout")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — PATTERN
# ══════════════════════════════════════════════════════════════════════════════

def engulfing(df):
    o,c,po,pc = df["Open"],df["Close"],df["Open"].shift(1),df["Close"].shift(1)
    bull = (pc < po) & (c > po) & (o < pc)
    bear = (pc > po) & (c < po) & (o > pc)
    raw  = np.where(bull, 1., np.where(bear, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("engulfing")

def three_bar_reversal(df):
    d    = df["Close"].diff()
    dn3  = (d.shift(2) < 0) & (d.shift(1) < 0) & (d < 0)
    up3  = (d.shift(2) > 0) & (d.shift(1) > 0) & (d > 0)
    raw  = np.where(dn3, 1., np.where(up3, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("three_bar_reversal")

def higher_highs_lows(df, period=10):
    """Uptrend when N-bar high AND low both higher than prior N-bar high/low."""
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    hh = hi > hi.shift(period); hl = lo > lo.shift(period)
    lh = hi < hi.shift(period); ll = lo < lo.shift(period)
    raw = np.where(hh & hl, 1., np.where(lh & ll, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("higher_highs_lows")

def pivot_bounce(df, period=5):
    """Price near classic pivot support/resistance levels."""
    ph = df["High"].shift(1).rolling(period).max()
    pl = df["Low"].shift(1).rolling(period).min()
    pc = df["Close"].shift(1).rolling(period).mean()
    pivot = (ph + pl + pc) / 3
    r1 = 2*pivot - pl; s1 = 2*pivot - ph
    cur = df["Close"]
    tol = (ph - pl) * 0.1
    raw = np.where((cur - s1).abs() < tol, 1.,
          np.where((cur - r1).abs() < tol, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("pivot_bounce")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES — COMPOSITE
# ══════════════════════════════════════════════════════════════════════════════

def macd_rsi_combo(df, fast=12, slow=26, signal=9,
                   rsi_period=14, rsi_lo=40., rsi_hi=60.):
    """MACD histogram direction confirmed by RSI above/below neutral."""
    _, _, hist = _macd(df["Close"], fast, slow, signal)
    r = _rsi(df["Close"], rsi_period)
    raw = np.where((hist > 0) & (r > rsi_hi), 1.,
          np.where((hist < 0) & (r < rsi_lo), -1., 0.))
    return _lag(_pos(raw, df.index)).rename("macd_rsi_combo")

def triple_screen(df, trend_period=20, osc_period=14, osc_lo=40., osc_hi=60.):
    """Elder triple screen: weekly trend + daily RSI oscillator."""
    c = df["Close"]
    weekly_trend = np.sign((_ema(c, trend_period) - _ema(c, trend_period).shift(1)).values)
    rsi_v = _rsi(c, osc_period)
    raw = np.where((weekly_trend > 0) & (rsi_v < osc_lo), 1.,
          np.where((weekly_trend < 0) & (rsi_v > osc_hi), -1., 0.))
    return _lag(_pos(raw, df.index)).rename("triple_screen")

def chandelier_exit(df, period=22, mult=3.0):
    """Long when close above N-bar-high minus mult*ATR; short below."""
    atr_v = _atr(df, period)
    ls    = df["High"].rolling(period).max() - mult * atr_v
    ss    = df["Low"].rolling(period).min()  + mult * atr_v
    c     = df["Close"]
    raw   = np.where(c > ls, 1., np.where(c < ss, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("chandelier_exit")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES FROM ACADEMIC LITERATURE  (George & Hwang 2004, Jegadeesh 1990,
# Ang et al 2006, Crabel 1990, Lou et al 2019, Coppock 1962)
# ══════════════════════════════════════════════════════════════════════════════

def high52_momentum(df, period=252, threshold=0.75):
    """Proximity to N-period high/low as momentum signal (George & Hwang 2004)."""
    c   = df["Close"]
    hi  = c.rolling(period).max()
    lo  = c.rolling(period).min()
    rng = (hi - lo).clip(lower=1e-8)
    prox = (c - lo) / rng          # 0 = at period low, 1 = at period high
    raw  = np.where(prox > threshold, 1., np.where(prox < 1 - threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("high52_momentum")


def short_reversal(df, period=5):
    """Short-term reversal: bet against recent N-day price change (Jegadeesh 1990)."""
    raw = -np.sign(df["Close"].pct_change(period).values)
    return _lag(_pos(raw, df.index)).rename("short_reversal")


def low_vol_regime(df, period=20, ma_period=120):
    """
    Long when Garman-Klass realized vol < its own rolling MA (Ang et al 2006).
    Short when vol spikes well above MA.
    """
    gk_var = (0.5 * np.log((df["High"] / df["Low"]).clip(lower=1e-8)) ** 2
              - (2 * np.log(2) - 1) * np.log((df["Close"] / df["Open"]).clip(lower=1e-8)) ** 2)
    vol    = gk_var.rolling(period).mean().pow(0.5) * np.sqrt(252)
    vol_ma = vol.rolling(ma_period).mean()
    raw    = np.where(vol < vol_ma * 0.8, 1., np.where(vol > vol_ma * 1.4, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("low_vol_regime")


def nr_breakout(df, nr_period=7, exit_bars=5):
    """
    Narrow-range breakout (Crabel 1990): trade the directional squeeze
    when today's H-L is the tightest in the last nr_period bars.
    """
    hi    = df["High"]
    lo    = df["Low"]
    c     = df["Close"]
    rng   = hi - lo
    trend = np.sign(c.pct_change(20))
    is_nr = rng < rng.rolling(nr_period).min().shift(1)

    pos      = pd.Series(0.0, index=df.index)
    in_trade = 0
    held     = 0

    for i in range(len(df)):
        if held >= exit_bars:
            in_trade = 0
            held     = 0
        if in_trade != 0:
            pos.iloc[i] = in_trade
            held += 1
        elif is_nr.iloc[i] and not np.isnan(trend.iloc[i]):
            pos.iloc[i] = trend.iloc[i]
            in_trade     = trend.iloc[i]
            held         = 1

    return _lag(pos.clip(-1, 1)).rename("nr_breakout")


def overnight_gap(df, period=3, threshold=0.002):
    """
    Overnight gap momentum (Lou, Polk & Skouras 2019):
    smoothed gap direction predicts near-term drift.
    """
    gap  = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1).clip(lower=1e-8)
    smth = gap.rolling(period).mean()
    raw  = np.where(smth > threshold, 1., np.where(smth < -threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("overnight_gap")


def coppock_curve(df, roc1=252, roc2=189, wma_period=63):
    """
    Coppock Curve (Coppock 1962): WMA of two long-period ROC sums.
    Positive curve → long, negative → short.
    """
    c         = df["Close"]
    raw_curve = c.pct_change(roc1) + c.pct_change(roc2)
    w         = np.arange(1, wma_period + 1, dtype=float)
    w        /= w.sum()
    curve     = raw_curve.rolling(wma_period).apply(lambda x: np.dot(x, w), raw=True)
    return _lag(_pos(np.sign(curve.values), df.index)).rename("coppock_curve")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 7 — (CMO-adaptive Fisher, RVI divergence, Adaptive RSI,
#            Elder Impulse System, Klinger Volume Oscillator,
#            Price Oscillator, VWAP momentum)
# ══════════════════════════════════════════════════════════════════════════════

def cmo_fisher(df, fisher_period=10, cmo_period=14, score_thresh=0.30):
    """
    CMO-adaptive Fisher: Fisher direction weighted by CMO trend strength.
    High |CMO| = strong trend → take Fisher signal; low → stay flat.
    """
    c    = df["Close"]
    hi   = df["High"].rolling(fisher_period).max()
    lo   = df["Low"].rolling(fisher_period).min()
    val  = (2.0 * (c - lo) / (hi - lo + 1e-8) - 1.0).clip(-0.999, 0.999)
    fish = 0.5 * np.log((1 + val) / (1 - val))

    d    = c.diff()
    up   = d.clip(lower=0).rolling(cmo_period).sum()
    dn   = (-d).clip(lower=0).rolling(cmo_period).sum()
    cmo  = (up - dn) / (up + dn + 1e-8)   # -1 to +1

    score = fish * cmo.abs()
    raw   = np.where(score > score_thresh, 1.,
            np.where(score < -score_thresh, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("cmo_fisher")


def rvi_divergence(df, rvi_period=10, lookback=5):
    """
    RVI-price divergence: bullish when price lower low but RVI higher low,
    bearish when price higher high but RVI lower high.
    """
    co   = df["Close"] - df["Open"]
    hl   = (df["High"] - df["Low"]).clip(lower=1e-8)
    def _sym4(s):
        return (s + 2*s.shift(1) + 2*s.shift(2) + s.shift(3)) / 6.0
    rvi  = _sym4(co) / _sym4(hl)

    c    = df["Close"]
    pr_lo_now  = c == c.rolling(lookback).min()
    pr_lo_prev = c.shift(lookback) == c.shift(lookback).rolling(lookback).min()
    rv_lo_now  = rvi == rvi.rolling(lookback).min()
    rv_lo_prev = rvi.shift(lookback) == rvi.shift(lookback).rolling(lookback).min()

    bull = pr_lo_now & pr_lo_prev & rv_lo_now & rv_lo_prev \
           & (c < c.shift(lookback)) & (rvi > rvi.shift(lookback))

    pr_hi_now  = c == c.rolling(lookback).max()
    pr_hi_prev = c.shift(lookback) == c.shift(lookback).rolling(lookback).max()
    rv_hi_now  = rvi == rvi.rolling(lookback).max()
    rv_hi_prev = rvi.shift(lookback) == rvi.shift(lookback).rolling(lookback).max()

    bear = pr_hi_now & pr_hi_prev & rv_hi_now & rv_hi_prev \
           & (c > c.shift(lookback)) & (rvi < rvi.shift(lookback))

    pos = pd.Series(0.0, index=df.index)
    pos[bull] =  1.0
    pos[bear] = -1.0
    pos = pos.replace(0, np.nan).ffill(limit=lookback).fillna(0)
    return _lag(pos.clip(-1, 1)).rename("rvi_divergence")


def adaptive_rsi(df, base_period=14, fast_n=2, slow_n=30):
    """
    Adaptive RSI: RSI period scaled by Kaufman ER — shorter in trends, longer in chop.
    Long when adaptive RSI < 30, short when > 70.
    """
    c    = df["Close"]
    er_p = base_period
    d    = c.diff(er_p).abs()
    path = c.diff().abs().rolling(er_p).sum().clip(lower=1e-8)
    er   = d / path
    # Adaptive smoothing constant
    fast_sc = 2 / (fast_n + 1)
    slow_sc = 2 / (slow_n + 1)
    sc      = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Build adaptive MA, then RSI-style gain/loss on it
    ama = c.copy().values.astype(float)
    sc_v = sc.fillna(0).values
    for i in range(1, len(ama)):
        ama[i] = ama[i-1] + sc_v[i] * (c.iloc[i] - ama[i-1])
    ama_s = pd.Series(ama, index=c.index)

    d2 = ama_s.diff()
    g  = d2.clip(lower=0).rolling(base_period).mean()
    l  = (-d2).clip(lower=0).rolling(base_period).mean().clip(lower=1e-8)
    arsi = 100 - 100 / (1 + g / l)
    raw  = np.where(arsi < 30, 1., np.where(arsi > 70, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("adaptive_rsi")


def elder_impulse(df, ema_period=13, macd_fast=12, macd_slow=26, macd_sig=9):
    """
    Elder Impulse System (Alexander Elder):
    green bar = EMA rising AND MACD histogram rising → long.
    red bar   = EMA falling AND MACD histogram falling → short.
    """
    c           = df["Close"]
    ema         = _ema(c, ema_period)
    _, _, hist  = _macd(c, macd_fast, macd_slow, macd_sig)
    ema_up      = ema > ema.shift(1)
    hist_up     = hist > hist.shift(1)
    raw         = np.where(ema_up & hist_up,   1.,
                  np.where(~ema_up & ~hist_up, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("elder_impulse")


def klinger_osc(df, fast=34, slow=55, signal=13):
    """
    Klinger Volume Oscillator: EMA difference of volume-force.
    Volume force = volume * (2 * ((close-low-high+close)/(high-low)) - 1) * trend * 100
    """
    c   = df["Close"]
    h   = df["High"]
    lo  = df["Low"]
    v   = df["Volume"].replace(0, np.nan).fillna(1)
    hl  = (h - lo).clip(lower=1e-8)
    dm  = h - lo
    cm  = np.where(
        (dm + (c - lo)) > (hl + (c - lo).abs()),
        dm + (c - lo),
        hl + (c - lo).abs()
    )
    trend = np.sign(c.diff())
    vf    = v * (2 * dm / pd.Series(cm, index=c.index).clip(lower=1e-8) - 1) * trend * 100
    kvo   = _ema(vf, fast) - _ema(vf, slow)
    sig_l = _ema(kvo, signal)
    raw   = np.sign((kvo - sig_l).values)
    return _lag(_pos(raw, df.index)).rename("klinger_osc")


def price_oscillator(df, fast=10, slow=30, signal=9):
    """
    Price Oscillator (PPO): (fast EMA - slow EMA) / slow EMA * 100.
    Trade the signal-line crossover (same logic as MACD but % normalized).
    """
    c    = df["Close"]
    ppo  = (_ema(c, fast) - _ema(c, slow)) / _ema(c, slow).clip(lower=1e-8) * 100
    sig  = _ema(ppo, signal)
    raw  = np.sign((ppo - sig).values)
    return _lag(_pos(raw, df.index)).rename("price_oscillator")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 6 — (Kaufman ER regime, McGinley Dynamic, CCI trend-follow,
#            Volume rate-of-change, HV ratio regime, Mean-reversion z + trend,
#            Breadth momentum proxy)
# ══════════════════════════════════════════════════════════════════════════════

def er_regime(df, er_period=10, mom_period=20, er_thresh=0.40):
    """
    Kaufman Efficiency Ratio as regime filter:
    ER > thresh → trending → follow price momentum.
    ER < thresh → choppy  → fade price momentum (mean revert).
    """
    c    = df["Close"]
    d    = c.diff(er_period).abs()
    path = c.diff().abs().rolling(er_period).sum().clip(lower=1e-8)
    er   = d / path
    mom  = np.sign(c.pct_change(mom_period).values)
    raw  = np.where(er > er_thresh,  mom,
           np.where(er < er_thresh * 0.5, -mom, 0.))
    return _lag(_pos(raw, df.index)).rename("er_regime")


def mcginley_dynamic(df, period=14, k=0.6, fast=5, slow=20):
    """
    McGinley Dynamic crossover: self-adjusting MA that speeds up in fast markets,
    slows in slow ones.  MD[i] = MD[i-1] + (close - MD[i-1]) / (k * period * (close/MD)^4)
    Long when fast MD > slow MD.
    """
    c = df["Close"].values
    n = len(c)

    def md_series(per):
        md = np.zeros(n)
        md[0] = c[0]
        for i in range(1, n):
            ratio = (c[i] / md[i-1]) if md[i-1] > 0 else 1.0
            md[i] = md[i-1] + (c[i] - md[i-1]) / (k * per * ratio**4)
        return md

    fast_md = md_series(fast)
    slow_md = md_series(slow)
    raw     = np.sign(fast_md - slow_md)
    return _lag(_pos(raw, df.index)).rename("mcginley_dynamic")


def cci_trend(df, period=20, threshold=50.):
    """
    CCI as trend-follower (opposite of cci_reversion):
    long when CCI > +threshold (bullish momentum), short when < -threshold.
    """
    tp  = _tp(df)
    ma  = tp.rolling(period).mean()
    md  = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - ma) / (0.015 * md.clip(lower=1e-8))
    raw = np.where(cci > threshold, 1., np.where(cci < -threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("cci_trend")


def volume_roc(df, period=14, signal_period=10):
    """
    Volume Rate-of-Change momentum: rising volume + rising price → long.
    High volume contraction → possible reversal.
    """
    v    = df["Volume"].replace(0, np.nan)
    vroc = v.pct_change(period)
    c    = df["Close"]
    prc  = np.sign(c.pct_change(period).values)
    vroc_sig = np.sign(vroc.rolling(signal_period).mean().values)
    # only trade when volume and price move together
    raw  = np.where((vroc_sig > 0) & (prc > 0),  1.,
           np.where((vroc_sig < 0) & (prc < 0), -1., 0.))
    return _lag(_pos(raw, df.index)).rename("volume_roc")


def hv_ratio(df, fast_period=10, slow_period=100, thresh_hi=1.5, thresh_lo=0.7):
    """
    Historical Volatility Ratio regime filter (HV_fast / HV_slow):
    Low ratio (quiet vs history) → trend-follow; high ratio → mean revert.
    """
    rdly   = np.log(df["Close"]).diff()
    hv_f   = rdly.rolling(fast_period).std() * np.sqrt(252)
    hv_s   = rdly.rolling(slow_period).std() * np.sqrt(252)
    ratio  = (hv_f / hv_s.clip(lower=1e-8))
    mom    = np.sign(df["Close"].pct_change(fast_period * 2).values)
    raw    = np.where(ratio < thresh_lo,  mom,
             np.where(ratio > thresh_hi, -mom, 0.))
    return _lag(_pos(raw, df.index)).rename("hv_ratio")


def zscore_trend(df, z_period=20, trend_period=50, z_thresh=1.5):
    """
    Z-score with trend filter: mean-revert only when z-score is extreme
    AND price is on the right side of long-run trend. Avoids fading strong trends.
    """
    c     = df["Close"]
    z     = (c - c.rolling(z_period).mean()) / c.rolling(z_period).std().clip(lower=1e-8)
    trend = np.sign(c - c.rolling(trend_period).mean())
    # fade extreme z-scores BUT only when counter-trend (safer reversion)
    raw   = np.where((z < -z_thresh) & (trend > 0),  1.,    # oversold in uptrend
            np.where((z >  z_thresh) & (trend < 0), -1., 0.))  # overbought in downtrend
    return _lag(_pos(raw, df.index)).rename("zscore_trend")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 5 — (Ehlers Fisher Transform, Relative Vigor Index, Chande Momentum,
#            DeMarker, Stochastic RSI, Hurst regime, Psychological Line)
# ══════════════════════════════════════════════════════════════════════════════

def fisher_transform(df, period=10):
    """
    Ehlers Fisher Transform: converts price to near-Gaussian distribution.
    Extreme readings (±2.5) signal reversals. Trade direction of the transform.
    """
    c    = df["Close"]
    hi   = df["High"].rolling(period).max()
    lo   = df["Low"].rolling(period).min()
    rng  = (hi - lo).clip(lower=1e-8)
    val  = 2.0 * ((c - lo) / rng - 0.5)
    val  = val.clip(-0.999, 0.999)
    fish = 0.5 * np.log((1 + val) / (1 - val))
    # signal: direction of fisher (trend), invert at extremes (reversal)
    raw  = np.sign(fish.values)
    return _lag(_pos(raw, df.index)).rename("fisher_transform")


def relative_vigor(df, period=10):
    """
    Relative Vigor Index (RVI): ratio of close-open range to high-low range,
    smoothed. Positive = bullish vigor (closes near high), negative = bearish.
    """
    co   = df["Close"] - df["Open"]
    hl   = (df["High"] - df["Low"]).clip(lower=1e-8)
    # 4-bar symmetric weighted average (Ehlers triangular weights)
    def _sym4(s):
        return (s + 2*s.shift(1) + 2*s.shift(2) + s.shift(3)) / 6.0
    rvi  = _sym4(co) / _sym4(hl)
    sig  = (rvi + 2*rvi.shift(1) + 2*rvi.shift(2) + rvi.shift(3)) / 6.0
    raw  = np.sign((rvi - sig).values)
    return _lag(_pos(raw, df.index)).rename("relative_vigor")


def chande_momentum(df, period=20, threshold=0.0):
    """
    Chande Momentum Oscillator: (sum_up - sum_down) / (sum_up + sum_down) * 100.
    Ranges -100 to +100. Positive = uptrend; threshold filters noise.
    """
    d    = df["Close"].diff()
    up   = d.clip(lower=0).rolling(period).sum()
    dn   = (-d).clip(lower=0).rolling(period).sum()
    denom = (up + dn).clip(lower=1e-8)
    cmo  = (up - dn) / denom * 100
    raw  = np.where(cmo > threshold, 1., np.where(cmo < -threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("chande_momentum")


def demarker(df, period=14, hi_thresh=0.7, lo_thresh=0.3):
    """
    DeMarker indicator (Tom DeMark): compares intrabar highs/lows to prior bar.
    Overbought > hi_thresh → short; oversold < lo_thresh → long (mean reversion).
    """
    dh   = (df["High"] - df["High"].shift(1)).clip(lower=0)
    dl   = (df["Low"].shift(1) - df["Low"]).clip(lower=0)
    dem  = dh.rolling(period).mean() / (dh.rolling(period).mean()
                                        + dl.rolling(period).mean()).clip(lower=1e-8)
    raw  = np.where(dem < lo_thresh, 1., np.where(dem > hi_thresh, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("demarker")


def stoch_rsi(df, rsi_period=14, stoch_period=14, smooth_k=3, lo=20., hi=80.):
    """
    Stochastic RSI: applies stochastic formula to RSI values.
    More sensitive than plain RSI or stochastic alone.
    """
    rsi  = _rsi(df["Close"], rsi_period)
    rsi_lo = rsi.rolling(stoch_period).min()
    rsi_hi = rsi.rolling(stoch_period).max()
    k    = 100 * (rsi - rsi_lo) / (rsi_hi - rsi_lo + 1e-8)
    k_sm = k.rolling(smooth_k).mean()
    raw  = np.where(k_sm < lo, 1., np.where(k_sm > hi, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("stoch_rsi")


def hurst_regime(df, period=100, trend_thresh=0.55, mr_thresh=0.45):
    """
    Hurst exponent regime filter: H > 0.55 → trending (follow momentum),
    H < 0.45 → mean-reverting (fade moves). Uses RS (rescaled range) method.
    Combined with price direction for position sign.
    """
    c    = df["Close"]
    rdly = np.log(c).diff()

    def hurst_rs(window):
        if len(window) < 10:
            return 0.5
        mean  = np.mean(window)
        devs  = np.cumsum(window - mean)
        R     = devs.max() - devs.min()
        S     = np.std(window, ddof=1)
        if S < 1e-10:
            return 0.5
        return np.log(R / S) / np.log(len(window))

    H_vals = rdly.rolling(period).apply(hurst_rs, raw=True)
    mom    = np.sign(c.pct_change(period // 2).values)

    raw = np.where(H_vals > trend_thresh, mom,
          np.where(H_vals < mr_thresh,   -mom, 0.))
    return _lag(_pos(raw, df.index)).rename("hurst_regime")


def psych_line(df, period=12, hi_thresh=75., lo_thresh=25.):
    """
    Psychological Line: % of up-closes in last N bars.
    High % → overbought (short); Low % → oversold (long). Mean reversion.
    """
    up   = (df["Close"].diff() > 0).astype(float)
    pct  = up.rolling(period).mean() * 100
    raw  = np.where(pct < lo_thresh, 1., np.where(pct > hi_thresh, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("psych_line")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 4 — (Baz 2015, Novy-Marx 2012, Daniel-Moskowitz crash protection,
#            Alpha Architect QMOM consistency, signed range momentum,
#            Moskowitz-Ooi-Pedersen multi-lookback TSMOM)
# ══════════════════════════════════════════════════════════════════════════════

def baz_trend(df, pairs=((8,24),(16,48),(32,96)), ewmstd_span=63,
              weights=(1/3, 1/3, 1/3)):
    """
    Baz et al 2015 multi-EWMA trend signal: vol-normalised EWMA pairs
    combined into a single composite position.
    """
    c    = df["Close"]
    rdly = c.pct_change()
    evol = rdly.ewm(span=ewmstd_span, adjust=False).std().clip(lower=1e-8)
    composite = pd.Series(0.0, index=c.index)
    for (s, L), w in zip(pairs, weights):
        raw = (c.ewm(span=s, adjust=False).mean()
               - c.ewm(span=L, adjust=False).mean())
        q   = raw / (0.89 * evol * c.clip(lower=1e-8))
        composite += w * q
    return _lag(_pos(np.sign(composite.values), df.index)).rename("baz_trend")


def intermediate_mom(df, start=252, end=126):
    """
    Intermediate momentum (Novy-Marx 2012): return from t-start to t-end.
    Standard 12-1 is (252,21); Novy-Marx uses the older half of that window.
    """
    c   = df["Close"]
    raw = np.sign((c.shift(end) / c.shift(start) - 1).values)
    return _lag(_pos(raw, df.index)).rename("intermediate_mom")


def mom_crash_protect(df, mom_period=252, skip=21,
                      bear_lookback=504, bear_thresh=-0.20, damp=0.0):
    """
    Momentum with crash-risk off-switch (Daniel & Moskowitz 2016).
    Zeros (or damps) position when market in bear + early rebound.
    """
    c        = df["Close"]
    raw      = np.sign(c.pct_change(mom_period).shift(skip).values)
    bear     = (c / c.shift(bear_lookback) - 1) < bear_thresh
    rebound  = c.pct_change(21) > 0
    crash    = bear & rebound
    pos      = pd.Series(raw, index=c.index)
    pos[crash] = pos[crash] * damp
    return _lag(_pos(pos.values, df.index)).rename("mom_crash_protect")


def consistent_mom(df, formation=252, skip=21, leg_months=12,
                   consistency_thresh=0.58):
    """
    QMOM path-quality filter (Alpha Architect / Gray & Vogel):
    only trade momentum if >= threshold fraction of monthly legs positive.
    """
    c        = df["Close"]
    signals  = np.zeros(len(c))
    leg_days = formation // leg_months

    for i in range(formation + skip, len(c)):
        raw_mom = c.iloc[i - skip] / c.iloc[i - skip - formation] - 1
        if raw_mom == 0:
            continue
        legs = [
            c.iloc[i - skip - j * leg_days] / c.iloc[i - skip - (j + 1) * leg_days] - 1
            for j in range(leg_months)
            if i - skip - (j + 1) * leg_days >= 0
        ]
        if not legs:
            continue
        consistency = sum(1 for r in legs if r > 0) / len(legs)
        if consistency >= consistency_thresh:
            signals[i] = np.sign(raw_mom)

    return _lag(_pos(signals, df.index)).rename("consistent_mom")


def signed_range_mom(df, period=21, threshold=0.10):
    """
    Signed range momentum (Llorente et al 2002):
    cumulative directional intraday pressure over N bars.
    signal = Σ (H-L)/C * sign(C-O)  /  Σ |H-L|/C
    """
    sr    = (df["High"] - df["Low"]) / df["Close"].clip(lower=1e-8) \
            * np.sign(df["Close"] - df["Open"])
    abs_r = (df["High"] - df["Low"]) / df["Close"].clip(lower=1e-8)
    roll_sr  = sr.rolling(period).sum()
    roll_abs = abs_r.rolling(period).sum().clip(lower=1e-8)
    norm     = roll_sr / roll_abs
    raw      = np.where(norm > threshold, 1., np.where(norm < -threshold, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("signed_range_mom")


def multi_tsmom(df, lookbacks=(21, 63, 126, 252), weights=None):
    """
    Multi-lookback TSMOM composite (Moskowitz, Ooi, Pedersen 2012 Table 2):
    equal-weight (or custom-weight) sign of return across lookbacks.
    """
    c   = df["Close"]
    w   = weights if weights is not None else [1/len(lookbacks)] * len(lookbacks)
    sig = sum(wi * np.sign(c.pct_change(lb).fillna(0).values)
              for wi, lb in zip(w, lookbacks))
    raw = np.sign(sig)
    return _lag(_pos(raw, df.index)).rename("multi_tsmom")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 3 — (DEMA/TEMA crossover, price acceleration, turn-of-month,
#            relative strength vs benchmark, trend intensity index,
#            MA-regime filter, 52-week range momentum combo)
# ══════════════════════════════════════════════════════════════════════════════

def dema_crossover(df, fast=10, slow=40):
    """
    DEMA crossover: 2*EMA(n) - EMA(EMA(n)) — more responsive than EMA.
    Long when fast DEMA > slow DEMA.
    """
    c = df["Close"]
    def dema(s, n):
        e = _ema(s, n)
        return 2 * e - _ema(e, n)
    raw = np.sign((dema(c, fast) - dema(c, slow)).values)
    return _lag(_pos(raw, df.index)).rename("dema_crossover")


def tema_crossover(df, fast=10, slow=40):
    """
    TEMA crossover: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)) — even tighter lag.
    """
    c = df["Close"]
    def tema(s, n):
        e1 = _ema(s, n)
        e2 = _ema(e1, n)
        e3 = _ema(e2, n)
        return 3 * e1 - 3 * e2 + e3
    raw = np.sign((tema(c, fast) - tema(c, slow)).values)
    return _lag(_pos(raw, df.index)).rename("tema_crossover")


def price_acceleration(df, period=20, smooth=5):
    """
    Momentum of momentum — second derivative of price.
    Long when ROC is itself accelerating (trend strengthening).
    """
    c   = df["Close"]
    roc = c.pct_change(period)
    acc = roc.diff(smooth)
    raw = np.sign(acc.values)
    return _lag(_pos(raw, df.index)).rename("price_acceleration")


def turn_of_month(df, days_before=4, days_after=4):
    """
    Turn-of-month effect: long the last `days_before` + first `days_after`
    trading days of each month; flat otherwise.
    Documented: Ariel 1987, Lakonishok & Smidt 1988.
    """
    idx  = df.index
    month_ends   = idx.to_series().groupby([idx.year, idx.month]).transform("last")
    month_starts = idx.to_series().groupby([idx.year, idx.month]).transform("first")

    pos = pd.Series(0.0, index=idx)
    for i, date in enumerate(idx):
        # position in month (1-indexed from start, -1 from end)
        same_month = idx[(idx.year == date.year) & (idx.month == date.month)]
        pos_from_end   = (same_month[-1] - date).days
        pos_from_start = (date - same_month[0]).days
        # rough check: within days_before of end or days_after of start
        rank_from_start = list(same_month).index(date)
        rank_from_end   = len(same_month) - 1 - rank_from_start
        if rank_from_end < days_before or rank_from_start < days_after:
            pos.iloc[i] = 1.0

    return _lag(pos).rename("turn_of_month")


def rel_strength_vs_market(df, period=63, market_col="Close"):
    """
    Relative strength vs SPY: long when asset return > SPY return over N days.
    Uses SPY from the same universe data — if df IS SPY, always flat.
    Falls back to absolute momentum if no external benchmark.
    Single-asset version: compare trailing return to zero (i.e. absolute momentum proxy).
    """
    c   = df["Close"]
    ret = c.pct_change(period)
    # Without cross-asset comparison, use zero as threshold (= absolute momentum)
    raw = np.sign(ret.values)
    return _lag(_pos(raw, df.index)).rename("rel_strength_vs_market")


def trend_intensity(df, period=30):
    """
    Trend Intensity Index (TII): % of closes in last N bars above the
    midpoint MA → measures how consistently price is above/below trend.
    Long > 50%, short < 50%. (Davis 2005)
    """
    c   = df["Close"]
    ma  = c.rolling(period).mean()
    above = c.rolling(period).apply(lambda x: (x > x.mean()).sum() / len(x), raw=True)
    raw   = np.where(above > 0.60, 1., np.where(above < 0.40, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("trend_intensity")


def ma_regime_momentum(df, regime_ma=200, mom_period=126, skip=21):
    """
    Absolute momentum gated by 200-day MA regime filter.
    Long only when close > 200MA AND trailing return positive.
    Short only when close < 200MA AND trailing return negative.
    Flat when signals conflict. (Faber 2007 / Antonacci 2014 hybrid)
    """
    c      = df["Close"]
    ma200  = c.rolling(regime_ma).mean()
    mom    = c.pct_change(mom_period).shift(skip)
    above  = c > ma200
    raw    = np.where(above & (mom > 0),  1.,
             np.where(~above & (mom < 0), -1., 0.))
    return _lag(_pos(raw, df.index)).rename("ma_regime_momentum")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH 2 — (BLL 1992, Da-Gurun-Warachka 2014, Lee-Swaminathan 2000,
#            Heston-Sadka 2008, Blitz-Huij-Martens 2011, Daniel-Moskowitz 2016)
# ══════════════════════════════════════════════════════════════════════════════

def bll_filter(df, pct=0.01, hold=0):
    """
    BLL filter rule (Brock, Lakonishok, LeBaron 1992):
    long when price rises pct% from running trough; short when falls pct% from peak.
    Optional hold_period: go flat after N bars (0 = hold until reversal).
    """
    c       = df["Close"].values
    n       = len(c)
    raw     = np.zeros(n)
    state   = 0
    peak    = c[0]
    trough  = c[0]
    held    = 0

    for i in range(1, n):
        if hold > 0 and held >= hold:
            state  = 0
            held   = 0
            peak   = c[i]
            trough = c[i]

        if state != 1:
            trough = min(trough, c[i])
            if c[i] > trough * (1 + pct):
                state  = 1
                peak   = c[i]
                held   = 0
        if state != -1:
            peak = max(peak, c[i])
            if c[i] < peak * (1 - pct):
                state  = -1
                trough = c[i]
                held   = 0

        raw[i] = state
        if hold > 0 and state != 0:
            held += 1

    return _lag(_pos(raw, df.index)).rename("bll_filter")


def frog_in_pan(df, lookback=252, skip=21, id_cutoff=0.50):
    """
    Frog-in-the-Pan momentum (Da, Gurun, Warachka 2014):
    weight raw momentum by information continuity — continuous drifts outperform
    discrete jumps. ID < 0.5 means most days support the trend direction.
    """
    c    = np.log(df["Close"])
    rdly = c.diff()

    signals = np.zeros(len(df))
    for i in range(lookback + skip, len(df)):
        window = rdly.iloc[i - skip - lookback : i - skip]
        R      = window.sum()
        if abs(R) < 1e-10:
            continue
        n_against = (window < 0).sum() if R > 0 else (window > 0).sum()
        ID        = n_against / len(window)
        if ID < id_cutoff:
            signals[i] = np.sign(R)

    return _lag(_pos(signals, df.index)).rename("frog_in_pan")


def vol_mom(df, mom_period=126, vol_period=126, skip=21, vol_weight=0.5):
    """
    Volume-momentum interaction (Lee & Swaminathan 2000):
    momentum filtered by relative volume — quiet winners persist, loud losers reverse.
    Combined score = mom_return - vol_weight * vol_ratio_zscore.
    Single-asset: long if mom > 0 and vol below its MA; short if mom < 0 and vol below MA.
    """
    c         = df["Close"]
    v         = df["Volume"].replace(0, np.nan)
    mom       = c.pct_change(mom_period).shift(skip)
    vol_ratio = v.rolling(vol_period).mean() / v.rolling(vol_period * 2).mean()
    vol_z     = (vol_ratio - vol_ratio.rolling(vol_period).mean()) / (
                 vol_ratio.rolling(vol_period).std().clip(lower=1e-8))
    score     = np.sign(mom) - vol_weight * np.sign(vol_z)
    raw       = np.where(score > 0.5, 1., np.where(score < -0.5, -1., 0.))
    return _lag(_pos(raw, df.index)).rename("vol_mom")


def seasonality(df, n_years=3):
    """
    Same-month return seasonality (Heston & Sadka 2008):
    average return in the same calendar month across prior n_years predicts this month.
    Signal set monthly on the first bar of each month.
    """
    c     = df["Close"]
    ret   = c.pct_change()
    month = df.index.month

    signals = pd.Series(0.0, index=df.index)
    prev_m  = None

    for i in range(len(df)):
        m = month[i]
        if m == prev_m:
            signals.iloc[i] = signals.iloc[i - 1]
            continue
        prev_m = m

        same_m_rets = []
        for yr in range(1, n_years + 1):
            mask = (month == m) & (df.index.year == df.index[i].year - yr)
            if mask.sum() > 0:
                same_m_rets.append(ret[mask].sum())

        if same_m_rets:
            signals.iloc[i] = np.sign(np.mean(same_m_rets))

    return _lag(signals.clip(-1, 1)).rename("seasonality")


def residual_mom(df, lookback=252, skip=21, market_ser=None):
    """
    Residual / alpha momentum (Blitz, Huij, Martens 2011):
    rank by CAPM alpha t-stat rather than raw return — reduces crash risk.
    market_ser: externally supplied SPY return series (aligned to df index).
    Falls back to raw momentum if market_ser unavailable.
    """
    c    = df["Close"]
    rdly = c.pct_change()

    if market_ser is None:
        raw = np.sign(rdly.rolling(lookback).sum().shift(skip))
        return _lag(_pos(raw.values, df.index)).rename("residual_mom")

    signals = np.zeros(len(df))
    mkt     = market_ser.reindex(df.index).fillna(0)

    for i in range(lookback + skip, len(df)):
        r_s = rdly.iloc[i - skip - lookback : i - skip].values
        r_m = mkt.iloc[i - skip - lookback : i - skip].values
        if len(r_s) < 60 or np.std(r_m) < 1e-10:
            continue
        b      = np.cov(r_s, r_m)[0, 1] / np.var(r_m)
        alpha  = np.mean(r_s) - b * np.mean(r_m)
        resid  = r_s - (alpha + b * r_m)
        se     = np.std(resid) / np.sqrt(len(resid))
        if se > 0:
            signals[i] = np.sign(alpha / se)

    return _lag(_pos(signals, df.index)).rename("residual_mom")


def vol_scaled_mom(df, mom_period=252, skip=21, rv_window=21,
                   vol_target=0.12, damp=0.5, down_thresh=-0.20,
                   market_ser=None):
    """
    Vol-scaled momentum with crash damper (Daniel & Moskowitz 2016):
    position = sign(mom) * (vol_target / realized_vol); halved in distressed markets.
    """
    c    = df["Close"]
    rdly = np.log(c).diff()
    mom  = c.pct_change(mom_period).shift(skip)

    rv   = rdly.rolling(rv_window).var() * 252
    rv   = rv.clip(lower=1e-8)
    scale = (vol_target / rv.pow(0.5)).clip(upper=3.0)

    raw  = np.sign(mom) * scale

    # market distress: use supplied market_ser or own returns
    bench = market_ser.reindex(df.index) if market_ser is not None else c.pct_change()
    mkt_12m = bench.rolling(252).sum()
    in_distress = (mkt_12m < down_thresh) & (rv > rv.rolling(252).quantile(0.75))
    raw[in_distress] *= damp

    return _lag(_pos(raw.values, df.index)).rename("vol_scaled_mom")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CONFIGS  — parameter grid
# ══════════════════════════════════════════════════════════════════════════════

def build_configs() -> list[tuple]:
    """
    Returns list of (name, fn, params_dict, category).
    name = "family__key1_val1__key2_val2"
    Total configs prints at end of __main__.
    """
    configs: list[tuple] = []

    def add(cat, fn, **params):
        label = "__".join(f"{k}{v}" for k, v in params.items())
        name  = f"{fn.__name__}__{label}" if label else fn.__name__
        configs.append((name, fn, params, cat))

    # ── TREND ────────────────────────────────────────────────────────────────
    for fast, slow, kind in [
        (5,20,"sma"),(10,30,"sma"),(20,50,"sma"),(50,100,"sma"),
        (50,200,"sma"),(100,200,"sma"),(10,50,"sma"),(20,100,"sma"),
        (5,20,"ema"),(12,26,"ema"),(10,50,"ema"),(20,100,"ema"),
    ]:
        add("trend", ma_crossover, fast=fast, slow=slow, kind=kind)

    for p in [21, 42, 63, 126, 252]:
        add("trend", ts_momentum, period=p)

    for p in [5, 10, 20, 30, 60]:
        add("trend", roc_momentum, period=p)

    for fast, slow, sig in [(12,26,9),(8,17,9),(5,35,5),(3,10,16)]:
        add("trend", macd_signal, fast=fast, slow=slow, signal=sig)

    for p in [10, 20, 30, 40, 55]:
        add("trend", donchian_breakout, period=p)

    for p, s in [(20,1.5),(20,2.0),(20,2.5),(30,2.0),(50,2.0)]:
        add("trend", bb_breakout, period=p, std=s)

    for p, m in [(7,2.0),(7,3.0),(10,3.0),(14,2.0),(14,3.0)]:
        add("trend", supertrend, period=p, mult=m)

    for step, maf in [(0.01,0.1),(0.02,0.2),(0.03,0.2),(0.05,0.2)]:
        add("trend", parabolic_sar, step=step, max_af=maf)

    for p, th in [(14,20),(14,25),(14,30),(21,20),(21,25)]:
        add("trend", adx_trend, period=p, threshold=th)

    for tk, kj, sb in [(9,26,52),(7,22,44)]:
        add("trend", ichimoku, tenkan=tk, kijun=kj, senkou_b=sb)

    for p in [10, 20, 30, 50]:
        add("trend", linreg_slope, period=p)

    for p in [14, 25, 50]:
        add("trend", aroon, period=p)

    for p in [10, 14, 21]:
        add("trend", vortex, period=p)

    for p in [9, 12, 18]:
        add("trend", trix, period=p)

    for p in [14, 20, 30, 50]:
        add("trend", hull_ma, period=p)

    for p in [5, 10, 20]:
        add("trend", kama, period=p)

    for e, x in [(20,10),(55,20),(10,5)]:
        add("trend", turtle, entry=e, exit_p=x)

    for p in [63, 126, 252]:
        add("trend", dual_momentum, period=p)

    for p in [13, 21, 34]:
        add("trend", elder_ray, period=p)

    # ── MEAN REVERSION ───────────────────────────────────────────────────────
    for p, lo, hi in [
        (7,30,70),(7,25,75),(14,30,70),(14,25,75),
        (14,35,65),(21,30,70),(21,25,75),(28,30,70),(30,30,70),(30,25,75),
    ]:
        add("meanrev", rsi_reversion, period=p, lo=lo, hi=hi)

    for p, s in [(20,1.5),(20,2.0),(20,2.5),(30,2.0),(30,2.5),(50,2.0)]:
        add("meanrev", bb_reversion, period=p, std=s)

    for p, th in [(10,1.0),(20,1.5),(20,2.0),(30,1.5),(30,2.0),(60,2.0)]:
        add("meanrev", zscore_reversion, period=p, threshold=th)

    for k, d, lo, hi in [(5,3,20,80),(14,3,20,80),(14,3,30,70),(14,5,20,80),(21,3,20,80)]:
        add("meanrev", stochastic, k=k, d=d, lo=lo, hi=hi)

    for p, th in [(14,100),(14,150),(20,100),(20,150),(30,100)]:
        add("meanrev", cci_reversion, period=p, threshold=th)

    for p, lo, hi in [(14,-80,-20),(14,-90,-10),(21,-80,-20),(28,-80,-20)]:
        add("meanrev", williams_r, period=p, lo=lo, hi=hi)

    for p, m in [(20,1.5),(20,2.0),(20,2.5),(30,2.0)]:
        add("meanrev", keltner_reversion, period=p, mult=m)

    for p, z in [(20,1.0),(20,1.5),(20,2.0)]:
        add("meanrev", vwap_reversion, period=p, z_thresh=z)

    for p, s, lo, hi in [(20,2.0,0.05,0.95),(20,2.0,0.1,0.9),(20,2.0,0.2,0.8),(30,2.0,0.2,0.8)]:
        add("meanrev", percent_b, period=p, std=s, lo=lo, hi=hi)

    for rp, sp, rnk in [(3,2,100),(2,3,100)]:
        add("meanrev", connors_rsi, rsi_period=rp, streak_period=sp, rank_period=rnk)

    for p1, p2, p3 in [(7,14,28),(4,8,16)]:
        add("meanrev", ultimate_osc, p1=p1, p2=p2, p3=p3)

    for th in [0.01, 0.015, 0.02]:
        add("meanrev", gap_fade, threshold=th)

    # ── VOLUME ───────────────────────────────────────────────────────────────
    for p in [10, 20, 50]:
        add("volume", obv_trend, period=p)

    for p in [10, 20, 21]:
        add("volume", chaikin_mf, period=p)

    for p in [10, 14, 21]:
        add("volume", mfi, period=p)

    for p, m in [(10,1.5),(20,2.0),(20,3.0)]:
        add("volume", volume_surge, period=p, mult=m)

    for p in [2, 13, 20]:
        add("volume", force_index, period=p)

    for f, s in [(3,10),(3,12),(5,12)]:
        add("volume", chaikin_osc, fast=f, slow=s)

    # ── VOLATILITY ───────────────────────────────────────────────────────────
    for p, m in [(10,1.5),(14,1.5),(14,2.0),(20,2.0),(20,2.5)]:
        add("volatility", atr_breakout, period=p, mult=m)

    for p, m in [(10,0.5),(20,1.0),(20,1.5),(30,1.0)]:
        add("volatility", vol_breakout, period=p, mult=m)

    for bp, kp, bs, km in [(20,20,2.0,1.5),(20,20,2.0,1.0),(30,30,2.0,1.5)]:
        add("volatility", squeeze_breakout, bb_period=bp, kc_period=kp, bb_std=bs, kc_mult=km)

    # ── PATTERN ──────────────────────────────────────────────────────────────
    add("pattern", engulfing)
    add("pattern", three_bar_reversal)
    for p in [5, 10, 20]:
        add("pattern", higher_highs_lows, period=p)
    for p in [5, 10, 20]:
        add("pattern", pivot_bounce, period=p)

    # ── COMPOSITE ────────────────────────────────────────────────────────────
    for fast, slow, sig, rp, rlo, rhi in [
        (12,26,9,14,40,60),(12,26,9,14,45,55),(8,21,5,14,40,60),(5,35,5,14,40,60),
    ]:
        add("composite", macd_rsi_combo,
            fast=fast, slow=slow, signal=sig, rsi_period=rp, rsi_lo=rlo, rsi_hi=rhi)

    for tp, op, olo, ohi in [(20,14,40,60),(30,14,40,60),(50,14,35,65),(20,7,30,70)]:
        add("composite", triple_screen,
            trend_period=tp, osc_period=op, osc_lo=olo, osc_hi=ohi)

    for p, m in [(14,2.0),(22,3.0),(22,2.0),(10,2.0)]:
        add("composite", chandelier_exit, period=p, mult=m)

    # ── HIGH-52 MOMENTUM (George & Hwang 2004) ───────────────────────────────
    for period, threshold in [(252, 0.70), (252, 0.75), (252, 0.80),
                               (126, 0.70), (126, 0.75)]:
        add("trend", high52_momentum, period=period, threshold=threshold)

    # ── SHORT-TERM REVERSAL (Jegadeesh 1990) ─────────────────────────────────
    for p in [5, 10, 15, 20]:
        add("meanrev", short_reversal, period=p)

    # ── LOW-VOL REGIME (Ang et al 2006) ──────────────────────────────────────
    for period, ma in [(20, 60), (20, 120), (60, 120),
                       (20, 252), (60, 252), (120, 252)]:
        add("volatility", low_vol_regime, period=period, ma_period=ma)

    # ── NR BREAKOUT (Crabel 1990) ─────────────────────────────────────────────
    for nr, ex in [(4, 3), (4, 5), (7, 3), (7, 5), (10, 5), (10, 10)]:
        add("pattern", nr_breakout, nr_period=nr, exit_bars=ex)

    # ── OVERNIGHT GAP MOMENTUM (Lou, Polk & Skouras 2019) ───────────────────
    for p, t in [(1, 0.001), (1, 0.003), (3, 0.001),
                 (3, 0.003), (5, 0.001), (5, 0.003)]:
        add("volume", overnight_gap, period=p, threshold=t)

    # ── COPPOCK CURVE (Coppock 1962) ──────────────────────────────────────────
    for r1, r2, wp in [(252, 189, 63), (189, 126, 42), (126, 100, 21)]:
        add("trend", coppock_curve, roc1=r1, roc2=r2, wma_period=wp)

    # ── BLL FILTER RULE (Brock, Lakonishok, LeBaron 1992) ────────────────────
    for pct, hold in [(0.005, 0), (0.01, 0), (0.02, 0),
                      (0.005, 10), (0.01, 10), (0.01, 20), (0.02, 10)]:
        add("trend", bll_filter, pct=pct, hold=hold)

    # ── FROG-IN-THE-PAN (Da, Gurun, Warachka 2014) ───────────────────────────
    for lb, sk, idc in [(252, 21, 0.50), (252, 21, 0.45), (252, 0, 0.50),
                        (126, 21, 0.50), (126, 21, 0.45)]:
        add("trend", frog_in_pan, lookback=lb, skip=sk, id_cutoff=idc)

    # ── VOLUME-MOMENTUM (Lee & Swaminathan 2000) ──────────────────────────────
    for mp, vp, sk, vw in [(126, 126, 21, 0.5), (126, 126, 0, 0.5),
                            (252, 126, 21, 0.5), (63, 63, 0, 0.3),
                            (126, 126, 21, 0.3)]:
        add("composite", vol_mom, mom_period=mp, vol_period=vp, skip=sk, vol_weight=vw)

    # ── SEASONALITY (Heston & Sadka 2008) ────────────────────────────────────
    for ny in [2, 3, 5]:
        add("pattern", seasonality, n_years=ny)

    # ── RESIDUAL MOMENTUM (Blitz, Huij, Martens 2011) ────────────────────────
    for lb, sk in [(252, 21), (252, 0), (126, 21)]:
        add("trend", residual_mom, lookback=lb, skip=sk)

    # ── VOL-SCALED MOMENTUM (Daniel & Moskowitz 2016) ────────────────────────
    for rv, vt, damp in [(21, 0.12, 0.5), (21, 0.08, 0.5),
                          (63, 0.12, 0.5), (21, 0.12, 0.0),
                          (21, 0.15, 0.5)]:
        add("trend", vol_scaled_mom, rv_window=rv, vol_target=vt, damp=damp)

    # ── DEMA CROSSOVER ────────────────────────────────────────────────────────
    for fast, slow in [(5,20),(10,40),(12,26),(20,50),(10,30)]:
        add("trend", dema_crossover, fast=fast, slow=slow)

    # ── TEMA CROSSOVER ────────────────────────────────────────────────────────
    for fast, slow in [(5,20),(10,40),(12,26),(20,50),(10,30)]:
        add("trend", tema_crossover, fast=fast, slow=slow)

    # ── PRICE ACCELERATION ────────────────────────────────────────────────────
    for period, smooth in [(10,3),(20,5),(20,3),(30,5),(40,10),(10,5)]:
        add("trend", price_acceleration, period=period, smooth=smooth)

    # ── TURN OF MONTH (Ariel 1987) ────────────────────────────────────────────
    for db, da in [(3,3),(4,4),(5,5),(3,5)]:
        add("pattern", turn_of_month, days_before=db, days_after=da)

    # ── TREND INTENSITY INDEX ─────────────────────────────────────────────────
    for p in [20, 30, 50, 63]:
        add("trend", trend_intensity, period=p)

    # ── MA-REGIME MOMENTUM (Faber 2007 / Antonacci 2014) ─────────────────────
    for regime, mom, skip in [(200,126,21),(200,252,21),(200,63,0),
                               (100,126,21),(50,63,0)]:
        add("trend", ma_regime_momentum, regime_ma=regime, mom_period=mom, skip=skip)

    # ── FISHER TRANSFORM (Ehlers) ─────────────────────────────────────────────
    for p in [5, 10, 20, 30]:
        add("trend", fisher_transform, period=p)

    # ── RELATIVE VIGOR INDEX ──────────────────────────────────────────────────
    for p in [10, 14, 20]:
        add("trend", relative_vigor, period=p)

    # ── CHANDE MOMENTUM OSCILLATOR ────────────────────────────────────────────
    for p, th in [(14, 0.), (20, 0.), (20, 10.), (30, 0.), (9, 0.)]:
        add("trend", chande_momentum, period=p, threshold=th)

    # ── DEMARKER ──────────────────────────────────────────────────────────────
    for p, hi, lo in [(14, 0.7, 0.3), (14, 0.8, 0.2), (21, 0.7, 0.3), (9, 0.7, 0.3)]:
        add("meanrev", demarker, period=p, hi_thresh=hi, lo_thresh=lo)

    # ── STOCHASTIC RSI ────────────────────────────────────────────────────────
    for rp, sp, lo, hi in [(14,14,20.,80.), (14,14,10.,90.),
                            (7,14,20.,80.), (21,21,20.,80.)]:
        add("meanrev", stoch_rsi, rsi_period=rp, stoch_period=sp, lo=lo, hi=hi)

    # ── HURST REGIME ──────────────────────────────────────────────────────────
    for p, tt, mt in [(100, 0.55, 0.45), (60, 0.55, 0.45), (100, 0.58, 0.42)]:
        add("trend", hurst_regime, period=p, trend_thresh=tt, mr_thresh=mt)

    # ── PSYCHOLOGICAL LINE ────────────────────────────────────────────────────
    for p, hi, lo in [(12, 75., 25.), (12, 65., 35.), (20, 70., 30.)]:
        add("meanrev", psych_line, period=p, hi_thresh=hi, lo_thresh=lo)

    # ── KAUFMAN ER REGIME ────────────────────────────────────────────────────
    for ep, mp, th in [(10,20,0.40),(10,20,0.30),(14,20,0.40),(20,40,0.40),(10,10,0.40)]:
        add("trend", er_regime, er_period=ep, mom_period=mp, er_thresh=th)

    # ── MCGINLEY DYNAMIC ─────────────────────────────────────────────────────
    for fast, slow in [(5,20),(10,40),(12,26),(5,14)]:
        add("trend", mcginley_dynamic, fast=fast, slow=slow)

    # ── CCI TREND-FOLLOW ─────────────────────────────────────────────────────
    for p, th in [(14,50.),(20,50.),(20,100.),(30,50.),(14,100.)]:
        add("trend", cci_trend, period=p, threshold=th)

    # ── VOLUME ROC ────────────────────────────────────────────────────────────
    for p, sp in [(10,5),(14,10),(20,10),(5,3)]:
        add("volume", volume_roc, period=p, signal_period=sp)

    # ── HV RATIO REGIME ──────────────────────────────────────────────────────
    for fp, sp, hi, lo in [(10,100,1.5,0.7),(10,60,1.5,0.7),(21,100,1.5,0.7),(10,100,2.0,0.5)]:
        add("volatility", hv_ratio, fast_period=fp, slow_period=sp,
            thresh_hi=hi, thresh_lo=lo)

    # ── ZSCORE + TREND FILTER ────────────────────────────────────────────────
    for zp, tp, zt in [(20,50,1.5),(20,50,1.0),(20,100,1.5),(30,100,2.0)]:
        add("meanrev", zscore_trend, z_period=zp, trend_period=tp, z_thresh=zt)

    # ── CMO-ADAPTIVE FISHER ───────────────────────────────────────────────────
    for fp, cp, th in [(10,14,0.30),(10,14,0.20),(14,20,0.30),(8,14,0.20)]:
        add("trend", cmo_fisher, fisher_period=fp, cmo_period=cp, score_thresh=th)

    # ── RVI DIVERGENCE ────────────────────────────────────────────────────────
    for rp, lb in [(10,5),(10,8),(14,5)]:
        add("meanrev", rvi_divergence, rvi_period=rp, lookback=lb)

    # ── ADAPTIVE RSI ──────────────────────────────────────────────────────────
    for bp, fn, sn in [(14,2,30),(14,2,60),(9,2,30),(21,2,30)]:
        add("meanrev", adaptive_rsi, base_period=bp, fast_n=fn, slow_n=sn)

    # ── ELDER IMPULSE SYSTEM ──────────────────────────────────────────────────
    for ep in [8, 13, 20]:
        add("composite", elder_impulse, ema_period=ep)

    # ── KLINGER VOLUME OSCILLATOR ─────────────────────────────────────────────
    for fast, slow, sig in [(34,55,13),(21,34,9),(55,89,13)]:
        add("volume", klinger_osc, fast=fast, slow=slow, signal=sig)

    # ── PRICE OSCILLATOR (PPO) ────────────────────────────────────────────────
    for f, s, sg in [(10,30,9),(12,26,9),(5,20,5),(20,50,9)]:
        add("trend", price_oscillator, fast=f, slow=s, signal=sg)

    # ── BAZ MULTI-EWMA TREND (Baz et al 2015 / AQR) ──────────────────────────
    for ewmstd_span in [42, 63, 126]:
        add("trend", baz_trend, ewmstd_span=ewmstd_span)

    # ── INTERMEDIATE MOMENTUM (Novy-Marx 2012) ────────────────────────────────
    for start, end in [(252,126),(189,105),(168,84),(252,63)]:
        add("trend", intermediate_mom, start=start, end=end)

    # ── MOMENTUM CRASH PROTECTION (Daniel & Moskowitz 2016) ──────────────────
    for bear_lb, bear_th, damp in [(504,-0.20,0.0),(504,-0.20,0.25),
                                    (252,-0.20,0.0),(504,-0.25,0.0)]:
        add("trend", mom_crash_protect,
            bear_lookback=bear_lb, bear_thresh=bear_th, damp=damp)

    # ── CONSISTENT MOMENTUM / QMOM (Alpha Architect) ─────────────────────────
    for thresh in [0.50, 0.58, 0.67]:
        add("trend", consistent_mom, consistency_thresh=thresh)

    # ── SIGNED RANGE MOMENTUM (Llorente et al 2002) ───────────────────────────
    for period, threshold in [(10,0.10),(21,0.10),(21,0.15),(42,0.10),(63,0.10)]:
        add("trend", signed_range_mom, period=period, threshold=threshold)

    # ── MULTI-LOOKBACK TSMOM (Moskowitz, Ooi, Pedersen 2012) ─────────────────
    add("trend", multi_tsmom)  # default: equal-weight 1/3/6/12m
    add("trend", multi_tsmom, lookbacks=(63, 126, 252))
    add("trend", multi_tsmom, lookbacks=(21, 63, 252))
    add("trend", multi_tsmom, lookbacks=(21, 63, 126, 252),
        weights=(0.4, 0.3, 0.2, 0.1))
    add("trend", multi_tsmom, lookbacks=(21, 63, 126, 252),
        weights=(0.1, 0.2, 0.3, 0.4))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE API
# ══════════════════════════════════════════════════════════════════════════════

def get_positions(df: pd.DataFrame, name: str, fn, params: dict) -> pd.Series:
    return fn(df, **params)


def run_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """Run every config against df. Returns DataFrame[config_name → position series]."""
    out = {}
    for name, fn, params, _ in build_configs():
        try:
            out[name] = fn(df, **params)
        except Exception as exc:
            print(f"  [warn] {name}: {exc}")
            out[name] = pd.Series(0., index=df.index, name=name)
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from collections import Counter

    print("=" * 62)
    print("LAYER 1 — Data + Strategy Library")
    print("=" * 62)

    configs = build_configs()
    cats    = Counter(c for _, _, _, c in configs)

    print("\nStrategy registry:")
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<12} {n:>3} configs")
    print(f"  {'TOTAL':<12} {len(configs):>3} configs")
    print(f"\n  × {len(TICKERS)} assets = {len(configs)*len(TICKERS):,} backtests")

    print("\n[1/2] Downloading market data …\n")
    universe = download_data()

    print(f"\n[2/2] Smoke-testing all configs on SPY …")
    spy = universe.get("SPY")
    if spy is None:
        raise RuntimeError("SPY not in universe")

    positions = run_all_strategies(spy)
    vals = positions.values.ravel()
    vals = vals[~np.isnan(vals)]
    assert set(np.unique(vals)).issubset({-1.,0.,1.}), "out-of-range positions"
    print(f"  shape   : {positions.shape}")
    print(f"  val dist: {dict(zip(*np.unique(vals, return_counts=True)))}")
    print(f"\n  Date range : {spy.index[0].date()} → {spy.index[-1].date()}")
    print(f"  Universe   : {len(universe)} assets")
    print("\nImport in later layers:")
    print("  from layer1 import download_data, build_configs, run_all_strategies")
