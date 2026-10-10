"""
Chart Scanner
=============
Upload a chart screenshot. The app reads the green and red candles from the
image, runs 20 widely taught technical-analysis strategies on them, and issues
a BUY, SELL or HOLD signal with a confidence score.

  * Each strategy votes BUY, SELL or NEUTRAL on the latest candles.
  * Confidence = share of the 20 strategies that agree.
  * A BUY or SELL is issued only when confidence reaches MIN_CONFIDENCE.
    Opposing votes count against the total, so conflicts lower confidence.
  * Each strategy is backtested on the candles in the screenshot.

No price scale is needed. The strategies only use candle shape (crossovers,
ratios and ranges), so the signal is the same at any price level. Stop and
target are expressed in ATR units, the average candle range.

Educational use only. This is not financial advice.
"""

import io

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from PIL import Image


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MIN_CONFIDENCE = 70      # % of strategies that must agree before a signal is issued
VOTE_LOOKBACK = 5        # a strategy's latest trigger within this many candles still counts as its vote
ATR_PERIOD = 14
STOP_ATR = 1.5           # stop distance, in ATR units
TARGET_ATR = 2.5         # target distance, in ATR units
BACKTEST_HORIZON = 50    # candles a simulated trade may stay open
MIN_CANDLES = 60         # strategies need this many candles to warm up
MAX_UPLOAD_MB = 10


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def sma(s, n):
    return s.rolling(n).mean()


def atr(df, n=ATR_PERIOD):
    prev_close = df["close"].shift()
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def stochastic(df, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    pct_k = 100 * (df["close"] - lo) / (hi - lo)
    return pct_k, pct_k.rolling(d).mean()


def williams_r(df, n=14):
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    return -100 * (hi - df["close"]) / (hi - lo)


def cci(df, n=20):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean_dev = typical.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - typical.rolling(n).mean()) / (0.015 * mean_dev)


def bollinger(close, n=20, k=2):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def keltner(df, n=20, mult=2.0):
    mid = ema(df["close"], n)
    band = mult * atr(df, 10)
    return mid, mid + band, mid - band


def adx(df, n=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    smoothed_atr = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / smoothed_atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / smoothed_atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def ichimoku(df):
    hi, lo = df["high"], df["low"]
    tenkan = (hi.rolling(9).max() + lo.rolling(9).min()) / 2
    kijun = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    return tenkan, kijun, span_a, span_b


def supertrend(df, n=10, mult=3.0):
    band = mult * atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    basic_up, basic_dn = hl2 + band, hl2 - band
    close = df["close"].to_numpy()
    up, dn = basic_up.copy(), basic_dn.copy()
    trend = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        up[i] = basic_up[i] if (basic_up[i] < up[i - 1] or close[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = basic_dn[i] if (basic_dn[i] > dn[i - 1] or close[i - 1] < dn[i - 1]) else dn[i - 1]
        if trend[i - 1] == 1 and close[i] < dn[i]:
            trend[i] = -1
        elif trend[i - 1] == -1 and close[i] > up[i]:
            trend[i] = 1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)


def heikin_ashi(df):
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.empty(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    hc = ha_close.to_numpy()
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + hc[i - 1]) / 2
    return pd.Series(ha_open, index=df.index), ha_close


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def crossed_above(a, b):
    b = pd.Series(b, index=a.index) if np.isscalar(b) else b
    return (a > b) & (a.shift() <= b.shift())


def crossed_below(a, b):
    b = pd.Series(b, index=a.index) if np.isscalar(b) else b
    return (a < b) & (a.shift() >= b.shift())


def _signal(index, buy, sell):
    """Turn two boolean masks into +1 (buy), -1 (sell) or 0 for every candle."""
    b = buy.fillna(False).astype(bool).to_numpy()
    s = sell.fillna(False).astype(bool).to_numpy()
    return pd.Series(np.select([b, s], [1, -1], default=0), index=index)


# ---------------------------------------------------------------------------
# The 20 strategies. Each returns +1 (buy), -1 (sell) or 0 for every candle.
# ---------------------------------------------------------------------------

# Trend following
def s_ema_cross(df):
    fast, slow = ema(df["close"], 20), ema(df["close"], 50)
    return _signal(df.index, crossed_above(fast, slow), crossed_below(fast, slow))


def s_golden_cross(df):
    fast, slow = sma(df["close"], 50), sma(df["close"], 200)
    return _signal(df.index, crossed_above(fast, slow), crossed_below(fast, slow))


def s_macd(df):
    line, signal_line = macd(df["close"])
    return _signal(df.index, crossed_above(line, signal_line), crossed_below(line, signal_line))


def s_adx_di(df):
    adx_v, plus_di, minus_di = adx(df)
    trending = adx_v > 25
    return _signal(
        df.index,
        crossed_above(plus_di, minus_di) & trending,
        crossed_below(plus_di, minus_di) & trending,
    )


def s_ichimoku(df):
    tenkan, kijun, span_a, span_b = ichimoku(df)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)
    close = df["close"]
    return _signal(
        df.index,
        crossed_above(tenkan, kijun) & (close > cloud_top),
        crossed_below(tenkan, kijun) & (close < cloud_bottom),
    )


def s_supertrend(df):
    trend = supertrend(df)
    return _signal(df.index, crossed_above(trend, 0), crossed_below(trend, 0))


def s_donchian(df, n=20):
    high_n = df["high"].rolling(n).max().shift()
    low_n = df["low"].rolling(n).min().shift()
    return _signal(df.index, df["close"] > high_n, df["close"] < low_n)


def s_heikin_ashi(df):
    ha_open, ha_close = heikin_ashi(df)
    bull = (ha_close > ha_open).astype(float)
    bear = 1 - bull
    bull_run, bear_run = bull.rolling(3).sum(), bear.rolling(3).sum()
    return _signal(
        df.index,
        (bull_run == 3) & (bull_run.shift() < 3),
        (bear_run == 3) & (bear_run.shift() < 3),
    )


# Momentum
def s_rsi_reversal(df):
    r = rsi(df["close"])
    return _signal(df.index, crossed_above(r, 30), crossed_below(r, 70))


def s_rsi_centerline(df):
    r = rsi(df["close"])
    return _signal(df.index, crossed_above(r, 50), crossed_below(r, 50))


def s_stochastic(df):
    k, d = stochastic(df)
    return _signal(df.index, crossed_above(k, d) & (k < 20), crossed_below(k, d) & (k > 80))


def s_williams_r(df):
    wr = williams_r(df)
    return _signal(df.index, crossed_above(wr, -80), crossed_below(wr, -20))


def s_cci(df):
    value = cci(df)
    return _signal(df.index, crossed_above(value, -100), crossed_below(value, 100))


def s_momentum(df, n=12):
    """Close minus the close n bars ago, in ATR units. Crossing zero means direction changed."""
    change = (df["close"] - df["close"].shift(n)) / atr(df)
    return _signal(df.index, crossed_above(change, 0), crossed_below(change, 0))


# Volatility
def s_bollinger_bounce(df):
    close = df["close"]
    _, upper, lower = bollinger(close)
    buy = (close.shift() < lower.shift()) & (close > lower)   # closed back above the lower band
    sell = (close.shift() > upper.shift()) & (close < upper)  # closed back below the upper band
    return _signal(df.index, buy, sell)


def s_keltner_breakout(df):
    _, upper, lower = keltner(df)
    close = df["close"]
    return _signal(df.index, crossed_above(close, upper), crossed_below(close, lower))


def s_bb_squeeze(df):
    close = df["close"]
    _, upper, lower = bollinger(close)
    width = upper - lower  # absolute width, so the result does not depend on the price level
    squeezed = (width <= width.rolling(120).quantile(0.2)).shift().fillna(False).astype(bool)
    return _signal(df.index, squeezed & (close > upper), squeezed & (close < lower))


# Market structure
def s_fibonacci(df, swing=30):
    close = df["close"]
    hi = df["high"].rolling(swing).max()
    lo = df["low"].rolling(swing).min()
    rng = hi - lo
    band = 0.25 * atr(df)
    ema50, ema200 = ema(close, 50), ema(close, 200)
    fib_up = hi - 0.618 * rng  # 61.8% retracement of the swing, in an uptrend
    fib_dn = lo + 0.618 * rng  # 61.8% retracement of the swing, in a downtrend
    buy = (ema50 > ema200) & (df["low"] <= fib_up + band) & (close > fib_up) & (close > df["open"])
    sell = (ema50 < ema200) & (df["high"] >= fib_dn - band) & (close < fib_dn) & (close < df["open"])
    return _signal(df.index, buy, sell)


# Candlestick price action
def s_engulfing(df):
    o, c = df["open"], df["close"]
    po, pc = o.shift(), c.shift()
    trend = ema(c, 50)
    bullish = (pc < po) & (c > o) & (o <= pc) & (c >= po) & (pc < trend.shift())
    bearish = (pc > po) & (c < o) & (o >= pc) & (c <= po) & (pc > trend.shift())
    return _signal(df.index, bullish, bearish)


def s_pin_bar(df):
    o, h, lo, c = df["open"], df["high"], df["low"], df["close"]
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    body = top - bottom
    rng = (h - lo).replace(0, np.nan)
    upper_wick, lower_wick = h - top, bottom - lo
    small_body = body <= 0.3 * rng
    trend = ema(c, 20)
    bullish = small_body & (lower_wick >= 2 * body) & (lower_wick >= 0.6 * rng) & (c < trend)
    bearish = small_body & (upper_wick >= 2 * body) & (upper_wick >= 0.6 * rng) & (c > trend)
    return _signal(df.index, bullish, bearish)


STRATEGIES = [
    ("EMA 20/50 crossover", "Trend", s_ema_cross),
    ("SMA 50/200 golden cross", "Trend", s_golden_cross),
    ("MACD signal-line cross", "Trend", s_macd),
    ("ADX + DI cross", "Trend", s_adx_di),
    ("Ichimoku TK cross", "Trend", s_ichimoku),
    ("Supertrend flip", "Trend", s_supertrend),
    ("Donchian 20 breakout", "Trend", s_donchian),
    ("Heikin-Ashi 3-candle flip", "Trend", s_heikin_ashi),
    ("RSI 30/70 reversal", "Momentum", s_rsi_reversal),
    ("RSI 50 centerline cross", "Momentum", s_rsi_centerline),
    ("Stochastic cross in extremes", "Momentum", s_stochastic),
    ("Williams %R cross", "Momentum", s_williams_r),
    ("CCI +/-100 cross", "Momentum", s_cci),
    ("12-bar momentum zero cross", "Momentum", s_momentum),
    ("Bollinger band bounce", "Volatility", s_bollinger_bounce),
    ("Keltner channel breakout", "Volatility", s_keltner_breakout),
    ("Bollinger squeeze breakout", "Volatility", s_bb_squeeze),
    ("Fibonacci 61.8% pullback", "Structure", s_fibonacci),
    ("Engulfing candle", "Price action", s_engulfing),
    ("Pin bar rejection", "Price action", s_pin_bar),
]


# ---------------------------------------------------------------------------
# Reading candles from a screenshot
# ---------------------------------------------------------------------------
def _grow(mask, seed):
    """Return the (top, bottom) rows of the contiguous True block that contains seed."""
    top = bottom = seed
    while top > 0 and mask[top - 1]:
        top -= 1
    while bottom < len(mask) - 1 and mask[bottom + 1]:
        bottom += 1
    return top, bottom


def _candle_grid(strong_counts):
    """
    Find the spacing between candles and the x position of the first candle centre.

    Each run of coloured columns is one candle, so its middle is that candle's centre.
    The centres sit on a regular grid. A rough spacing comes from the typical gap
    between neighbouring centres, then a straight line is fitted through each centre
    against its candle index. Runs that merge two candles are outliers and are dropped.
    """
    mask = strong_counts > 0
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    centres = (starts + ends) / 2.0
    if len(centres) < 3:
        raise ValueError("Could not find enough candles in this image.")

    diffs = np.diff(centres)
    typical = np.median(diffs)
    single = diffs[(diffs > 0.75 * typical) & (diffs < 1.25 * typical)]  # gaps of exactly one candle
    if single.size < 3:
        raise ValueError("Candles touch each other in this screenshot. Use fewer bars or a wider chart.")
    rough = single.mean()  # sub-pixel: whole-pixel gaps average out to the true spacing
    index = np.round((centres - centres[0]) / rough)
    keep = np.ones(len(centres), dtype=bool)
    for _ in range(4):
        design = np.c_[np.ones(keep.sum()), index[keep]]
        (offset, period), *_ = np.linalg.lstsq(design, centres[keep], rcond=None)
        keep = np.abs(centres - (offset + period * index)) < 0.25 * period

    widths = ends - starts + 1
    if np.median(widths) >= 0.95 * period:
        raise ValueError("Candles touch each other in this screenshot. Use fewer bars or a wider chart.")
    return float(period), float(offset % period)


def _longest_run(mask):
    """Longest horizontal run of True pixels in each row of a small mask."""
    run = np.zeros(mask.shape[0], dtype=int)
    best = np.zeros(mask.shape[0], dtype=int)
    for x in range(mask.shape[1]):
        run = (run + 1) * mask[:, x]
        best = np.maximum(best, run)
    return best


def read_candles(image_bytes):
    """
    Turn a chart screenshot into OHLC candles, left to right.

    Candles are assumed to sit at a regular spacing, as on standard price charts.
    Each candle is read from its own slot, so touching candles are kept apart.
    Prices are relative: 1.0 at the top edge of the image and 0.0 at the bottom.
    Strategies only use candle shape, so the absolute price level is not needed.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("That file is not an image the scanner can read. Use PNG, JPG or WEBP.")

    hsv = np.asarray(img.convert("HSV")).astype(int)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]   # Pillow HSV: each channel 0-255
    vivid = (sat >= 90) & (val >= 60)                       # ignores grey, white and dark backgrounds
    green = vivid & (hue >= 60) & (hue <= 140)
    red = vivid & ((hue <= 15) | (hue >= 235))
    colored = green | red

    height, width = colored.shape
    col_counts = colored.sum(axis=0)
    strong_counts = np.where(col_counts >= 3, col_counts, 0)  # wicks and bodies; ignores 1-2 px dotted lines
    if strong_counts.max() == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    period, phase = _candle_grid(strong_counts)

    def to_price(y):
        return 1.0 - y / (height - 1)

    candles = []
    for centre in np.arange(phase, width, period):
        # Only columns whose centre lies inside this candle's slot. Rounding outwards
        # would pick up the edge of a neighbouring candle.
        x0 = max(0, int(np.ceil(centre - period / 2 + 1e-9)))
        x1 = min(width, int(np.floor(centre + period / 2 - 1e-9)) + 1)
        if x1 <= x0:
            continue
        g = green[:, x0:x1]
        r = red[:, x0:x1]
        both = g | r
        if both.sum(axis=0).max() < 3:  # no wick or body in this slot
            continue

        # The body is the block of wide rows around the widest row. The wick is the
        # contiguous run of coloured rows above and below it, so stray pixels elsewhere
        # in the slot (such as a dotted price line) do not stretch the candle.
        runs = _longest_run(both)
        peak_row = int(np.argmax(runs))
        peak = runs[peak_row]
        body_top, body_bottom = _grow(runs >= max(min(2, peak), 0.75 * peak), peak_row)
        present = both.any(axis=1)
        high_y = _grow(present, body_top)[0]
        low_y = _grow(present, body_bottom)[1]

        body_green = g[body_top:body_bottom + 1].sum()
        body_red = r[body_top:body_bottom + 1].sum()
        if body_green >= body_red:
            open_y, close_y = body_bottom, body_top
        else:
            open_y, close_y = body_top, body_bottom

        candles.append({
            "open": to_price(open_y),
            "high": to_price(high_y),
            "low": to_price(low_y),
            "close": to_price(close_y),
        })

    return pd.DataFrame(candles, columns=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def latest_vote(signals):
    """A strategy's vote is its most recent trigger within VOTE_LOOKBACK candles."""
    recent = signals.iloc[-VOTE_LOOKBACK:]
    fired = recent[recent != 0]
    return int(fired.iloc[-1]) if len(fired) else 0


def backtest(df, signals, atr_series):
    """Enter at each signal's close; exit at stop or target in ATR units (stop wins ties)."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atrs = atr_series.to_numpy()
    sig = signals.to_numpy()
    wins = losses = 0

    for i in np.flatnonzero(sig):
        if np.isnan(atrs[i]):
            continue
        side, entry = sig[i], closes[i]
        stop = entry - side * STOP_ATR * atrs[i]
        target = entry + side * TARGET_ATR * atrs[i]
        for j in range(i + 1, min(i + 1 + BACKTEST_HORIZON, len(df))):
            stop_hit = lows[j] <= stop if side == 1 else highs[j] >= stop
            target_hit = highs[j] >= target if side == 1 else lows[j] <= target
            if stop_hit:
                losses += 1
                break
            if target_hit:
                wins += 1
                break

    trades = wins + losses
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": round(100 * wins / trades, 1) if trades else None,
    }


def analyze(df):
    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"Found {len(df)} candles. The scanner needs at least {MIN_CANDLES}. "
            "Use a screenshot with more bars."
        )

    atr_series = atr(df)
    rows = []
    for name, family, fn in STRATEGIES:
        signals = fn(df).fillna(0).astype(int)
        rows.append({
            "name": name,
            "category": family,
            "vote": latest_vote(signals),
            "backtest": backtest(df, signals, atr_series),
        })

    total = len(rows)
    buy_votes = sum(r["vote"] == 1 for r in rows)
    sell_votes = sum(r["vote"] == -1 for r in rows)
    neutral = total - buy_votes - sell_votes

    direction = 0 if buy_votes == sell_votes else (1 if buy_votes > sell_votes else -1)
    agreeing = max(buy_votes, sell_votes)
    confidence = round(100 * agreeing / total, 1)
    dissenting = sell_votes if direction == 1 else buy_votes

    if direction == 0:
        signal, reason = "HOLD", "Buy and sell votes are tied."
    elif confidence < MIN_CONFIDENCE:
        signal = "HOLD"
        reason = f"Only {agreeing} of {total} strategies agree. {MIN_CONFIDENCE}% is required."
    else:
        signal = "BUY" if direction == 1 else "SELL"
        reason = f"{agreeing} of {total} strategies agree on {signal}."
        if dissenting:
            reason += f" {dissenting} voted the other way."

    last = df.iloc[-1]
    return {
        "signal": signal,
        "reason": reason,
        "confidence": confidence,
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "neutral": neutral,
        "candles": len(df),
        "last_candle": "Bullish" if last["close"] >= last["open"] else "Bearish",
        "stop_atr": STOP_ATR,
        "target_atr": TARGET_ATR,
        "reward_risk": round(TARGET_ATR / STOP_ATR, 2),
        "strategies": rows,
    }


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.get("/")
def home():
    return PAGE


@app.post("/api/scan")
def scan():
    try:
        upload = request.files.get("image")
        if not upload or not upload.filename:
            raise ValueError("Upload a chart screenshot.")

        df = read_candles(upload.read())
        if df.empty:
            raise ValueError("No green or red candles found. Try a screenshot of the candles only.")

        return jsonify(analyze(df))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Scan failed")
        return jsonify(error="Scan failed. Try a clearer screenshot."), 500


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Chart Scanner</title>
<style>
:root{
  --bg:#070b14;--panel:#0d1424;--panel-2:#09101e;--line:#1a2539;--track:#17223a;
  --text:#e4ebf7;--muted:#7b889f;--accent:#58c4ff;--buy:#2bd98b;--sell:#ff4d6d;--hold:#ffc857;
}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);
  font:13px/1.5 Inter,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(700px 380px at 50% -8%,rgba(88,196,255,.13),transparent 62%),
             radial-gradient(560px 340px at 100% 100%,rgba(155,140,255,.10),transparent 62%)}
#bg{position:fixed;inset:0;width:100%;height:100%;z-index:0;pointer-events:none}
.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}

.top{position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;
  padding:12px 18px;border-bottom:1px solid var(--line);background:rgba(7,11,20,.72);backdrop-filter:blur(6px)}
.brand{display:flex;align-items:center;gap:10px;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase}
.logo{width:20px;height:20px}
.meta{font-size:11px;color:var(--muted);letter-spacing:.5px}

main{position:relative;z-index:1;max-width:720px;margin:0 auto;padding:22px 16px 36px}

.drop{position:relative;display:block;height:300px;cursor:pointer;outline:none;overflow:hidden;
  border:1.5px dashed rgba(88,196,255,.42);border-radius:18px;
  background:linear-gradient(180deg,rgba(13,20,36,.92),rgba(9,16,30,.92));
  transition:border-color .2s,box-shadow .2s,transform .2s}
.drop:hover,.drop:focus-visible,.drop.over{border-color:var(--accent);
  box-shadow:0 0 0 1px rgba(88,196,255,.22),0 0 30px rgba(88,196,255,.16)}
.drop.over{transform:scale(1.01)}
.drop input{display:none}
.drop-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px}
.plus{width:54px;height:54px;fill:none;stroke:var(--accent);stroke-width:2;stroke-linecap:round;
  transition:transform .3s ease}
.drop:hover .plus{transform:rotate(90deg)}
.drop-empty span{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#a9c9e6}

.drop-preview{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#05080f}
.drop-preview img{max-width:calc(100% - 24px);max-height:calc(100% - 24px);object-fit:contain;border-radius:10px}
.beam{position:absolute;left:0;right:0;top:0;height:2px;opacity:0;pointer-events:none;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);box-shadow:0 0 14px var(--accent)}
.drop.scanning .beam{opacity:1;animation:sweep 1.5s ease-in-out infinite}
@keyframes sweep{0%{top:0}50%{top:calc(100% - 2px)}100%{top:0}}

.hint{margin:10px 2px 0;font-size:11px;color:var(--muted)}
.status{min-height:18px;margin:8px 2px 0;font-size:12px;color:var(--muted)}
.status.error{color:var(--sell)}

.results{margin-top:18px;display:grid;gap:12px;animation:rise .35s ease-out}
@keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}
.hero{display:flex;justify-content:space-between;align-items:center;gap:16px}
.sig{font-size:18px;font-weight:700;letter-spacing:1.5px}
.sub{margin-top:4px;font-size:12px;color:var(--muted)}
.results.buy{--tone:var(--buy)}
.results.sell{--tone:var(--sell)}
.results.hold{--tone:var(--hold)}
.results .sig{color:var(--tone)}

.ring{position:relative;width:76px;height:76px;flex:none}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:6}
.ring .track{stroke:var(--track)}
.ring .arc{stroke:var(--tone);stroke-linecap:round;transition:stroke-dashoffset .9s ease}
.ring .val{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:600}

.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.fact{background:var(--panel-2);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.fact span{display:block;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:var(--muted)}
.fact b{display:block;margin-top:2px;font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:13px;font-weight:600}

.section{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px}
.section h3{margin:0;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);font-weight:600}
.section small{font-size:10px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.tile{background:var(--panel-2);border:1px solid var(--line);border-radius:12px;padding:11px 12px;animation:rise .4s both}
.tile-top{display:flex;justify-content:space-between;align-items:center}
.fam{font-size:10px;letter-spacing:1.1px;text-transform:uppercase;color:var(--muted)}
.chip{font-size:10px;font-weight:700;letter-spacing:1px;padding:2px 8px;border-radius:6px;border:1px solid}
.chip.buy{color:var(--buy);border-color:rgba(43,217,139,.4);background:rgba(43,217,139,.08)}
.chip.sell{color:var(--sell);border-color:rgba(255,77,109,.4);background:rgba(255,77,109,.08)}
.chip.flat{color:var(--muted);border-color:var(--line)}
.tile-name{margin:7px 0 9px;font-size:13px;font-weight:500}
.bt{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted)}
.bt-bar{flex:1;height:4px;border-radius:4px;background:var(--track);overflow:hidden}
.bt-bar span{display:block;height:100%;background:var(--accent)}
.bt .mono{color:var(--text)}

.foot{margin:22px 0 0;text-align:center;font-size:11px;color:var(--muted)}

@media(max-width:600px){
  .facts{grid-template-columns:repeat(2,1fr)}
  .grid{grid-template-columns:1fr}
  .drop{height:260px}
}
</style>
</head>
<body>
<canvas id="bg" aria-hidden="true"></canvas>

<header class="top">
  <div class="brand">
    <svg class="logo" viewBox="0 0 20 20" aria-hidden="true">
      <line x1="4" y1="1" x2="4" y2="19" stroke="#2bd98b" stroke-width="1.4"/>
      <rect x="2.5" y="5" width="3" height="8" fill="#2bd98b"/>
      <line x1="10" y1="0" x2="10" y2="18" stroke="#ff4d6d" stroke-width="1.4"/>
      <rect x="8.5" y="4" width="3" height="9" fill="#ff4d6d"/>
      <line x1="16" y1="2" x2="16" y2="20" stroke="#58c4ff" stroke-width="1.4"/>
      <rect x="14.5" y="7" width="3" height="7" fill="#58c4ff"/>
    </svg>
    <span>Chart Scanner</span>
  </div>
  <div class="meta">20 strategies</div>
</header>

<main>
  <label id="drop" class="drop" tabindex="0" role="button" aria-label="Upload chart">
    <input id="file" type="file" accept="image/png,image/jpeg,image/webp">
    <div id="drop-empty" class="drop-empty">
      <svg class="plus" viewBox="0 0 54 54" aria-hidden="true">
        <path d="M27 8v38M8 27h38"/>
      </svg>
      <span>Upload chart</span>
    </div>
    <div id="drop-preview" class="drop-preview" hidden>
      <img id="preview" alt="Uploaded chart">
      <div class="beam"></div>
    </div>
  </label>
  <p class="hint">Candles only. Leave out the price scale and volume panes.</p>
  <p id="status" class="status"></p>

  <section id="results" class="results" hidden></section>

  <p class="foot">Educational tool, not financial advice. Backtests are simplified and past results do not guarantee future ones.</p>
</main>

<script>
const $ = (id) => document.getElementById(id);
const drop = $("drop");
const fileInput = $("file");
const emptyBox = $("drop-empty");
const previewBox = $("drop-preview");
const preview = $("preview");
const statusLine = $("status");
const results = $("results");

const VOTE = {"1": ["BUY", "buy"], "-1": ["SELL", "sell"], "0": ["NEUTRAL", "flat"]};
const RING = 2 * Math.PI * 34;
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);

function setStatus(msg, kind) {
  statusLine.textContent = msg;
  statusLine.className = "status" + (kind ? " " + kind : "");
}

fileInput.addEventListener("change", () => {
  const picked = fileInput.files[0];
  fileInput.value = "";
  if (picked) scan(picked);
});

drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault();
  drop.classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
  e.preventDefault();
  drop.classList.remove("over");
}));
drop.addEventListener("drop", (e) => {
  const dropped = e.dataTransfer.files[0];
  if (dropped) scan(dropped);
});

async function scan(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("Use a PNG, JPG or WEBP screenshot.", "error");
    return;
  }
  preview.src = URL.createObjectURL(file);
  emptyBox.hidden = true;
  previewBox.hidden = false;
  drop.classList.add("scanning");
  results.hidden = true;
  results.innerHTML = "";
  setStatus("Reading candles and running 20 strategies", "");

  const body = new FormData();
  body.append("image", file);
  try {
    const response = await fetch("/api/scan", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Scan failed.");
    setStatus("", "");
    render(data);
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    drop.classList.remove("scanning");
  }
}

function render(d) {
  const sig = d.signal.toLowerCase();
  const pct = d.confidence;

  const tiles = d.strategies.map((s, i) => {
    const [label, tone] = VOTE[String(s.vote)];
    const wr = s.backtest.win_rate;
    return `<div class="tile" style="animation-delay:${i * 35}ms">
      <div class="tile-top"><span class="fam">${esc(s.category)}</span><span class="chip ${tone}">${label}</span></div>
      <div class="tile-name">${esc(s.name)}</div>
      <div class="bt">
        <div class="bt-bar"><span style="width:${wr ?? 0}%"></span></div>
        <span class="mono">${wr === null ? "n/a" : wr.toFixed(1) + "%"}</span>
        <span class="mono">${s.backtest.trades} trades</span>
      </div>
    </div>`;
  }).join("");

  results.className = `results ${sig}`;
  results.innerHTML = `
    <div class="card">
      <div class="hero">
        <div>
          <div class="sig">${esc(d.signal)}</div>
          <div class="sub">${esc(d.reason)}</div>
        </div>
        <div class="ring">
          <svg viewBox="0 0 76 76">
            <circle class="track" cx="38" cy="38" r="34"></circle>
            <circle class="arc" cx="38" cy="38" r="34" stroke-dasharray="${RING}"
              stroke-dashoffset="${RING}" data-target="${RING * (1 - pct / 100)}"></circle>
          </svg>
          <div class="val mono">${pct.toFixed(0)}%</div>
        </div>
      </div>
    </div>

    <div class="facts">
      <div class="fact"><span>Buy votes</span><b>${d.buy_votes}</b></div>
      <div class="fact"><span>Sell votes</span><b>${d.sell_votes}</b></div>
      <div class="fact"><span>Neutral</span><b>${d.neutral}</b></div>
      <div class="fact"><span>Last candle</span><b>${esc(d.last_candle)}</b></div>
    </div>

    <div class="facts">
      <div class="fact"><span>Candles read</span><b>${d.candles}</b></div>
      <div class="fact"><span>Stop</span><b>${d.stop_atr} ATR</b></div>
      <div class="fact"><span>Target</span><b>${d.target_atr} ATR</b></div>
      <div class="fact"><span>Reward : risk</span><b>1 : ${d.reward_risk}</b></div>
    </div>

    <div class="card">
      <div class="section">
        <h3>Strategies</h3>
        <small>backtest win rate and trades</small>
      </div>
      <div class="grid">${tiles}</div>
    </div>`;
  results.hidden = false;

  requestAnimationFrame(() => {
    const arc = results.querySelector(".arc");
    if (arc) arc.style.strokeDashoffset = arc.dataset.target;
  });
}

// Faint candlesticks drifting behind the page
const bg = $("bg");
const bctx = bg.getContext("2d");
const STEP = 18;
let series = [];
let shift = 0;
const yAt = (v, h) => h * 0.5 - (v - 60) * 5;

function nextCandle(prev) {
  const o = prev;
  const c = o + (60 - o) * 0.04 + (Math.random() - 0.5) * 4;
  return {o, c, h: Math.max(o, c) + Math.random() * 2, l: Math.min(o, c) - Math.random() * 2};
}

function resizeBg() {
  bg.width = innerWidth;
  bg.height = innerHeight;
  series = [];
  let p = 60;
  for (let i = 0; i < Math.ceil(innerWidth / STEP) + 2; i++) {
    const k = nextCandle(p);
    series.push(k);
    p = k.c;
  }
}

function paintBg() {
  const h = bg.height;
  bctx.clearRect(0, 0, bg.width, h);
  shift += 0.3;
  if (shift >= STEP) {
    shift -= STEP;
    series.shift();
    series.push(nextCandle(series[series.length - 1].c));
  }
  series.forEach((k, i) => {
    const x = i * STEP - shift;
    const rgb = k.c >= k.o ? "43,217,139" : "255,77,109";
    bctx.strokeStyle = `rgba(${rgb},0.16)`;
    bctx.fillStyle = `rgba(${rgb},0.12)`;
    bctx.beginPath();
    bctx.moveTo(x, yAt(k.h, h));
    bctx.lineTo(x, yAt(k.l, h));
    bctx.stroke();
    const top = yAt(Math.max(k.o, k.c), h);
    const bottom = yAt(Math.min(k.o, k.c), h);
    bctx.fillRect(x - 3.5, top, 7, Math.max(2, bottom - top));
  });
  requestAnimationFrame(paintBg);
}

addEventListener("resize", resizeBg);
resizeBg();
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) requestAnimationFrame(paintBg);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Development server. In production run: gunicorn app:app --bind 0.0.0.0:$PORT
    app.run(host="127.0.0.1", port=5000, debug=True)
