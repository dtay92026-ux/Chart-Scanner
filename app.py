"""
AI Chart Scanner (screenshot edition)
=====================================
Upload a screenshot of a chart. The app reads the green and red candles from
the image, then scores them with 20 widely taught technical-analysis strategies.

  * Every strategy casts one vote on the latest candles: BUY, SELL or NEUTRAL.
  * Confidence = share of the 20 strategies that agree on one direction.
  * A BUY or SELL signal is issued only when confidence reaches MIN_CONFIDENCE.
    Strategies that vote the other way still count against the total, so
    conflicting votes lower the confidence and can block the signal.
  * Each strategy is backtested on the candles in the screenshot, so its hit
    rate and trade count are shown next to its vote.

How the screenshot is read: candles are found by colour, not by an AI model.
Green pixels are bullish and red pixels are bearish. Each candle's body and
wick are measured in pixel rows. The user enters the prices at the top and
bottom edges of the cropped image, which turns rows into prices.

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
STOP_ATR = 1.5           # stop-loss distance, in ATR multiples
TARGET_ATR = 2.5         # take-profit distance, in ATR multiples
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


def s_roc(df):
    roc = 100 * df["close"].pct_change(12)
    return _signal(df.index, crossed_above(roc, 0), crossed_below(roc, 0))


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
    mid, upper, lower = bollinger(close)
    width = (upper - lower) / mid
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
    ("ROC zero-line cross", "Momentum", s_roc),
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
def read_candles(image_bytes, top_price, bottom_price):
    """
    Turn a cropped chart screenshot into OHLC candles, left to right.

    The image's top edge is top_price and its bottom edge is bottom_price.
    Pixel rows are converted to prices with that linear scale.
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

    height = colored.shape[0]
    has_candle = colored.sum(axis=0) >= 3   # ignores 1-2 px lines such as dotted price markers
    edges = np.diff(np.r_[0, has_candle.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1

    def to_price(y):
        return top_price - (y / (height - 1)) * (top_price - bottom_price)

    candles = []
    for x0, x1 in zip(starts, ends):
        g = green[:, x0:x1 + 1]
        r = red[:, x0:x1 + 1]
        row_counts = (g | r).sum(axis=1)
        rows = np.flatnonzero(row_counts)
        if rows.size == 0:
            continue

        # Body rows are the wide rows; the wick is a thin line through the middle.
        peak = row_counts.max()
        body = np.flatnonzero(row_counts >= max(min(2, peak), 0.5 * peak))
        high_y, low_y = rows[0], rows[-1]
        body_top, body_bottom = body[0], body[-1]

        bullish = g.sum() >= r.sum()
        if bullish:
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
    """Enter at each signal's close; exit at stop-loss or take-profit (stop wins ties)."""
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


def price_decimals(price):
    if price >= 1000:
        return 2  # gold, indices
    if price >= 20:
        return 3  # JPY pairs
    return 5      # most FX pairs


def analyze(df):
    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"Found {len(df)} candles. The scanner needs at least {MIN_CANDLES}. "
            "Use a screenshot that shows more bars."
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
    last_price = float(df["close"].iloc[-1])
    last_atr = float(atr_series.iloc[-1])
    decimals = price_decimals(last_price)

    direction = 0 if buy_votes == sell_votes else (1 if buy_votes > sell_votes else -1)
    agreeing = max(buy_votes, sell_votes)
    confidence = round(100 * agreeing / total, 1)
    dissenting = sell_votes if direction == 1 else buy_votes

    if direction == 0:
        signal, reason = "HOLD", "Buy and sell votes are tied."
    elif confidence < MIN_CONFIDENCE:
        signal = "HOLD"
        reason = f"Only {agreeing} of {total} strategies agree; {MIN_CONFIDENCE}% is required."
    else:
        signal = "BUY" if direction == 1 else "SELL"
        reason = f"{agreeing} of {total} strategies agree on {signal}."
        if dissenting:
            reason += f" {dissenting} voted the other way."

    entry = stop_loss = take_profit = None
    if signal != "HOLD":
        entry = round(last_price, decimals)
        stop_loss = round(last_price - direction * STOP_ATR * last_atr, decimals)
        take_profit = round(last_price + direction * TARGET_ATR * last_atr, decimals)

    return {
        "signal": signal,
        "reason": reason,
        "confidence": confidence,
        "buy_votes": buy_votes,
        "sell_votes": sell_votes,
        "neutral": neutral,
        "last_price": round(last_price, decimals),
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "decimals": decimals,
        "strategies": rows,
    }


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def parse_price(text, name):
    try:
        return float((text or "").strip())
    except ValueError:
        raise ValueError(f"Enter the {name} shown on the price scale.")


@app.get("/")
def home():
    return PAGE


@app.post("/api/scan")
def scan():
    try:
        upload = request.files.get("image")
        if not upload or not upload.filename:
            raise ValueError("Upload a chart screenshot.")

        top = parse_price(request.form.get("top_price"), "top price")
        bottom = parse_price(request.form.get("bottom_price"), "bottom price")
        if top <= bottom:
            raise ValueError("The top price must be higher than the bottom price.")

        df = read_candles(upload.read(), top, bottom)
        if df.empty:
            raise ValueError(
                "No green or red candles found. Crop to the candles and use a chart with coloured candles."
            )

        result = analyze(df)
        result.update(
            symbol=request.form.get("symbol", "").strip() or upload.filename,
            candles=len(df),
            top_price=top,
            bottom_price=bottom,
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Scan failed")
        return jsonify(error="Scan failed. Try a clearer, uncompressed screenshot."), 500


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>AI Chart Scanner</title>
<style>
:root{--bg:#0a0f1c;--panel:#111a2e;--panel-2:#0c1426;--line:#1d2943;--text:#e7edf7;--muted:#8793ab;--accent:#4c8dff;--buy:#14c784;--sell:#f0434c;--hold:#f5b93a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:18px 28px;border-bottom:1px solid var(--line)}
.brand{font-weight:700;font-size:17px;letter-spacing:.2px}
.brand span{color:var(--accent)}
main{max-width:980px;margin:0 auto;padding:28px 20px 56px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:18px}
.muted{color:var(--muted)}
.small{font-size:13px}
.error{color:var(--sell)}
form{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;align-items:end}
.span-2{grid-column:span 2}
label{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px}
input{width:100%;padding:11px 12px;background:var(--panel-2);border:1px solid var(--line);border-radius:9px;color:var(--text);font:inherit}
input[type=file]{padding:9px 10px}
input:focus{outline:none;border-color:var(--accent)}
button{width:100%;padding:12px 22px;border:0;border-radius:9px;background:var(--accent);color:#fff;font-family:inherit;font-size:15px;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
#status{min-height:20px;margin:14px 0 4px}
.signal-head{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}
.sig-label{font-size:40px;font-weight:800;letter-spacing:1px;margin:4px 0}
.signal.buy .sig-label{color:var(--buy)}
.signal.sell .sig-label{color:var(--sell)}
.signal.hold .sig-label{color:var(--hold)}
.conf{text-align:right}
.conf-value{font-size:34px;font-weight:700}
.bar{height:8px;background:var(--panel-2);border-radius:99px;overflow:hidden;margin:18px 0}
.bar span{display:block;height:100%;background:var(--accent);transition:width .6s}
.signal.buy .bar span{background:var(--buy)}
.signal.sell .bar span{background:var(--sell)}
.signal.hold .bar span{background:var(--hold)}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.levels div{background:var(--panel-2);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.levels span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
.levels b{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:16px;font-weight:600}
h3{margin:0 0 12px;font-size:16px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:10px 8px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:11px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.num{text-align:right;font-family:ui-monospace,Menlo,Consolas,monospace}
.pill{display:inline-block;min-width:76px;text-align:center;padding:3px 10px;border-radius:99px;font-size:12px;font-weight:700}
.pill.buy{background:rgba(20,199,132,.14);color:var(--buy)}
.pill.sell{background:rgba(240,67,76,.14);color:var(--sell)}
.pill.flat{background:#18223a;color:var(--muted)}
.disclaimer{font-size:12px;color:var(--muted);text-align:center;margin-top:24px}
@media(max-width:720px){
  form{grid-template-columns:1fr}
  .span-2{grid-column:auto}
  .levels{grid-template-columns:1fr 1fr}
  .signal-head{flex-direction:column}
  .conf{text-align:left}
  .sig-label{font-size:30px}
}
</style>
</head>
<body>
<header>
  <div class="brand">AI <span>Chart Scanner</span></div>
  <div class="muted small">Screenshot in &middot; consensus signal out</div>
</header>
<main>
  <section class="card">
    <form id="scan-form">
      <div>
        <label for="symbol">Label (optional)</label>
        <input id="symbol" name="symbol" placeholder="EURUSD 1H" autocomplete="off">
      </div>
      <div>
        <label for="top_price">Price at top edge</label>
        <input id="top_price" name="top_price" inputmode="decimal" placeholder="1.0850" required>
      </div>
      <div>
        <label for="bottom_price">Price at bottom edge</label>
        <input id="bottom_price" name="bottom_price" inputmode="decimal" placeholder="1.0780" required>
      </div>
      <div class="span-2">
        <label for="image">Chart screenshot</label>
        <input id="image" name="image" type="file" accept="image/png,image/jpeg,image/webp" required>
      </div>
      <button id="scan-btn" type="submit">Scan chart</button>
    </form>
    <p id="status" class="muted small"></p>
    <p class="muted small">Crop the screenshot to the candles only, leaving out the price scale, volume and indicator panes. Enter the prices shown at the top and bottom edges of the cropped image. Works with green and red candles.</p>
  </section>

  <div id="results"></div>

  <p class="disclaimer">Educational tool, not financial advice. Backtests use simplified fills and past results do not guarantee future performance. Trading carries a high risk of loss.</p>
</main>

<script>
const $ = (id) => document.getElementById(id);
const form = $("scan-form");
const button = $("scan-btn");
const statusLine = $("status");
const results = $("results");

const VOTE = {"1": ["BUY", "buy"], "-1": ["SELL", "sell"], "0": ["NEUTRAL", "flat"]};
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  button.disabled = true;
  results.innerHTML = "";
  statusLine.className = "muted small";
  statusLine.textContent = "Reading candles and running 20 strategies\u2026";
  try {
    const response = await fetch("/api/scan", {method: "POST", body: new FormData(form)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Scan failed.");
    statusLine.textContent = "";
    render(data);
  } catch (err) {
    statusLine.className = "error small";
    statusLine.textContent = err.message;
  } finally {
    button.disabled = false;
  }
});

function render(d) {
  const cls = d.signal.toLowerCase();
  const fmt = (v) => (v === null ? "\u2014" : Number(v).toFixed(d.decimals));

  const rows = d.strategies.map((s) => {
    const [label, tone] = VOTE[String(s.vote)];
    const win = s.backtest.win_rate === null
      ? '<span class="muted">n/a</span>'
      : `${s.backtest.win_rate}% <span class="muted">(${s.backtest.trades})</span>`;
    return `<tr>
      <td>${esc(s.name)}</td>
      <td class="muted">${esc(s.category)}</td>
      <td><span class="pill ${tone}">${label}</span></td>
      <td class="num">${win}</td>
    </tr>`;
  }).join("");

  results.innerHTML = `
    <section class="card signal ${cls}">
      <div class="signal-head">
        <div>
          <div class="muted small">${esc(d.symbol)} &middot; ${d.candles} candles read from screenshot</div>
          <div class="sig-label">${esc(d.signal)}</div>
          <div class="muted">${esc(d.reason)}</div>
        </div>
        <div class="conf">
          <div class="conf-value">${d.confidence.toFixed(0)}%</div>
          <div class="muted small">confidence</div>
        </div>
      </div>
      <div class="bar"><span style="width:${d.confidence}%"></span></div>
      <div class="levels">
        <div><span>Last close</span><b>${fmt(d.last_price)}</b></div>
        <div><span>Stop loss</span><b>${fmt(d.stop_loss)}</b></div>
        <div><span>Take profit</span><b>${fmt(d.take_profit)}</b></div>
        <div><span>Buy / sell / neutral</span><b>${d.buy_votes} / ${d.sell_votes} / ${d.neutral}</b></div>
      </div>
    </section>

    <section class="card">
      <h3>Strategy breakdown</h3>
      <table>
        <thead>
          <tr><th>Strategy</th><th>Family</th><th>Vote now</th><th class="num">Backtest win rate (trades)</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    # Development server. In production run: gunicorn app:app
    app.run(host="127.0.0.1", port=5000, debug=True)
