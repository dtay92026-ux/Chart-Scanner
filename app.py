"""
Chart Scanner
=============
Upload a chart screenshot. The app reads the green and red candles from the
image and checks five widely used strategies on the chart as it is now.

  * Each strategy reads the current state of the chart and votes BUY, SELL or
    NEUTRAL. Votes do not depend on crossovers from earlier in the chart.
  * Neutral bands mean a small move cannot flip a vote.
  * Confidence = share of the five strategies that agree.
  * A BUY or SELL needs at least MIN_AGREE agreeing and at most MAX_DISSENT
    dissenting. Otherwise the signal is HOLD.

Only the uploaded screenshot is used. No prices or data are fetched.
Educational use only. This is not financial advice.
"""

import io
import mimetypes

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from PIL import Image

mimetypes.add_type("application/manifest+json", ".webmanifest")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MIN_CANDLES = 60       # indicators need this many candles to settle
MIN_AGREE = 3          # strategies that must point the same way for a signal
MAX_DISSENT = 1        # strategies allowed to point the other way
ATR_PERIOD = 14
STOP_ATR = 1.5         # stop distance, in ATR units (the average candle range)
TARGET_ATR = 2.5       # target distance, in ATR units
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


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
    return (100 - 100 / (1 + gain / loss)).fillna(50.0)


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def adx(df, n=14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    smoothed_atr = atr(df, n)
    plus_di = (100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / smoothed_atr).fillna(0.0)
    minus_di = (100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / smoothed_atr).fillna(0.0)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0.0)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), plus_di, minus_di


def supertrend(df, n=10, mult=3.0):
    """Returns (trend, line): trend is +1 above the Supertrend line and -1 below it."""
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
    line = np.where(trend == 1, dn, up)
    return pd.Series(trend, index=df.index), pd.Series(line, index=df.index)


# ---------------------------------------------------------------------------
# The five strategies. Each reads the chart as it is now, including the newest
# candle, and returns BUY (1), SELL (-1) or NEUTRAL (0) with a short reading.
# Thresholds are in ATR units or fixed levels, so they do not depend on price.
# ---------------------------------------------------------------------------
def read_strategies(df):
    close = df["close"]
    unit = float(atr(df).iloc[-1])
    readings = []

    # 1. Moving average trend: price and the 20 and 50 averages stacked in order.
    e20, e50 = ema(close, 20), ema(close, 50)
    c, a20, a50 = float(close.iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1])
    if c > a20 > a50 and a20 - a50 >= 0.2 * unit:
        vote, text = 1, "Price is above the 20 and 50 averages, and the 20 is above the 50"
    elif c < a20 < a50 and a50 - a20 >= 0.2 * unit:
        vote, text = -1, "Price is below the 20 and 50 averages, and the 20 is below the 50"
    else:
        vote, text = 0, "The averages are mixed or too close to call"
    readings.append(("Moving average trend", "Trend", vote, text))

    # 2. Supertrend: which side of the trailing ATR line price is on.
    trend, line_st = supertrend(df)
    st, st_line = int(trend.iloc[-1]), float(line_st.iloc[-1])
    if abs(c - st_line) < 0.1 * unit:
        vote, text = 0, "Price is too close to the Supertrend line to call"
    elif st == 1:
        vote, text = 1, "Price is above the Supertrend line"
    else:
        vote, text = -1, "Price is below the Supertrend line"
    readings.append(("Supertrend", "Trend", vote, text))

    # 3. MACD: the line must sit on one side of zero and clearly on one side of its signal.
    line, signal_line = macd(close)
    m_now, s_now = float(line.iloc[-1]), float(signal_line.iloc[-1])
    if m_now > 0 and m_now - s_now >= 0.05 * unit:
        vote, text = 1, "MACD is above zero and above its signal line"
    elif m_now < 0 and s_now - m_now >= 0.05 * unit:
        vote, text = -1, "MACD is below zero and below its signal line"
    else:
        vote, text = 0, "MACD is mixed or flat"
    readings.append(("MACD", "Momentum", vote, text))

    # 4. RSI zones: a neutral band between 45 and 55 keeps small moves from flipping it.
    r = float(rsi(close).iloc[-1])
    if r >= 55:
        vote, text = 1, f"RSI {r:.0f}, bullish zone"
    elif r <= 45:
        vote, text = -1, f"RSI {r:.0f}, bearish zone"
    else:
        vote, text = 0, f"RSI {r:.0f}, neutral zone"
    readings.append(("RSI zone", "Momentum", vote, text))

    # 5. ADX direction: a trend must be strong enough, and one side must lead clearly.
    adx_v, plus_di, minus_di = adx(df)
    v, p_di, m_di = float(adx_v.iloc[-1]), float(plus_di.iloc[-1]), float(minus_di.iloc[-1])
    if v >= 25 and p_di - m_di >= 5:
        vote, text = 1, f"ADX {v:.0f}, strong upward trend"
    elif v >= 25 and m_di - p_di >= 5:
        vote, text = -1, f"ADX {v:.0f}, strong downward trend"
    elif v >= 20:
        vote, text = 0, f"ADX {v:.0f}, trend forming"
    else:
        vote, text = 0, f"ADX {v:.0f}, no clear trend"
    readings.append(("ADX direction", "Trend strength", vote, text))

    return [
        {"name": name, "family": family, "vote": int(vote), "reading": text}
        for name, family, vote, text in readings
    ]


# ---------------------------------------------------------------------------
# Reading candles from a screenshot
# ---------------------------------------------------------------------------
def _colour_masks(img):
    """
    Find green and red pixels. The chart background is the most common colour, and a
    pixel counts as coloured when it is far enough from that background. Testing the
    distance from the background, rather than a brightness threshold, keeps thin wicks
    and blurred edges (JPEG) readable. Channel ratios keep out cyan, blue and orange lines.
    """
    rgb = np.asarray(img).astype(np.int16)
    coarse = (rgb // 16).reshape(-1, 3)
    codes = coarse[:, 0] * 256 + coarse[:, 1] * 16 + coarse[:, 2]
    values, counts = np.unique(codes, return_counts=True)
    most_common = int(values[np.argmax(counts)])
    background = np.array([most_common // 256, (most_common // 16) % 16, most_common % 16]) * 16 + 8

    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2).astype(float))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    is_green = (g >= 1.2 * r) & (g >= b - 10) & (g - r >= 25)
    is_red = (r >= 1.6 * g) & (r - b >= 25) & (r - g >= 40)
    green = (distance >= 50) & is_green
    red = (distance >= 50) & is_red
    solid = ((distance >= 110) & is_green) | ((distance >= 110) & is_red)   # bright cores only
    # Drop pixels with no coloured neighbour above or below. Candle bodies and wicks are
    # vertical, so they keep their pixels, but dotted or dashed price lines do not.
    above = np.zeros_like(solid)
    above[1:] = solid[:-1]
    below = np.zeros_like(solid)
    below[:-1] = solid[1:]
    solid = solid & (above | below)
    return green, red, solid


def _longest_run(mask):
    """Longest horizontal run of True pixels in each row of a small mask."""
    run = np.zeros(mask.shape[0], dtype=int)
    best = np.zeros(mask.shape[0], dtype=int)
    for x in range(mask.shape[1]):
        run = (run + 1) * mask[:, x]
        best = np.maximum(best, run)
    return best


def _row_groups(present, max_gap=2):
    """
    Groups of coloured rows, as (top, bottom). Gaps of up to max_gap rows stay inside one
    group, because compression can break a wick into pieces. One-pixel-tall groups (dotted
    price lines) are dropped.
    """
    edges = np.diff(np.r_[0, present.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    groups = []
    for s, e in zip(starts, ends):
        if groups and s - groups[-1][1] - 1 <= max_gap:
            groups[-1] = (groups[-1][0], int(e))
        else:
            groups.append((int(s), int(e)))
    return [(s, e) for s, e in groups if e - s + 1 >= 2]


def _grow(mask, seed, max_gap=0):
    """
    Return the (top, bottom) rows of the block of True rows that contains seed.
    Gaps of up to max_gap rows do not end the block, because compression can
    break a thin wick into pieces.
    """
    def walk(step):
        position = last_true = seed
        while True:
            nxt = position + step
            if nxt < 0 or nxt >= len(mask):
                break
            if mask[nxt]:
                position = last_true = nxt
            elif abs(nxt - last_true) <= max_gap:
                position = nxt
            else:
                break
        return last_true

    return walk(-1), walk(1)


def _fit_grid(strong_counts):
    """
    Find the candle grid. Each run of coloured columns is one candle, so its middle
    is that candle's centre. Runs much wider than a candle (price labels, buttons)
    and runs that do not sit on the grid are ignored.

    Returns the spacing, the position of candle 0, and the grid index of each candle.
    """
    mask = strong_counts > 0
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    if len(starts) < 3:
        raise ValueError("No candles found in this screenshot. Try a screenshot of the chart.")

    widths = ends - starts + 1
    centres = (starts + ends) / 2.0
    typical_width = np.percentile(widths, 75)
    fits = widths <= 2.5 * max(typical_width, 3)   # candles share one width; labels and buttons do not
    centres, widths = centres[fits], widths[fits]
    if len(centres) < 3:
        raise ValueError("No candles found in this screenshot. Try a screenshot of the chart.")

    diffs = np.diff(centres)
    typical = np.median(diffs)
    single = diffs[(diffs > 0.75 * typical) & (diffs < 1.25 * typical)]   # gaps of exactly one candle
    touching = "Candles touch each other in this screenshot. Use fewer bars or a wider chart."
    if single.size < 3:
        raise ValueError(touching)
    rough = single.mean()   # sub-pixel: whole-pixel gaps average out to the true spacing

    index = np.round((centres - centres[0]) / rough)
    inlier = np.ones(len(centres), dtype=bool)
    for _ in range(4):
        design = np.c_[np.ones(inlier.sum()), index[inlier]]
        (offset, period), *_ = np.linalg.lstsq(design, centres[inlier], rcond=None)
        inlier = np.abs(centres - (offset + period * index)) < 0.25 * period

    if np.median(widths[inlier]) >= 0.95 * period:
        raise ValueError(touching)
    return float(period), float(offset), index[inlier].astype(int)


def _price_band(solid):
    """
    Find the price pane: the top-most block of rows that holds a substantial share of the
    candle colour. Volume bars and indicator panes sit below it, separated by blank rows.
    """
    groups = _row_groups(solid.sum(axis=1) > 0, max_gap=12)
    if not groups:
        raise ValueError("No green or red candles found. Try a screenshot of the candles.")
    totals = [int(solid[top:bottom + 1].sum()) for top, bottom in groups]
    biggest = max(totals)
    for (top, bottom), total in zip(groups, totals):   # ordered top to bottom
        if total >= 0.3 * biggest:
            return top, bottom
    return groups[0]


def read_candles(image_bytes):
    """
    Turn a chart screenshot into OHLC candles, oldest first.

    Prices are relative: 1.0 at the top edge of the image and 0.0 at the bottom.
    The strategies only use candle shape, so the absolute price level is not needed.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("That file is not an image the scanner can read. Use PNG, JPG or WEBP.")

    green, red, solid = _colour_masks(img)
    full_height = solid.shape[0]
    band_top, band_bottom = _price_band(solid)
    green = green[band_top:band_bottom + 1]
    red = red[band_top:band_bottom + 1]
    solid = solid[band_top:band_bottom + 1]
    width = solid.shape[1]

    col_counts = solid.sum(axis=0)                             # candle cores find the candle columns
    strong_counts = np.where(col_counts >= 3, col_counts, 0)   # wicks and bodies; ignores 1-2 px lines
    if strong_counts.max() == 0:
        raise ValueError("No green or red candles found. Try a screenshot of the candles.")

    period, offset, grid = _fit_grid(strong_counts)

    def to_price(y_abs):
        return 1.0 - y_abs / (full_height - 1)

    candles = []
    for k in range(int(grid.min()), int(grid.max()) + 1):
        centre = offset + period * k
        # Only columns whose centre lies inside this candle's slot.
        x0 = max(0, int(np.ceil(centre - period / 2 + 1e-9)))
        x1 = min(width, int(np.floor(centre + period / 2 - 1e-9)) + 1)
        if x1 <= x0:
            continue
        g = green[:, x0:x1]
        r = red[:, x0:x1]
        both = g | r
        if solid[:, x0:x1].sum(axis=0).max() < 2:   # no candle in this slot (a 2 px doji still counts)
            continue

        # The body is the block of wide rows around the widest row. The wick is the
        # contiguous run of coloured rows above and below it.
        runs = _longest_run(both)
        peak_row = int(np.argmax(runs))
        peak = runs[peak_row]
        body_top, body_bottom = _grow(runs >= max(min(2, peak), 0.75 * peak), peak_row)
        present = both.any(axis=1)
        high_y = _grow(present, body_top, max_gap=3)[0]
        low_y = _grow(present, body_bottom, max_gap=3)[1]

        body_green = g[body_top:body_bottom + 1].sum()
        body_red = r[body_top:body_bottom + 1].sum()
        if body_green >= body_red:
            open_y, close_y = body_bottom, body_top
        else:
            open_y, close_y = body_top, body_bottom

        candles.append({
            "open": to_price(open_y + band_top),
            "high": to_price(high_y + band_top),
            "low": to_price(low_y + band_top),
            "close": to_price(close_y + band_top),
        })

    if not candles:
        raise ValueError("No green or red candles found. Try a screenshot of the candles.")
    return pd.DataFrame(candles, columns=["open", "high", "low", "close"])


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def analyze(df):
    if len(df) < MIN_CANDLES:
        raise ValueError("Not enough candles in this screenshot. Use a screenshot with more bars.")

    strategies = read_strategies(df)
    total = len(strategies)
    buy = sum(s["vote"] == 1 for s in strategies)
    sell = sum(s["vote"] == -1 for s in strategies)
    neutral = total - buy - sell
    lead = max(buy, sell)
    other = min(buy, sell)
    confidence = round(100 * lead / total)

    if buy == sell:
        signal, reason = "HOLD", "Buy and sell readings are level."
    else:
        word = "BUY" if buy > sell else "SELL"
        if lead >= MIN_AGREE and other <= MAX_DISSENT:
            signal = word
            reason = f"{lead} of {total} strategies point {word.lower()}."
            if other:
                reason += f" {other} voted the other way."
        elif lead < MIN_AGREE:
            signal = "HOLD"
            reason = f"Only {lead} of {total} strategies point {word.lower()}. {MIN_AGREE} are needed."
        else:
            signal = "HOLD"
            reason = f"Strategies are split: {lead} {word.lower()}, {other} the other way."

    newest = df.iloc[-1]
    return {
        "signal": signal,
        "reason": reason,
        "confidence": confidence,
        "buy_votes": buy,
        "sell_votes": sell,
        "neutral": neutral,
        "newest_candle": "Bullish" if newest["close"] >= newest["open"] else "Bearish",
        "stop_atr": STOP_ATR,
        "target_atr": TARGET_ATR,
        "reward_risk": round(TARGET_ATR / STOP_ATR, 2),
        "strategies": strategies,
    }


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return PAGE


@app.post("/api/scan")
def scan():
    try:
        upload = request.files.get("image")
        if not upload or not upload.filename:
            raise ValueError("Upload a chart screenshot.")
        return jsonify(analyze(read_candles(upload.read())))
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
<meta name="theme-color" content="#070b14">
<link rel="icon" href="/static/favicon.ico" sizes="any">
<link rel="icon" type="image/png" sizes="32x32" href="/static/favicon-32.png">
<link rel="apple-touch-icon" href="/static/apple-touch-icon.png">
<link rel="manifest" href="/static/manifest.webmanifest">
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
.facts.three{grid-template-columns:repeat(3,1fr)}
.fact{background:var(--panel-2);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.fact span{display:block;font-size:10px;letter-spacing:1.2px;text-transform:uppercase;color:var(--muted)}
.fact b{display:block;margin-top:2px;font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:13px;font-weight:600}

.section{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:12px}
.section h3{margin:0;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);font-weight:600}
.section small{font-size:10px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.grid .tile:last-child:nth-child(odd){grid-column:1 / -1}
.tile{background:var(--panel-2);border:1px solid var(--line);border-radius:12px;padding:11px 12px;animation:rise .4s both}
.tile-top{display:flex;justify-content:space-between;align-items:center}
.fam{font-size:10px;letter-spacing:1.1px;text-transform:uppercase;color:var(--muted)}
.chip{font-size:10px;font-weight:700;letter-spacing:1px;padding:2px 8px;border-radius:6px;border:1px solid}
.chip.buy{color:var(--buy);border-color:rgba(43,217,139,.4);background:rgba(43,217,139,.08)}
.chip.sell{color:var(--sell);border-color:rgba(255,77,109,.4);background:rgba(255,77,109,.08)}
.chip.flat{color:var(--muted);border-color:var(--line)}
.tile-name{margin:7px 0 4px;font-size:13px;font-weight:500}
.reading{font-size:12px;color:var(--muted);line-height:1.4}

.foot{margin:22px 0 0;text-align:center;font-size:11px;color:var(--muted)}

@media(max-width:600px){
  .facts{grid-template-columns:repeat(2,1fr)}
  .facts.three{grid-template-columns:repeat(3,1fr)}
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
  <div class="meta">5 strategies</div>
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
  <p class="hint">Use a screenshot of the chart. Green and red candles are read.</p>
  <p id="status" class="status"></p>

  <section id="results" class="results" hidden></section>

  <p class="foot">Educational tool, not financial advice. ATR is the average candle range. Past readings do not guarantee future ones.</p>
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
  setStatus("Reading the chart", "");

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
    return `<div class="tile" style="animation-delay:${i * 40}ms">
      <div class="tile-top"><span class="fam">${esc(s.family)}</span><span class="chip ${tone}">${label}</span></div>
      <div class="tile-name">${esc(s.name)}</div>
      <div class="reading">${esc(s.reading)}</div>
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
          <div class="val mono">${pct}%</div>
        </div>
      </div>
    </div>

    <div class="facts">
      <div class="fact"><span>Buy votes</span><b>${d.buy_votes}</b></div>
      <div class="fact"><span>Sell votes</span><b>${d.sell_votes}</b></div>
      <div class="fact"><span>Neutral</span><b>${d.neutral}</b></div>
      <div class="fact"><span>Newest candle</span><b>${esc(d.newest_candle)}</b></div>
    </div>

    <div class="facts three">
      <div class="fact"><span>Stop</span><b>${d.stop_atr} ATR</b></div>
      <div class="fact"><span>Target</span><b>${d.target_atr} ATR</b></div>
      <div class="fact"><span>Reward : risk</span><b>1 : ${d.reward_risk}</b></div>
    </div>

    <div class="card">
      <div class="section">
        <h3>Strategies</h3>
        <small>current state of the chart</small>
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
