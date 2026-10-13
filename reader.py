"""
Local chart reader. Reads a chart screenshot with no external service:
  * Tesseract (installed in the container) reads the price and time labels.
  * Pixel analysis finds the plot box, the candles and their OHLC.
It refuses, with a reason, when the candles are too close together to read exactly.
"""
import io
import math
import re
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps

try:
    import pytesseract
except Exception:  # pragma: no cover - reported to the user at runtime
    pytesseract = None

MIN_PITCH_PX = 4.0      # candles closer than this cannot be separated reliably
MIN_PRICE_LABELS = 4    # OCR labels that must agree on the price scale
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
TF_CHOICES = [(1, "M1"), (5, "M5"), (15, "M15"), (30, "M30"), (60, "H1"), (240, "H4"), (1440, "D1")]


class ChartError(Exception):
    pass


def ocr_available():
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _load(raw):
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception:
        raise ChartError("That file is not a readable image.")
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (0, 0, 0))
        if "A" in im.getbands():
            bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
    return np.asarray(im).astype(np.int16)


def _masks(arr):
    """Bullish and bearish candle pixels, classified by hue so any chart theme works.
    Bullish: green to teal (hue about 90-200 degrees). Bearish: red to pink (about 335-20 degrees).
    Blue text (about 210 degrees), white text and grey borders are not candles and are excluded."""
    f = arr.astype(float) / 255.0
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    chroma = mx - mn
    sat = chroma / (mx + 1e-9)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    hue = np.zeros(mx.shape)
    d = np.where(chroma > 1e-9, chroma, 1.0)
    hue = np.where(mx == r, ((g - b) / d) % 6, hue)
    hue = np.where(mx == g, (b - r) / d + 2, hue)
    hue = np.where(mx == b, (r - g) / d + 4, hue)
    hue = hue * 60.0
    vivid = (sat >= 0.3) & (mx >= 0.25) & (chroma > 0.12)
    green = vivid & (hue >= 90) & (hue <= 200)
    red = vivid & ((hue <= 20) | (hue >= 335))
    return green, red


def _light_lines(arr):
    mn = arr.min(axis=2)
    mx = arr.max(axis=2)
    return (mn > 90) & ((mx - mn) < 45)


def find_plot(arr):
    """Plot box from the light border lines. Returns (x0, x1, y0, y1) with x1, y1 exclusive."""
    H, W, _ = arr.shape
    light = _light_lines(arr)
    colfrac = light.sum(axis=0) / float(H)
    rowfrac = light.sum(axis=1) / float(W)
    cols = [x for x in range(W // 3, W) if colfrac[x] > 0.4]
    rows = [y for y in range(H) if rowfrac[y] > 0.3]
    if not cols or len(rows) < 2:
        return None
    x1 = min(cols)
    rowfrac = light[:, :x1].sum(axis=1) / float(x1)
    rows = [y for y in range(H) if rowfrac[y] > 0.6]   # border lines span the whole chart
    if len(rows) < 2:
        return None
    y0, y1 = min(rows), max(rows)
    if x1 < W * 0.5 or y1 - y0 < H * 0.3:
        return None
    return 0, x1, y0 + 1, y1


def _ocr_words(img_gray):
    """Tesseract word boxes on a light-on-dark strip. Returns list of dicts with text and box."""
    if pytesseract is None:
        raise ChartError("The chart reader is not installed on this server (Tesseract is missing).")
    big = img_gray.resize((img_gray.width * 3, img_gray.height * 3), Image.LANCZOS)
    bw = big.point(lambda v: 255 if v > 110 else 0)
    inv = ImageOps.invert(bw)
    data = pytesseract.image_to_data(inv, config="--psm 11", output_type=pytesseract.Output.DICT)
    out = []
    for k, txt in enumerate(data["text"]):
        t = (txt or "").strip()
        if not t or float(data["conf"][k]) < 40:
            continue
        out.append({"text": t, "x": (data["left"][k] + data["width"][k] / 2.0) / 3.0,
                    "y": (data["top"][k] + data["height"][k] / 2.0) / 3.0,
                    "box": (data["left"][k] / 3.0, data["top"][k] / 3.0,
                            (data["left"][k] + data["width"][k]) / 3.0, (data["top"][k] + data["height"][k]) / 3.0)})
    return out


def read_price_scale(arr, plot):
    """Returns (a, b): price = a + b * y (full image y). b is negative (price falls as y grows)."""
    x0, x1, y0, y1 = plot
    H, W, _ = arr.shape
    strip = arr[y0:y1, x1 + 2:W].max(axis=2).astype(np.uint8)
    if strip.size == 0:
        raise ChartError("No price axis to the right of the chart.")
    words = _ocr_words(Image.fromarray(strip))
    pts = []
    for w in words:
        t = re.sub(r"^[^0-9]+", "", w["text"].replace(" ", ""))
        if re.fullmatch(r"\d{2,6}\.\d{1,6}", t):
            pts.append((w["y"] + y0, float(t)))
    if len(pts) < MIN_PRICE_LABELS:
        raise ChartError("The price axis could not be read. Make sure the price labels are visible.")
    pts.sort()
    ys = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    spacing = float(np.median(np.diff(ys))) if len(ys) > 1 else 0.0
    if spacing < 4:
        raise ChartError("The price labels are too close together to read.")
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(ys[j] - ys[i]) < 3 * spacing:
                continue
            b = (vs[j] - vs[i]) / (ys[j] - ys[i])
            a = vs[i] - b * ys[i]
            tol = 0.25 * abs(b) * spacing
            inl = np.abs(vs - (a + b * ys)) <= tol
            if best is None or inl.sum() > best[0]:
                best = (int(inl.sum()), inl.copy())
    if best is None or best[0] < MIN_PRICE_LABELS:
        raise ChartError("The price labels do not form a consistent scale, so the levels cannot be read.")
    inl = best[1]
    A = np.vstack([np.ones(inl.sum()), ys[inl]]).T
    a, b = np.linalg.lstsq(A, vs[inl], rcond=None)[0]
    if b >= 0:
        raise ChartError("The price scale is reversed or unreadable.")
    return float(a), float(b)


def _pitch_and_phase(profile):
    """Candle pitch (px) and the x of the first candle centre.
    Each candle's wick is a local peak in the column profile of coloured pixels; the median spacing of those
    peaks is the pitch, refined by a phase search against the profile."""
    xs = np.where(profile > 0)[0]
    if len(xs) < 12:
        raise ChartError("No green or red candles were found. The reader needs green (up) and red (down) "
                         "candles on a dark background.")
    prof = profile.astype(float)
    peaks = [x for x in range(1, len(prof) - 1)
             if prof[x] > 0 and prof[x] >= prof[x - 1] and prof[x] > prof[x + 1]]
    diffs = [b - a for a, b in zip(peaks, peaks[1:]) if 1 <= b - a <= 20]
    if len(diffs) < 8:
        raise ChartError("No green or red candles were found. The reader needs green (up) and red (down) "
                         "candles on a dark background.")
    p0 = float(np.median(diffs))
    best = None
    for pitch in np.arange(p0 - 0.5, p0 + 0.51, 0.02):
        if pitch < 2.0:
            continue
        for phase in np.arange(0, pitch, 0.25):
            s_, c = 0.0, phase
            while c <= xs[-1] + pitch / 2:
                if 0 <= c < len(prof):
                    s_ += prof[int(round(c))]
                c += pitch
            if best is None or s_ > best[0]:
                best = (s_, float(pitch), float(phase))
    _, pitch, phase = best
    return pitch, phase, int(xs[0]), int(xs[-1])


def _clean_masks(green, red, plot, arr):
    """Coloured masks for the candles, with price lines and header text removed."""
    x0, x1, y0, y1 = plot
    g = green[y0:y1, x0:x1].copy()
    r = red[y0:y1, x0:x1].copy()
    m = g | r
    count = m.sum(axis=1)
    W = float(m.shape[1])
    for y in np.where(count > 0.85 * W)[0]:          # a price line runs the full width of the chart
        g[y, :] = False
        r[y, :] = False
    band = arr[y0:y0 + max(20, int(0.2 * (y1 - y0))), x0:x1].max(axis=2).astype(np.uint8)
    if band.size and pytesseract is not None:
        for w in _ocr_words(Image.fromarray(band)):
            if not re.search(r"[A-Za-z0-9]{2,}", w["text"]):      # candle bodies can OCR as punctuation
                continue
            bx0, by0, bx1, by1 = w["box"]
            if by1 - by0 > 30:                                    # text is never this tall
                continue
            ya, yb = max(0, int(by0) - 2), min(g.shape[0], int(by1) + 3)
            xa, xb = max(0, int(bx0) - 2), min(g.shape[1], int(bx1) + 3)
            g[ya:yb, xa:xb] = False
            r[ya:yb, xa:xb] = False
    return g, r


def _time_labels(arr, plot):
    x0, x1, y0, y1 = plot
    H, W, _ = arr.shape
    strip = arr[y1 + 1:min(H, y1 + 60), x0:x1].max(axis=2).astype(np.uint8)
    if strip.size == 0:
        return []
    words = _ocr_words(Image.fromarray(strip))
    lines = {}
    for w in words:
        lines.setdefault(int(round(w["y"] / 14.0)), []).append(w)
    out = []
    for key in sorted(lines):
        toks = sorted(lines[key], key=lambda w: w["x"])
        k = 0
        while k < len(toks):
            t = toks[k]["text"]
            if (k + 2 < len(toks) and re.fullmatch(r"\d{1,2}", t) and toks[k + 1]["text"].title() in MONTHS
                    and re.fullmatch(r"\d{1,2}:\d{2}", toks[k + 2]["text"])):
                hh, mm = toks[k + 2]["text"].split(":")
                cx = (toks[k]["x"] + toks[k + 2]["x"]) / 2.0          # centre of the whole label
                out.append({"x": cx + x0, "dt": (int(t), MONTHS[toks[k + 1]["text"].title()], int(hh), int(mm))})
                k += 3
                continue
            if re.fullmatch(r"\d{1,2}:\d{2}", t):
                hh, mm = t.split(":")
                out.append({"x": toks[k]["x"] + x0, "dt": None, "hm": (int(hh), int(mm))})
            k += 1
    return out


def _symbol(arr, plot):
    x0, x1, y0, y1 = plot
    strip = arr[y0:min(y1, y0 + 90), x0:min(x1, x0 + 360)].max(axis=2).astype(np.uint8)
    if strip.size == 0:
        return None
    try:
        words = _ocr_words(Image.fromarray(strip))
    except ChartError:
        return None
    for w in words:
        m = re.search(r"\b([A-Z]{3,6}[A-Za-z.\-]{0,3})\b", w["text"])
        if m and re.search(r"[A-Z]{3}", m.group(1)):
            return m.group(1)[:12]
    return None


def _timeframe_and_times(labels, centres, pitch):
    """Bar size and a time for every candle centre, from the dated time-axis labels.
    Each pair of labels votes for a standard timeframe, so one misread label cannot decide it.
    Times are interpolated between labels, so a gap in the chart's data (MT5 skips missing bars) only
    affects the candles inside that gap. Returns (tf_minutes, times_or_None, note)."""
    from collections import Counter
    from datetime import timedelta
    year = datetime.utcnow().year
    dated = sorted([lb for lb in labels if lb.get("dt")], key=lambda lb: lb["x"])
    pts = []
    for lb in dated:
        d, mo, hh, mi = lb["dt"]
        try:
            t = datetime(year, mo, d, hh, mi)
        except ValueError:
            continue
        if not pts or (t > pts[-1][1] and lb["x"] > pts[-1][0]):
            pts.append((lb["x"], t))
    if len(pts) < 2:
        return None, None, "No readable date labels on the time axis"
    votes = []
    for (xa, ta), (xb, tb) in zip(pts, pts[1:]):
        bars = (xb - xa) / pitch
        if bars < 3:
            continue
        per = (tb - ta).total_seconds() / 60.0 / bars
        votes.append(min(TF_CHOICES, key=lambda c: abs(c[0] - per))[0])
    if not votes:
        return None, None, "Time labels are too close together to set the bar size"
    tf_min, count = Counter(votes).most_common(1)[0]
    if count * 2 < len(votes):
        return None, None, "The time labels do not agree on one bar size"

    def time_at(x):
        if x <= pts[0][0]:
            return pts[0][1] + timedelta(minutes=(x - pts[0][0]) / pitch * tf_min)
        if x >= pts[-1][0]:
            return pts[-1][1] + timedelta(minutes=(x - pts[-1][0]) / pitch * tf_min)
        for (xa, ta), (xb, tb) in zip(pts, pts[1:]):
            if xa <= x <= xb:
                return ta + (tb - ta) * ((x - xa) / (xb - xa))
        return pts[-1][1]

    # A label whose every neighbouring gap is impossible for this bar size was misread (for example the day).
    # Gaps in the data only make a gap wider, so a slightly wide gap is kept.
    def bad_gap(xa, ta, xb, tb):
        bars = (xb - xa) / pitch
        if bars < 1:
            return False
        per = (tb - ta).total_seconds() / 60.0 / bars
        return per > 2.0 * tf_min or per < 0.5 * tf_min

    keep = []
    for i, (x, t) in enumerate(pts):
        touching = []
        if i > 0:
            touching.append(bad_gap(pts[i - 1][0], pts[i - 1][1], x, t))
        if i + 1 < len(pts):
            touching.append(bad_gap(x, t, pts[i + 1][0], pts[i + 1][1]))
        if touching and all(touching):
            continue
        keep.append((x, t))
    if len(keep) >= 2:
        pts = keep

    times = [time_at(x).replace(second=0, microsecond=0) for x in centres]
    return tf_min, times, "Times read from the time axis"


def read_chart(raw):
    """Returns a dict with the same shape the scanner always used:
    {readable, quality, symbol, timeframe, scale, calibration, first_candle_time, last_candle_time, candles, notes}."""
    if not ocr_available():
        raise ChartError("The chart reader is not installed on this server (Tesseract is missing).")
    arr = _load(raw)
    H, W, _ = arr.shape
    plot = find_plot(arr)
    if plot is None:
        raise ChartError("The chart edges could not be found. Upload a screenshot of the whole chart window.")
    x0, x1, y0, y1 = plot
    a, b = read_price_scale(arr, plot)
    green, red = _masks(arr)
    g, r = _clean_masks(green, red, plot, arr)
    m = g | r
    profile = m.sum(axis=0)
    pitch, phase, xfirst, xlast = _pitch_and_phase(profile)
    occupied = profile > 0
    gaps, run = [], 0
    for xx in range(xfirst, xlast + 1):
        if occupied[xx]:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    median_gap = float(np.median(gaps)) if gaps else 0.0
    span = max(1, xlast - xfirst + 1)
    filled = float(occupied[xfirst:xlast + 1].sum()) / span
    if pitch < MIN_PITCH_PX or median_gap < 2 or filled > 0.85:
        raise ChartError("The candles are too close together to read exactly (about %.1f pixels apart). "
                         "Zoom the chart in so the candles are wider, then screenshot it again." % pitch)
    centres = []
    c = phase
    while c <= xlast + pitch / 2:
        if c >= 0:
            centres.append(c)
        c += pitch
    centres = [cc for cc in centres if cc >= xfirst - pitch / 2]
    if len(centres) < 40:
        raise ChartError("Only %d candles are visible; at least 40 are needed. Zoom out so more candles show, "
                         "then screenshot again." % len(centres))

    def price(yy):
        return a + b * (yy + y0)

    candles, kept = [], []
    for cc in centres:
        ia = max(0, int(math.floor(cc - pitch / 2 + 0.5)))
        ib = min(m.shape[1], int(math.ceil(cc + pitch / 2 - 0.5)) + 1)
        if ib <= ia:
            continue
        sub = m[:, ia:ib]
        ys = np.where(sub.any(axis=1))[0]
        if len(ys) == 0:
            continue
        hi_y, lo_y = ys.min(), ys.max()
        gc = int(g[:, ia:ib].sum())
        rc = int(r[:, ia:ib].sum())
        bull = gc >= rc
        rowc = sub.sum(axis=1)
        mx = rowc.max()
        if mx <= 1:
            body_top = body_bot = (hi_y + lo_y) / 2.0
        else:
            body_rows = np.where(rowc >= max(2, 0.6 * mx))[0]
            body_top, body_bot = body_rows.min(), body_rows.max()
        p_hi, p_lo = price(hi_y), price(lo_y)
        p_bt, p_bb = price(body_top), price(body_bot)
        if bull:
            o, cl = p_bb, p_bt
        else:
            o, cl = p_bt, p_bb
        candles.append([o, max(p_hi, o, cl), min(p_lo, o, cl), cl])
        kept.append(cc)

    if len(candles) < 40:
        raise ChartError("Only %d candles are visible; at least 40 are needed. Zoom out so more candles show, "
                         "then screenshot again." % len(candles))

    labels = _time_labels(arr, plot)
    tf_min, times, tnote = _timeframe_and_times(labels, kept, pitch)
    symbol = _symbol(arr, plot)
    fmt = "%Y-%m-%d %H:%M"
    out = {
        "readable": True,
        "quality": "medium",
        "symbol": symbol,
        "timeframe": None,
        "scale": "actual",
        "calibration": None,
        "first_candle_time": times[0].strftime(fmt) if times else None,
        "last_candle_time": times[-1].strftime(fmt) if times else None,
        "candle_times": [t.strftime(fmt) for t in times] if times else None,
        "candles": candles,
        "notes": tnote,
    }
    if tf_min:
        out["timeframe"] = dict(TF_CHOICES).get(tf_min)
    return out
