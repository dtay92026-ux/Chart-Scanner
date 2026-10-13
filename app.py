"""
AlgoD Scanner
=============
A chart scanner. The only input is a chart screenshot. Nothing is sent to any outside service.

  1. The chart reader finds the price and time axes (Tesseract OCR) and the candles (pixel analysis).
  2. Five rule-based strategies look at the latest candles and vote.
  3. The scanner shows the signal, the levels, and how many strategies agree.

The reader refuses charts it cannot read exactly (for example candles packed too closely together),
and says why. It never guesses. The same screenshot always gives the same result (cached by hash).

Environment variables (all optional)
    CHART_GMT_OFFSET      GMT offset of the chart's clock (default 2)
    MIN_AGREE             strategies that must agree before a signal is shown (default 3 of 5)
    ACCESS_PIN            optional PIN that must be entered before scanning
    SCAN_LIMIT_PER_HOUR   new screenshots per visitor per hour (default 30; cached results are free)
    BRAND_NAME            name shown in the header (default "AlgoD Scanner")
    PORT                  listen port (default 5000)

Run with one worker, because the result cache lives in memory:
    gunicorn -w 1 --threads 4 -t 150 -b 0.0.0.0:$PORT app:app
"""
import base64
import hashlib
import hmac
import io
import json
import math
import os
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone

from flask import Flask, Response, jsonify, request

import reader as chart_reader

try:
    from PIL import Image
except Exception:  # Pillow is optional; the image is then sent as uploaded
    Image = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BRAND = os.environ.get("BRAND_NAME", "AlgoD Scanner").strip() or "AlgoD Scanner"
ACCESS_PIN = os.environ.get("ACCESS_PIN", "").strip()
MODEL = "local-reader-1"  # part of the cache key; bump when the reader changes
SCAN_LIMIT = int(os.environ.get("SCAN_LIMIT_PER_HOUR", "30") or 30)
CHART_GMT = int(os.environ.get("CHART_GMT_OFFSET", "2") or 2)
MIN_AGREE = max(1, int(os.environ.get("MIN_AGREE", "3") or 3))
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MIN_SCAN_BARS = 60
VERSION = "2.0"
DEBUG_RAISE = False  # tests set this to surface strategy exceptions
CACHE_SIZE = 200
STRATEGY_COUNT = 5

_lock = threading.Lock()
_hits = {}
_fails = {}


# ============================================================ access control
def client_ip():
    fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return fwd or request.remote_addr or "?"


def rate_ok(bucket, limit, window=3600):
    now = time.time()
    key = (bucket, client_ip())
    with _lock:
        arr = [t for t in _hits.get(key, []) if now - t < window]
        if len(arr) >= limit:
            _hits[key] = arr
            return False
        arr.append(now)
        _hits[key] = arr
    return True


def gate():
    if not ACCESS_PIN:
        return None
    ip = client_ip()
    t = time.time()
    recent = [x for x in _fails.get(ip, []) if t - x < 300]
    if len(recent) >= 8:
        _fails[ip] = recent
        return jsonify({"ok": False, "error": "Too many attempts. Try again in a few minutes."}), 429
    given = request.headers.get("X-Access-Pin", "")
    if given and hmac.compare_digest(given.encode("utf-8"), ACCESS_PIN.encode("utf-8")):
        return None
    if given:
        recent.append(t)
    _fails[ip] = recent
    return jsonify({"ok": False, "error": "locked"}), 401


def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp



# ================================================================= data input
_DT_FORMATS = [
    "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%Y-%m-%d", "%Y.%m.%d",
]
TF_MINUTES = {"M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30, "H1": 60, "H2": 120,
              "H4": 240, "D1": 1440, "W1": 10080}
TF_SNAP = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 1440, 10080]


def tf_label(m):
    for k, v in TF_MINUTES.items():
        if v == m:
            return k
    return None


def parse_dt(s):
    s = str(s).strip().replace("Z", "")
    if re.fullmatch(r"\d{10}", s):
        return datetime.fromtimestamp(int(s), timezone.utc).replace(tzinfo=None)
    if re.fullmatch(r"\d{13}", s):
        return datetime.fromtimestamp(int(s) / 1000.0, timezone.utc).replace(tzinfo=None)
    for f in _DT_FORMATS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except ValueError:
        return None


def _num(s):
    return float(str(s).strip().replace(" ", ""))


def infer_tf(times):
    if not times or len(times) < 5 or any(t is None for t in times):
        return None
    deltas = [(times[k + 1] - times[k]).total_seconds() / 60.0 for k in range(min(len(times) - 1, 400))]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return None
    med = statistics.median(deltas)
    best = min(TF_SNAP, key=lambda v: abs(v - med))
    return best if abs(best - med) <= 0.25 * best else None


def guess_digits(p):
    p = abs(p)
    if p < 10:
        return 5
    if p < 500:
        return 3
    return 2


class VisionError(Exception):
    """A screenshot that cannot be read exactly. The message is shown to the user as-is."""


def bars_from_vision(js, tf_override=None):
    if not js.get("readable"):
        raise VisionError("Could not read this chart: %s" % (js.get("notes") or "no candles found") )
    raw = js.get("candles") or []
    out = []
    for c in raw:
        try:
            o, h, l, cl = (float(v) for v in c[:4])
        except (TypeError, ValueError, IndexError):
            continue
        if not all(math.isfinite(v) for v in (o, h, l, cl)):
            continue
        out.append([o, max(h, o, cl), min(l, o, cl), cl])
    if len(out) < MIN_SCAN_BARS:
        raise VisionError("Only %d candles could be read; at least %d are needed. Zoom out so more candles are visible, then upload again." % (len(out), MIN_SCAN_BARS))
    out = out[-300:]
    scale = js.get("scale") if js.get("scale") in ("actual", "normalized") else "normalized"
    cal = js.get("calibration")
    digits = 1
    if scale == "normalized" and isinstance(cal, list) and len(cal) >= 2:
        try:
            y1, p1, y2, p2 = float(cal[0]["y"]), float(cal[0]["price"]), float(cal[1]["y"]), float(cal[1]["price"])
            if abs(y2 - y1) > 5 and p1 != p2:
                k = (p2 - p1) / (y2 - y1)
                out = [[p1 + (v - y1) * k for v in row] for row in out]
                scale = "actual"
        except (KeyError, TypeError, ValueError):
            pass
    if scale == "actual":
        digits = guess_digits(out[-1][3])
    tf = js.get("timeframe") if js.get("timeframe") in TF_MINUTES else None
    if tf_override in TF_MINUTES:
        tf = tf_override
    tf_min = TF_MINUTES.get(tf)
    times = [None] * len(out)
    times_note = "No readable time axis"
    t0, t1 = parse_dt(js.get("first_candle_time") or ""), parse_dt(js.get("last_candle_time") or "")
    n_read = len(js.get("candles") or [])
    if t1 and tf_min:
        n = len(out)
        span = (n - 1) * tf_min
        if t0 and n_read == n:
            actual = (t1 - t0).total_seconds() / 60.0
            if abs(actual - span) <= 0.12 * span + tf_min:
                times = [t1 - timedelta(minutes=tf_min * (n - 1 - k)) for k in range(n)]
                times_note = "Times reconstructed from the time axis"
            else:
                times_note = "Time axis spans a weekend or gap, so session timing was not used"
        elif not t0:
            times = [t1 - timedelta(minutes=tf_min * (n - 1 - k)) for k in range(n)]
            times_note = "Times reconstructed from the last candle (not cross-checked)"
    bars = [{"t": times[k], "o": r[0], "h": r[1], "l": r[2], "c": r[3]} for k, r in enumerate(out)]
    meta = {"digits": digits, "scale": scale, "tf_min": tf_min, "has_times": times[-1] is not None,
            "symbol": js.get("symbol"), "quality": js.get("quality") or "medium", "notes": js.get("notes") or "",
            "times_note": times_note, "tf": tf_label(tf_min) if tf_min else None}
    return bars, meta


# ================================================================== indicators
def ema_l(v, n):
    L = len(v)
    out = [None] * L
    if L < n:
        return out
    k = 2.0 / (n + 1)
    e = sum(v[:n]) / n
    out[n - 1] = e
    for i in range(n, L):
        e = v[i] * k + e * (1 - k)
        out[i] = e
    return out


def atr_w(h, l, c, n=14):
    L = len(c)
    out = [None] * L
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, L)]
    if L <= n:
        return out
    a = sum(tr[1:n + 1]) / n
    out[n] = a
    for i in range(n + 1, L):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def rsi_w(c, n=14):
    L = len(c)
    out = [None] * L
    if L <= n:
        return out
    g = ls = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        g += max(d, 0)
        ls += max(-d, 0)
    ag, al = g / n, ls / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, L):
        d = c[i] - c[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def adx_w(h, l, c, n=14):
    L = len(c)
    adx, pdi, mdi = [None] * L, [None] * L, [None] * L
    if L < 2 * n + 2:
        return adx, pdi, mdi
    tr, pdm, mdm = [0.0] * L, [0.0] * L, [0.0] * L
    for i in range(1, L):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    a_, p_, m_ = sum(tr[1:n + 1]), sum(pdm[1:n + 1]), sum(mdm[1:n + 1])
    dx = [None] * L
    for i in range(n, L):
        if i > n:
            a_ = a_ - a_ / n + tr[i]
            p_ = p_ - p_ / n + pdm[i]
            m_ = m_ - m_ / n + mdm[i]
        pdi[i] = 100 * p_ / a_ if a_ > 0 else 0.0
        mdi[i] = 100 * m_ / a_ if a_ > 0 else 0.0
        s = pdi[i] + mdi[i]
        dx[i] = 100 * abs(pdi[i] - mdi[i]) / s if s > 0 else 0.0
    st = 2 * n - 1
    adx[st] = sum(dx[n:2 * n]) / n
    for i in range(st + 1, L):
        adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx, pdi, mdi


def boll(c, n=20, k=2.0):
    L = len(c)
    mid, up, lo = [None] * L, [None] * L, [None] * L
    for i in range(n - 1, L):
        w = c[i - n + 1:i + 1]
        m = sum(w) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in w) / n)
        mid[i], up[i], lo[i] = m, m + k * sd, m - k * sd
    return mid, up, lo


def find_swings(h, l, k=3):
    n = len(h)
    sh, sl = [], []
    for j in range(k, n - k):
        hj, lj = h[j], l[j]
        if all(hj > h[j - m] for m in range(1, k + 1)) and all(hj >= h[j + m] for m in range(1, k + 1)):
            sh.append((j + k, j, hj))
        if all(lj < l[j - m] for m in range(1, k + 1)) and all(lj <= l[j + m] for m in range(1, k + 1)):
            sl.append((j + k, j, lj))
    return sh, sl


# ---------------------------------------------------------------- sessions
def _nth_weekday(year, month, weekday, n):
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
        return d
    nxt = date(year + (month == 12), (month % 12) + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def uk_offset(d):
    return 1 if _nth_weekday(d.year, 3, 6, -1) <= d < _nth_weekday(d.year, 10, 6, -1) else 0


def us_offset(d):
    return -4 if _nth_weekday(d.year, 3, 6, 2) <= d < _nth_weekday(d.year, 11, 6, 1) else -5


class Ctx:
    """Pre-computed indicators for one price series. Strategies read data at index i only."""

    def __init__(self, bars, server_gmt=0, tf_min=None, digits=2, scale="actual", has_times=True):
        self.bars = bars
        self.n = len(bars)
        self.o = [b["o"] for b in bars]
        self.h = [b["h"] for b in bars]
        self.l = [b["l"] for b in bars]
        self.c = [b["c"] for b in bars]
        self.t = [b["t"] for b in bars]
        self.has_times = bool(has_times) and all(t is not None for t in self.t)
        self.u = [t - timedelta(hours=server_gmt) for t in self.t] if self.has_times else None
        self.tf_min = tf_min
        self.intraday = bool(self.has_times and tf_min and tf_min <= 60)
        self.digits = digits
        self.scale = scale
        self.warm = 55
        self.atr = atr_w(self.h, self.l, self.c)
        self.ema9 = ema_l(self.c, 9)
        self.ema21 = ema_l(self.c, 21)
        self.macd, self.macd_sig = macd_lines(self.c)
        self.atr10 = atr_w(self.h, self.l, self.c, 10)
        self.st_line, self.st_dir = supertrend(self.h, self.l, self.c, self.atr10, 3.0)
        self.ema20 = ema_l(self.c, 20)
        self.ema50 = ema_l(self.c, 50)
        self.rsi = rsi_w(self.c)
        self.adx, self.pdi, self.mdi = adx_w(self.h, self.l, self.c)
        self.bbm, self.bbu, self.bbl = boll(self.c)
        self.bbw = [((self.bbu[i] - self.bbl[i]) / self.atr[i]) if (self.bbu[i] is not None and self.atr[i]) else None
                    for i in range(self.n)]
        self.sh, self.sl = find_swings(self.h, self.l, 3)
        self.sh_conf = [s[0] for s in self.sh]
        self.sl_conf = [s[0] for s in self.sl]
        self._days = {}

    # -- formatting
    def fp(self, v):
        if self.scale == "actual":
            return "{:,.{d}f}".format(v, d=self.digits)
        return "{:.1f}".format(v)

    # -- swings (confirmed at or before i), newest first: [(index, price), ...]
    def sw(self, kind, i, cnt=2):
        conf, lst = (self.sh_conf, self.sh) if kind == "H" else (self.sl_conf, self.sl)
        pos = bisect_right(conf, i)
        return [(s[1], s[2]) for s in reversed(lst[max(0, pos - cnt):pos])]

    # -- support / resistance clusters from confirmed swings
    def levels(self, i, lookback=150, tol=0.35, min_touches=2):
        a = self.atr[i]
        if not a:
            return []
        pts = []
        for conf, lst in ((self.sh_conf, self.sh), (self.sl_conf, self.sl)):
            pos = bisect_right(conf, i)
            for s in lst[:pos]:
                if s[1] >= i - lookback:
                    pts.append((s[2], s[1]))
        pts.sort()
        clusters = []
        for price, idx in pts:
            if clusters and price - clusters[-1][-1][0] <= tol * a:
                clusters[-1].append((price, idx))
            else:
                clusters.append([(price, idx)])
        out = []
        for cl in clusters:
            if len(cl) >= min_touches:
                out.append({"price": sum(p for p, _ in cl) / len(cl), "touches": len(cl), "last": max(k for _, k in cl)})
        return out

    # -- sessions (UTC)
    def day(self, i):
        d = self.u[i].date()
        info = self._days.get(d)
        if info is None:
            uk, us = uk_offset(d), us_offset(d)
            mid = datetime(d.year, d.month, d.day)
            info = {
                "asia_s": mid, "lon_o": mid + timedelta(hours=8 - uk), "lon_c": mid + timedelta(hours=17 - uk),
                "ny_pre": mid + timedelta(hours=8 - us), "ny_data": mid + timedelta(hours=8.5 - us),
                "ny_o": mid + timedelta(hours=9.5 - us), "ny_c": mid + timedelta(hours=17 - us),
            }
            self._days[d] = info
        return info

    def at(self, dt):
        return bisect_left(self.u, dt)

    def covers(self, dt0):
        return self.u[0] <= dt0 + timedelta(minutes=(self.tf_min or 15) * 1.5)


def R(sig, ev, sl=None, tp=None, pat=None, forming=False):
    return {"sig": sig, "ev": ev, "sl": sl, "tp": tp, "pat": pat, "forming": forming}


def NA(reason):
    return {"sig": "NA", "ev": reason, "sl": None, "tp": None, "pat": None, "forming": False}


def _hi(x, i, n):
    return max(x.h[i - n + 1:i + 1])


def _lo(x, i, n):
    return min(x.l[i - n + 1:i + 1])


# ================================================================ strategies
STRATS = []


def strat(sid, name, group, desc, needs_time=False):
    def deco(fn):
        STRATS.append({"id": sid, "name": name, "group": group, "desc": desc, "needs_time": needs_time, "fn": fn})
        return fn
    return deco


def _session_ready(x):
    if not x.has_times:
        return "Needs candle times (not readable from this chart)"
    if not x.intraday:
        return "Needs an intraday timeframe (1 hour or lower)"
    return None


def _asian_range(x, i, info):
    """Asian range [00:00 UTC, London open). Returns (i0, i1, hi, lo) or an NA result."""
    if not x.covers(info["asia_s"]):
        return NA("The Asian session is not fully visible on this chart")
    i0, i1 = x.at(info["asia_s"]), x.at(info["lon_o"])
    if i1 - i0 < 6 or i1 > i:
        return NA("Not enough Asian-session candles on the chart")
    return i0, i1, max(x.h[i0:i1]), min(x.l[i0:i1])




# ============================================================== five strategies
# Each strategy reads the latest closed candles and only fires on a fresh event:
# the newest candle, or the one before it. The newest candle counts as the current state.

NYA_SWEEP_PTS = 3       # the wick must pierce the New York level by this many EA points
NYA_SL_PTS = 15         # the stop sits this many EA points beyond the sweep wick
NYA_MIN_RR = 1.0        # the opposite New York level must be at least this many R away
NYA_FALLBACK_RR = 3.0   # otherwise the target is 3R
GMT2 = timedelta(hours=2)  # the New York / Asian rules are written on a GMT+2 clock


def ea_point(x):
    """One point as AlgoD.mq5 counts it: 10 broker points on 3- and 5-digit symbols."""
    return (10.0 ** -x.digits) * (10 if x.digits in (3, 5) else 1)


def gmt2_clock(x):
    return [u + GMT2 for u in x.u]


@strat("ny_asia", "NY Levels Sweep", "Session",
       "Marks the high and low of the New York final hours, 20:00-23:00 GMT+2. When the Asian session "
       "(00:00-07:00 GMT+2) wicks beyond one of those levels by 3 points, it trades that candle: stop 15 points "
       "beyond the wick, target the opposite New York level, or 3R if that level is too close. One trade per session.",
       True)
def s_ny_asia(x, i):
    why = _session_ready(x)
    if why:
        return NA(why)
    loc = gmt2_clock(x)
    t = loc[i]
    a0 = datetime(t.year, t.month, t.day)
    if not (a0 <= t < a0 + timedelta(hours=7)):
        return NA("Outside the Asian sweep window (00:00-07:00 GMT+2)")
    tf = x.tf_min or 15
    ny0, ny1 = a0 - timedelta(hours=4), a0 - timedelta(hours=1)
    ny = [k for k in range(i + 1) if ny0 <= loc[k] < ny1]
    if len(ny) < max(3, int(0.6 * 180 / tf)) or loc[ny[-1]] < ny1 - timedelta(minutes=2 * tf):
        return NA("The New York final hours are not fully visible on this chart")
    nyh = max(x.h[k] for k in ny)
    nyl = min(x.l[k] for k in ny)
    levels = "NY high %s / NY low %s" % (x.fp(nyh), x.fp(nyl))
    asia = [k for k in range(i + 1) if a0 <= loc[k] <= t]
    point = ea_point(x)
    buf = NYA_SWEEP_PTS * point
    up = next((k for k in asia if x.h[k] >= nyh + buf), None)
    dn = next((k for k in asia if x.l[k] <= nyl - buf), None)
    if up is None and dn is None:
        return R("NEUTRAL", "%s not swept in this Asian session yet." % levels)
    if up is not None and dn is not None and up == dn:
        return R("NEUTRAL", "Both New York levels were wicked on one candle. Ambiguous, so no trade.")
    sell = up is not None and (dn is None or up < dn)
    s = up if sell else dn
    if s < i - 1:
        return R("NEUTRAL", "The sweep came earlier in this session. One trade per session, so no new entry.")
    entry = x.c[s]
    ext = max(x.h[s:i + 1]) if sell else min(x.l[s:i + 1])
    stop_gap = NYA_SL_PTS * point
    sl = ext + stop_gap if sell else ext - stop_gap
    risk = (sl - entry) if sell else (entry - sl)
    if risk <= 0:
        return R("NEUTRAL", "No valid stop could be placed beyond the sweep.")
    if s == i - 1 and abs(x.c[i] - entry) > 0.5 * risk:
        return R("NEUTRAL", "Price has moved on from the sweep candle. The entry is no longer fresh.")
    d = -1 if sell else 1
    opp = nyl if sell else nyh
    dist = (entry - opp) if sell else (opp - entry)
    if dist >= NYA_MIN_RR * risk:
        tp, note = opp, "target the opposite New York level %s" % x.fp(opp)
    else:
        tp, note = entry + d * NYA_FALLBACK_RR * risk, "opposite level too close, so 3R"
    cur = x.c[i]
    if (sell and cur <= tp) or (not sell and cur >= tp):
        return R("NEUTRAL", "Price has already reached the target.")
    if sell:
        ev = "Wick above the NY high (%s) in Asia. Stop above the wick, %s." % (x.fp(nyh), note)
    else:
        ev = "Wick below the NY low (%s) in Asia. Stop below the wick, %s." % (x.fp(nyl), note)
    return R("SELL" if sell else "BUY", ev, sl, tp)


def _cross(a, b, k):
    """True when series a crosses above b (up) or below b (down) at candle k."""
    return a[k - 1] is not None and b[k - 1] is not None and a[k] is not None and b[k] is not None


@strat("ema_cross", "EMA 9/21 Cross", "Trend",
       "EMA(9) crosses EMA(21) on one of the latest two candles, in the direction of the EMA(50). "
       "Stop 1.5 ATR, target 2R.")
def s_ema_cross(x, i):
    if i < 60:
        return NA("Not enough history")
    up = any(_cross(x.ema9, x.ema21, k) and x.ema9[k - 1] <= x.ema21[k - 1] and x.ema9[k] > x.ema21[k] for k in (i - 1, i))
    dn = any(_cross(x.ema9, x.ema21, k) and x.ema9[k - 1] >= x.ema21[k - 1] and x.ema9[k] < x.ema21[k] for k in (i - 1, i))
    if up == dn:
        return R("NEUTRAL", "No EMA 9/21 cross on the last two candles.")
    d = 1 if up else -1
    c, a = x.c[i], x.atr[i]
    if (c - x.ema50[i]) * d <= 0:
        return R("NEUTRAL", "The cross runs against the EMA 50 trend, so it is skipped.")
    return R("BUY" if up else "SELL",
             "EMA 9 crossed %s EMA 21 on the trend side of EMA 50." % ("above" if up else "below"),
             c - d * 1.5 * a, c + d * 3.0 * a)


@strat("rsi_reversal", "RSI Extreme Reversal", "Momentum",
       "RSI(14) leaves an extreme (below 30 or above 70) on one of the latest two candles, and a candle confirms "
       "the turn. Stop beyond the 3-candle extreme plus 0.1 ATR, target 2R.")
def s_rsi_reversal(x, i):
    if i < 60:
        return NA("Not enough history")
    up = any(x.rsi[k - 1] < 30 <= x.rsi[k] and x.c[k] > x.o[k] for k in (i - 1, i))
    dn = any(x.rsi[k - 1] > 70 >= x.rsi[k] and x.c[k] < x.o[k] for k in (i - 1, i))
    if up == dn:
        return R("NEUTRAL", "RSI %.0f: no exit from an extreme with a confirming candle." % x.rsi[i])
    c, a = x.c[i], x.atr[i]
    if up:
        sl = min(x.l[i - 3:i + 1]) - 0.1 * a
        risk = c - sl
        if risk <= 0:
            return R("NEUTRAL", "No valid stop could be placed.")
        return R("BUY", "RSI left oversold (below 30) and turned up on a bullish candle.", sl, c + 2 * risk)
    sl = max(x.h[i - 3:i + 1]) + 0.1 * a
    risk = sl - c
    if risk <= 0:
        return R("NEUTRAL", "No valid stop could be placed.")
    return R("SELL", "RSI left overbought (above 70) and turned down on a bearish candle.", sl, c - 2 * risk)


@strat("macd_cross", "MACD Cross", "Momentum",
       "MACD(12,26,9) crosses its signal line on one of the latest two candles, in the direction of the EMA(50). "
       "Stop 1.5 ATR, target 2R.")
def s_macd_cross(x, i):
    if i < 60:
        return NA("Not enough history")
    up = any(x.macd[k - 1] is not None and x.macd_sig[k - 1] is not None and x.macd_sig[k] is not None
             and x.macd[k - 1] <= x.macd_sig[k - 1] and x.macd[k] > x.macd_sig[k] for k in (i - 1, i))
    dn = any(x.macd[k - 1] is not None and x.macd_sig[k - 1] is not None and x.macd_sig[k] is not None
             and x.macd[k - 1] >= x.macd_sig[k - 1] and x.macd[k] < x.macd_sig[k] for k in (i - 1, i))
    if up == dn:
        return R("NEUTRAL", "No MACD cross on the last two candles.")
    d = 1 if up else -1
    c, a = x.c[i], x.atr[i]
    if (c - x.ema50[i]) * d <= 0:
        return R("NEUTRAL", "The cross runs against the EMA 50 trend, so it is skipped.")
    return R("BUY" if up else "SELL",
             "MACD crossed %s its signal line on the trend side of EMA 50." % ("above" if up else "below"),
             c - d * 1.5 * a, c + d * 3.0 * a)


@strat("supertrend", "Supertrend Flip", "Trend",
       "Supertrend(10, 3) flips direction on one of the latest two candles. Stop at the Supertrend line, target 2R. "
       "Skipped when that stop is wider than 3 ATR.")
def s_supertrend(x, i):
    if i < 60:
        return NA("Not enough history")
    up = any(x.st_dir[k - 1] == -1 and x.st_dir[k] == 1 for k in (i - 1, i))
    dn = any(x.st_dir[k - 1] == 1 and x.st_dir[k] == -1 for k in (i - 1, i))
    if up == dn:
        return R("NEUTRAL", "No Supertrend flip on the last two candles.")
    d = 1 if up else -1
    c, a, sl = x.c[i], x.atr[i], x.st_line[i]
    risk = (c - sl) * d
    if risk <= 0:
        return R("NEUTRAL", "Price is already beyond the Supertrend line.")
    if risk > 3 * a:
        return R("NEUTRAL", "The Supertrend stop is wider than 3 ATR, so it is skipped.")
    return R("BUY" if up else "SELL", "Supertrend flipped %s." % ("up" if up else "down"), sl, c + d * 2 * risk)


# ------------------------------------------------------------- helper series
def macd_lines(c, fast=12, slow=26, sig=9):
    ef, es = ema_l(c, fast), ema_l(c, slow)
    m = [(ef[k] - es[k]) if (ef[k] is not None and es[k] is not None) else None for k in range(len(c))]
    s = [None] * len(c)
    first = next((k for k, v in enumerate(m) if v is not None), None)
    if first is not None:
        e = ema_l(m[first:], sig)
        for k, v in enumerate(e):
            if v is not None:
                s[first + k] = v
    return m, s


def supertrend(h, l, c, atr, mult=3.0):
    """Returns (line, direction): direction +1 is up (line below price), -1 is down (line above)."""
    n = len(c)
    line, trend = [None] * n, [0] * n
    fub = flb = None
    for k in range(n):
        if atr[k] is None:
            continue
        hl2 = (h[k] + l[k]) / 2.0
        ub, lb = hl2 + mult * atr[k], hl2 - mult * atr[k]
        if fub is None:
            fub, flb = ub, lb
            trend[k] = 1 if c[k] >= hl2 else -1
        else:
            pc = c[k - 1]
            fub = ub if (ub < fub or pc > fub) else fub
            flb = lb if (lb > flb or pc < flb) else flb
            if trend[k - 1] == -1 and c[k] > fub:
                trend[k] = 1
            elif trend[k - 1] == 1 and c[k] < flb:
                trend[k] = -1
            else:
                trend[k] = trend[k - 1] or (1 if c[k] >= hl2 else -1)
        line[k] = flb if trend[k] == 1 else fub
    return line, trend


# ================================================================ scoring
def run_strategy(s, x, i):
    try:
        res = s["fn"](x, i)
    except Exception as e:  # a faulty rule must never take the scan down
        if DEBUG_RAISE:
            raise
        return NA("Could not be evaluated (%s)" % type(e).__name__)
    return res if res is not None else NA("Not enough data")


def evaluate_all(x, i):
    out = []
    for s in STRATS:
        r = dict(run_strategy(s, x, i))
        r.update({"id": s["id"], "name": s["name"], "group": s["group"]})
        out.append(r)
    return out


def trend_label(x, i):
    a, e20, e50, adx = x.atr[i], x.ema20[i], x.ema50[i], x.adx[i]
    sep = abs(e20 - e50) / a
    if e20 > e50 and x.c[i] > e50 and e50 > x.ema50[i - 5]:
        return ("Strong uptrend" if adx >= 25 else "Uptrend"), "up"
    if e20 < e50 and x.c[i] < e50 and e50 < x.ema50[i - 5]:
        return ("Strong downtrend" if adx >= 25 else "Downtrend"), "down"
    if sep < 0.4 and adx < 20:
        return "Ranging", "flat"
    return "Mixed / transitioning", "flat"


def session_label(x, i):
    if not x.has_times or not x.intraday:
        return None
    u, info = x.u[i], x.day(i)
    if u.weekday() >= 5:
        return "Weekend"
    if info["asia_s"] <= u < info["lon_o"]:
        return "Asian session"
    if info["lon_o"] <= u < info["ny_pre"]:
        return "London session"
    if info["ny_pre"] <= u < info["lon_c"]:
        return "London-New York overlap"
    if info["lon_c"] <= u < info["ny_c"]:
        return "New York session"
    return "Late New York / rollover"


def market_context(x, i):
    a = x.atr[i]
    tl, td = trend_label(x, i)
    avg = statistics.mean(v for v in x.atr[i - 50:i] if v is not None)
    ratio = a / avg if avg else 1.0
    vol = "Low" if ratio < 0.8 else "High" if ratio > 1.25 else "Normal"
    H, L = x.sw("H", i, 2), x.sw("L", i, 2)
    if len(H) == 2 and len(L) == 2:
        st = ("Higher highs, higher lows" if H[0][1] > H[1][1] and L[0][1] > L[1][1]
              else "Lower highs, lower lows" if H[0][1] < H[1][1] and L[0][1] < L[1][1] else "Mixed swings")
    else:
        st = "Not enough swings"
    r = x.rsi[i]
    mom = "Overbought" if r >= 70 else "Oversold" if r <= 30 else "Bullish" if r >= 55 else "Bearish" if r <= 45 else "Neutral"
    sq = False
    hist = [v for v in x.bbw[max(0, i - 100):i - 5] if v is not None]
    if len(hist) >= 30 and x.bbw[i] is not None:
        sq = x.bbw[i] <= sorted(hist)[int(len(hist) * 0.25)]
    return {"trend": tl, "trend_dir": td, "adx": round(x.adx[i], 1), "volatility": vol, "atr_ratio": round(ratio, 2),
            "structure": st, "rsi": round(r, 1), "momentum": mom, "squeeze": sq, "session": session_label(x, i)}


def clean(obj):
    """Round floats and drop non-finite numbers so the JSON is always valid."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj




# ================================================================= the vote
def aggregate(results):
    """Count the five votes. A signal needs MIN_AGREE strategies on one side and none on the other."""
    total = len(results)
    buys = [r for r in results if r["sig"] == "BUY"]
    sells = [r for r in results if r["sig"] == "SELL"]
    out = {"total": total, "buy": len(buys), "sell": len(sells), "required": MIN_AGREE,
           "direction": None, "verdict": "NO SIGNAL", "agree": 0, "confidence": 0, "reason": ""}
    if buys and sells:
        out["reason"] = "Strategies conflict: %d buy and %d sell. No trade." % (len(buys), len(sells))
        return out
    votes = buys or sells
    if not votes:
        out["reason"] = "No strategy is signalling on the latest candles."
        return out
    side = "BUY" if buys else "SELL"
    n = len(votes)
    out["agree"] = n
    out["confidence"] = int(round(100.0 * n / total))
    if n < MIN_AGREE:
        out["reason"] = "%d of %d strategies point %s. %d are required." % (n, total, side.lower(), MIN_AGREE)
        return out
    out.update(direction=side, verdict=side, reason="%d of %d strategies agree." % (n, total))
    return out


PLAN_PRIORITY = ["ny_asia", "ema_cross", "macd_cross", "rsi_reversal", "supertrend"]


def build_plan(x, i, side, results):
    """Levels come from one strategy, never a blend: the highest-priority strategy that agrees.
    Entry is the last close. The source is named in the result so the levels can be traced."""
    entry, a = x.c[i], x.atr[i]
    d = 1 if side == "BUY" else -1
    agreeing = [r for r in results if r["sig"] == side]
    src = min(agreeing, key=lambda r: PLAN_PRIORITY.index(r["id"]) if r["id"] in PLAN_PRIORITY else 99)
    sl = src.get("sl")
    if sl is None or (entry - sl) * d <= 0:
        sl = entry - d * 1.5 * a
    risk = abs(entry - sl)
    tp = src.get("tp")
    if tp is None or (tp - entry) * d <= 0:
        tp = entry + d * 2 * risk
    return {"side": side, "entry": entry, "sl": sl, "tp": tp, "risk": risk,
            "rr": round(abs(tp - entry) / risk, 2) if risk > 0 else None, "source": src["name"]}


def analyze(x, i):
    results = evaluate_all(x, i)
    summary = aggregate(results)
    plan = build_plan(x, i, summary["direction"], results) if summary["direction"] else None
    return {"results": results, "summary": summary, "plan": plan, "context": market_context(x, i)}


def make_result(x, i, meta):
    an = analyze(x, i)
    times_note = meta.get("times_note") or ""
    return clean({
        "ok": True,
        "version": VERSION,
        "symbol": meta.get("symbol"),
        "timeframe": tf_label(x.tf_min) if x.tf_min else None,
        "last_time": x.t[i].strftime("%d %b %Y %H:%M") if x.t[i] else None,
        "digits": x.digits,
        "session_times": bool(x.has_times and x.intraday),
        "times_note": times_note,
        "summary": an["summary"],
        "plan": an["plan"],
        "context": an["context"],
        "strategies": [{k: r.get(k) for k in ("id", "name", "group", "sig", "ev", "sl", "tp")} for r in an["results"]],
    })


# ==================================================================== routes
_CACHE = OrderedDict()
_cache_lock = threading.Lock()


def _read_upload(field):
    f = request.files.get(field)
    if not f:
        return None
    return f.read(MAX_IMAGE_BYTES + 1)


def _cache_key(raw):
    tag = "|".join([VERSION, MODEL, str(CHART_GMT), str(MIN_AGREE)])
    return hashlib.sha256(raw + tag.encode("utf-8")).hexdigest()


def _cache_get(key):
    with _cache_lock:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
        return hit


def _cache_put(key, value):
    with _cache_lock:
        _CACHE[key] = value
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)


@app.route("/api/config")
def api_config():
    return no_store(jsonify({"brand": BRAND, "reader": chart_reader.ocr_available(), "pin": bool(ACCESS_PIN), "version": VERSION,
                             "strategies": STRATEGY_COUNT, "min_agree": MIN_AGREE}))


@app.route("/api/health")
def api_health():
    return "ok"


@app.route("/api/scan", methods=["POST"])
def api_scan():
    blocked = gate()
    if blocked:
        return blocked
    raw = _read_upload("image")
    if not raw:
        return jsonify({"ok": False, "error": "Upload a chart screenshot."}), 400
    if len(raw) > MAX_IMAGE_BYTES:
        return jsonify({"ok": False, "error": "The screenshot is larger than 12 MB."}), 413

    key = _cache_key(raw)
    hit = _cache_get(key)
    if hit is not None:
        return no_store(jsonify(dict(hit, cached=True)))

    if not chart_reader.ocr_available():
        return jsonify({"ok": False, "code": "no_ocr",
                        "error": "The chart reader is not installed on this server."}), 503
    if not rate_ok("scan", SCAN_LIMIT):
        return jsonify({"ok": False, "error": "The scan limit for this hour has been reached. Try again later."}), 429
    try:
        js = chart_reader.read_chart(raw)
        bars, m = bars_from_vision(js)
        if m.get("scale") != "actual":
            raise VisionError("The price scale could not be read, so no levels can be shown. "
                              "Upload a screenshot where the price axis is visible.")
        x = Ctx(bars, CHART_GMT, m["tf_min"], m["digits"], "actual", m["has_times"])
        res = make_result(x, x.n - 1, m)
    except (VisionError, chart_reader.ChartError) as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    _cache_put(key, res)
    return no_store(jsonify(dict(res, cached=False)))


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#000000">
<title>__BRAND__</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%23000'/%3E%3Cpath d='M8 22l5-6 4 3 7-9' fill='none' stroke='%232ee59d' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#000;--s1:#07070a;--s2:#0c0c10;--line:#16161c;--line2:#23232b;
  --tx:#ececef;--mut:#82828d;--dim:#4f4f5a;
  --buy:#2ee59d;--sell:#ff5c6c;--warn:#f2b453;
  --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
[hidden]{display:none!important}
html{background:#000;color-scheme:dark}
body{background:#000;color:var(--tx);font:12px/1.5 var(--sans);-webkit-font-smoothing:antialiased;min-height:100vh;
  padding:calc(12px + env(safe-area-inset-top)) 14px calc(20px + env(safe-area-inset-bottom))}
.wrap{max-width:440px;margin:0 auto;display:flex;flex-direction:column;gap:14px}

/* header */
.top{display:flex;justify-content:space-between;align-items:center;padding:4px 2px 12px;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:9px}
.mark{width:22px;height:22px}
.wm{font:600 11px var(--sans);letter-spacing:.28em;text-transform:uppercase}
.wm b{color:var(--buy);font-weight:600}
.status{display:flex;align-items:center;gap:6px;font:600 9px var(--sans);letter-spacing:.16em;color:var(--mut);text-transform:uppercase}
.status i{width:6px;height:6px;border-radius:50%;background:var(--dim);display:block}
.status.ok i{background:var(--buy);box-shadow:0 0 8px rgba(46,229,157,.6)}
.status.err i{background:var(--sell)}

/* upload */
.upload{display:flex;flex-direction:column;align-items:center;gap:12px;padding:6px 0 2px}
.drop{position:relative;width:100%;height:168px;border-radius:24px;border:1px solid var(--line2);
  background:radial-gradient(120% 90% at 50% 0%,#101016 0%,#050508 70%);display:grid;place-items:center;cursor:pointer;
  transition:border-color .2s,box-shadow .25s,transform .12s;overflow:hidden;outline:none;-webkit-tap-highlight-color:transparent}
.drop:hover,.drop:focus-visible,.drop.over{border-color:rgba(46,229,157,.55);box-shadow:0 0 0 1px rgba(46,229,157,.12),0 0 40px -12px rgba(46,229,157,.45)}
.drop:active{transform:scale(.99)}
.plus{width:58px;height:58px;border-radius:50%;border:1px solid var(--line2);display:grid;place-items:center;
  font:300 34px/1 var(--sans);color:var(--buy);background:#08090c;transition:box-shadow .25s,border-color .25s}
.drop:hover .plus{border-color:rgba(46,229,157,.6);box-shadow:0 0 22px -4px rgba(46,229,157,.5)}
.cap{font:600 11px var(--sans);letter-spacing:.2em;text-transform:uppercase;color:var(--tx)}
.hint{font:500 10px var(--sans);letter-spacing:.06em;color:var(--dim);text-align:center}
.grid-bg{position:absolute;inset:0;pointer-events:none;opacity:.5;
  background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
  background-size:22px 22px;mask-image:radial-gradient(70% 70% at 50% 50%,#000,transparent)}
.scan{position:absolute;inset:0;pointer-events:none;display:none}
.scan::after{content:"";position:absolute;left:0;right:0;height:2px;top:0;
  background:linear-gradient(90deg,transparent,var(--buy),transparent);box-shadow:0 0 14px var(--buy);animation:sweep 1.4s ease-in-out infinite}
.busy .scan{display:block}
.busy .plus{animation:pulse 1.4s ease-in-out infinite}
@keyframes sweep{0%{top:0;opacity:0}10%{opacity:1}90%{opacity:1}100%{top:100%;opacity:0}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(46,229,157,.35)}50%{box-shadow:0 0 0 10px rgba(46,229,157,0)}}
.err{font-size:11px;color:var(--sell);text-align:center;min-height:16px}
.pin{display:flex;gap:8px;align-items:center;justify-content:center}
.pin input{background:var(--s1);border:1px solid var(--line2);border-radius:10px;color:var(--tx);padding:8px 10px;width:130px;
  font:500 13px var(--mono);letter-spacing:.3em;text-align:center;outline:none}
.pin button{background:var(--tx);color:#000;border:0;border-radius:10px;padding:8px 14px;font:600 10px var(--sans);letter-spacing:.14em;cursor:pointer}

/* result */
.card{background:linear-gradient(180deg,var(--s2),var(--s1));border:1px solid var(--line);border-radius:18px;padding:16px}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px}
.lbl{font:600 9px var(--sans);letter-spacing:.2em;color:var(--dim);text-transform:uppercase}
.verdict{font:600 30px/1 var(--sans);letter-spacing:.06em;margin:12px 0 4px}
.verdict.BUY{color:var(--buy)}.verdict.SELL{color:var(--sell)}.verdict.none{color:var(--mut)}
.reason{color:var(--mut);font-size:11.5px;min-height:17px}
.conf{margin-top:14px;display:flex;align-items:center;gap:12px}
.seg{display:flex;gap:4px;flex:1}
.seg span{flex:1;height:6px;border-radius:3px;background:var(--line2)}
.seg span.on.BUY{background:var(--buy)}.seg span.on.SELL{background:var(--sell)}.seg span.on.none{background:var(--dim)}
.pct{font:500 12px var(--mono);min-width:44px;text-align:right}
.plan{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:14px}
.plan div{background:var(--s1);padding:10px 12px}
.plan span{display:block;font:600 9px var(--sans);letter-spacing:.18em;color:var(--dim);text-transform:uppercase}
.plan b{font:500 14px var(--mono);letter-spacing:-.01em}
.plan .sl b{color:var(--sell)}.plan .tp b{color:var(--buy)}
.strats{margin-top:14px;border-top:1px solid var(--line)}
.st{display:grid;grid-template-columns:1fr auto;gap:2px 12px;padding:10px 0;border-bottom:1px solid var(--line);align-items:center}
.st:last-child{border-bottom:0}
.st .n{font:600 12px var(--sans)}
.st .g{font:500 9px var(--sans);letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin-left:8px}
.st .e{grid-column:1/2;color:var(--mut);font-size:11px;line-height:1.45}
.vote{grid-row:1/3;grid-column:2;font:600 9px var(--sans);letter-spacing:.14em;padding:4px 9px;border-radius:999px;border:1px solid var(--line2);color:var(--dim);min-width:52px;text-align:center}
.vote.BUY{color:var(--buy);border-color:rgba(46,229,157,.4);background:rgba(46,229,157,.07)}
.vote.SELL{color:var(--sell);border-color:rgba(255,92,108,.4);background:rgba(255,92,108,.07)}
.foot{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font:500 10px var(--mono);color:var(--dim);margin-top:12px}
.note{font-size:10.5px;color:var(--warn);margin-top:10px}
.psrc{font:500 10px var(--sans);color:var(--dim);margin-top:8px;letter-spacing:.02em}
.disc{text-align:center;color:var(--dim);font-size:10px;line-height:1.55;padding:4px 10px 0}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="brand">
      <svg class="mark" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="8" fill="#050507" stroke="#23232b"/><path d="M8 22l5-6 4 3 7-9" fill="none" stroke="#2ee59d" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="wm" id="brand">__BRAND__</div>
    </div>
    <div class="status" id="status"><i></i><span id="statusTxt">Connecting</span></div>
  </header>

  <section class="upload" id="uploadSec">
    <div class="pin" id="pinRow" hidden>
      <input id="pinIn" type="password" inputmode="numeric" placeholder="Access PIN" autocomplete="off">
      <button id="pinBtn" type="button">UNLOCK</button>
    </div>
    <div class="drop" id="drop" role="button" tabindex="0" aria-label="Upload chart">
      <div class="grid-bg"></div>
      <div class="plus" aria-hidden="true">+</div>
      <div class="scan"></div>
    </div>
    <div class="cap">Upload chart</div>
    <div class="hint">PNG or JPG screenshot, zoomed so the candles are clearly separated</div>
    <input id="file" type="file" accept="image/png,image/jpeg,image/webp" hidden>
    <div class="err" id="err"></div>
  </section>

  <section class="card" id="result" hidden>
    <div class="row">
      <span class="lbl">Signal</span>
      <span class="lbl" id="readTop"></span>
    </div>
    <div class="verdict none" id="verdict">NO SIGNAL</div>
    <div class="reason" id="reason"></div>
    <div class="conf">
      <div class="seg" id="seg"></div>
      <div class="pct" id="pct">0%</div>
    </div>
    <div class="plan" id="plan" hidden>
      <div><span>Entry</span><b id="pEntry">-</b></div>
      <div class="sl"><span>Stop</span><b id="pSl">-</b></div>
      <div class="tp"><span>Target</span><b id="pTp">-</b></div>
      <div><span>Reward / risk</span><b id="pRr">-</b></div>
    </div>
    <div class="psrc" id="pSrc" hidden></div>
    <div class="note" id="note" hidden></div>
    <div class="strats" id="strats"></div>
    <div class="foot"><span id="fLast"></span><span id="fCache"></span></div>
  </section>

  <p class="disc">Rule-based analysis of one screenshot. Not financial advice. Trading carries a high risk of loss.</p>
</div>

<script>
(function(){
'use strict';
var $=function(id){return document.getElementById(id);};
var cfg=null, pin=sessionStorage.getItem('scan_pin')||'';
var DASH='\u2014';

function setStatus(kind,txt){var s=$('status');s.className='status '+(kind||'');$('statusTxt').textContent=txt;}
function showErr(msg){$('err').textContent=msg||'';}
function el(tag,cls,text){var e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined&&text!==null)e.textContent=text;return e;}
function fmt(v,d){return (typeof v==='number'&&isFinite(v))?v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):DASH;}

function loadConfig(){
  fetch('/api/config',{cache:'no-store'}).then(function(r){return r.json();}).then(function(c){
    cfg=c;
    $('brand').innerHTML='';
    var parts=String(c.brand||'AlgoD Scanner').split(' ');
    var first=parts[0]||'';var rest=parts.slice(1).join(' ');
    $('brand').appendChild(document.createTextNode(first));
    if(rest){$('brand').appendChild(document.createTextNode(' '));var b=el('b','',rest);$('brand').appendChild(b);}
    if(!c.reader){setStatus('err','Offline');showErr('The chart reader is not installed on this server.');}
    else if(c.pin&&!pin){setStatus('','Locked');$('pinRow').hidden=false;}
    else setStatus('ok','Ready');
  }).catch(function(){setStatus('err','Offline');});
}

function unlock(){
  var v=$('pinIn').value.trim();
  if(!v)return;
  pin=v;sessionStorage.setItem('scan_pin',v);$('pinIn').value='';
  $('pinRow').hidden=true;setStatus('ok','Ready');showErr('');
}

function busy(on){
  var d=$('drop');
  if(on){d.parentNode.classList.add('busy');}else{d.parentNode.classList.remove('busy');}
}

function scan(file){
  if(!file)return;
  if(file.size>12*1024*1024){showErr('The screenshot is larger than 12 MB.');return;}
  showErr('');$('result').hidden=true;busy(true);setStatus('','Scanning');
  var fd=new FormData();fd.append('image',file,file.name||'chart.png');
  var h={};if(pin)h['X-Access-Pin']=pin;
  fetch('/api/scan',{method:'POST',body:fd,headers:h,cache:'no-store'}).then(function(r){
    return r.json().catch(function(){return {ok:false,error:'The server returned an unexpected reply.'};}).then(function(j){return {status:r.status,body:j};});
  }).then(function(x){
    busy(false);
    if(x.status===401){pin='';sessionStorage.removeItem('scan_pin');$('pinRow').hidden=false;setStatus('','Locked');showErr('Incorrect PIN.');return;}
    if(!x.body||!x.body.ok){showErr((x.body&&x.body.error)||'The chart could not be scanned.');setStatus(cfg&&cfg.reader?'ok':'err',cfg&&cfg.reader?'Ready':'Offline');return;}
    setStatus('ok','Ready');render(x.body);
  }).catch(function(){busy(false);setStatus('err','Offline');showErr('Could not reach the server. Try again.');});
}

function render(r){
  var s=r.summary||{}, dig=typeof r.digits==='number'?r.digits:2;
  var dir=s.direction;
  var v=$('verdict');
  v.textContent=dir||'NO SIGNAL';
  v.className='verdict '+(dir?dir:'none');
  $('reason').textContent=s.reason||'';
  var top=Math.max(s.buy||0,s.sell||0), total=s.total||5;
  var seg=$('seg');seg.textContent='';
  for(var k=0;k<total;k++){var sp=el('span',(k<top?'on '+(dir||'none'):''));seg.appendChild(sp);}
  $('pct').textContent=(s.confidence||0)+'%';
  $('readTop').textContent=[r.symbol,r.timeframe].filter(Boolean).join(' \u00b7 ')||'';
  $('readTop').style.textTransform='none';
  var p=r.plan;
  if(p){
    $('plan').hidden=false;
    $('pEntry').textContent=fmt(p.entry,dig);
    $('pSl').textContent=fmt(p.sl,dig);
    $('pTp').textContent=fmt(p.tp,dig);
    $('pRr').textContent=(typeof p.rr==='number')?(p.rr.toFixed(2)+'R'):DASH;
    $('pSrc').hidden=false;$('pSrc').textContent='Levels from '+(p.source||'the agreeing strategies')+'. Entry is the last close.';
  }else{$('plan').hidden=true;$('pSrc').hidden=true;}
  var note=$('note');
  if(!r.session_times){note.hidden=false;note.textContent='Session strategy skipped: candle times could not be read from this chart.';}
  else note.hidden=true;
  var list=$('strats');list.textContent='';
  (r.strategies||[]).forEach(function(st){
    var row=el('div','st');
    var nm=el('div','');nm.appendChild(el('span','n',st.name));nm.appendChild(el('span','g',st.group||''));
    row.appendChild(nm);
    var vt=(st.sig==='BUY'||st.sig==='SELL')?st.sig:DASH;
    row.appendChild(el('span','vote '+(vt==='BUY'||vt==='SELL'?vt:''),vt));
    row.appendChild(el('div','e',st.ev||''));
    list.appendChild(row);
  });
  var last=r.last_time?('Last candle '+r.last_time):'';
  $('fLast').textContent=last;
  $('fCache').textContent=r.cached?'Stored result':'';
  $('result').hidden=false;
  $('result').scrollIntoView({behavior:'smooth',block:'start'});
}

/* events */
var drop=$('drop'), file=$('file');
drop.addEventListener('click',function(){file.click();});
drop.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();file.click();}});
file.addEventListener('change',function(){if(file.files&&file.files[0])scan(file.files[0]);file.value='';});
['dragenter','dragover'].forEach(function(t){drop.addEventListener(t,function(e){e.preventDefault();drop.classList.add('over');});});
['dragleave','drop'].forEach(function(t){drop.addEventListener(t,function(e){e.preventDefault();drop.classList.remove('over');});});
drop.addEventListener('drop',function(e){var f=e.dataTransfer&&e.dataTransfer.files&&e.dataTransfer.files[0];if(f)scan(f);});
$('pinBtn').addEventListener('click',unlock);
$('pinIn').addEventListener('keydown',function(e){if(e.key==='Enter')unlock();});

loadConfig();
})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return no_store(Response(INDEX_HTML.replace("__BRAND__", BRAND), mimetype="text/html"))


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": "The upload is too large."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
