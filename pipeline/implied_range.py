"""Implied range for ES and NQ: EM = price x IV / sqrt(252) (IV/16), laid out as a sigma ladder.

* Front quarterly future (Mar/Jun/Sep/Dec, third Friday; roll 8 calendar days before expiry)
  and its fair value = the option-implied forward at that expiry (put-call parity).
* Calibration over the last five years: how often the index closed inside +-k sigma of the
  prior close, and how often the day's high / low touched +-k sigma, vs. the normal /
  driftless-Brownian benchmarks. Descriptive statistics, not a forecast test.
* Level study: per-session open / high / low / close relative to the prior close, so the browser can
  measure touch -> held / broke for any level, filter and volatility index.
"""
from __future__ import annotations

import datetime as dt, json, math, re, urllib.request

import numpy as np
from scipy.stats import norm

K_LEVELS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
UA = {"User-Agent": "Mozilla/5.0 (volatility research snapshot; github.com/jonaslffr-ship-it)", "Accept": "application/json"}
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}
NASDAQ_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                  "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
                  "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}   # the API stalls on non-browser clients


def _get(url, timeout=60, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or UA), timeout=timeout) as r:
        return r.read()


def third_friday(y: int, m: int) -> dt.date:
    d = dt.date(y, m, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def front_quarterly(today: dt.date) -> tuple[dt.date, str]:
    y, m = today.year, today.month
    for _ in range(6):
        qm = ((m - 1) // 3 + 1) * 3
        exp = third_friday(y, qm)
        if today < exp - dt.timedelta(days=8):
            return exp, MONTH_CODE[qm] + str(y % 10)
        m = qm + 1
        if m > 12:
            m, y = 1, y + 1
    raise RuntimeError("no front contract")


def parity_forward(options: list, target: dt.date, roots=("NDX", "NDXP")) -> dict | None:
    """Forward and discount factor for one expiry date from put-call parity (C - P = D (F - K))."""
    pat = re.compile(r"^([A-Z]+?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
    by = {}
    for o in options:
        m = pat.match(o["option"])
        if not m:
            continue
        root, yy, mm, dd, cp, kk = m.groups()
        if dt.date(2000 + int(yy), int(mm), int(dd)) != target or root not in roots:
            continue
        by.setdefault(root, {}).setdefault(int(kk) / 1000.0, {})[cp] = o
    for root in roots:                      # AM-settled monthly first (matches the futures' settlement)
        rows = by.get(root)
        if not rows:
            continue
        pts = []
        for K, sides in rows.items():
            c, p = sides.get("C"), sides.get("P")
            if not c or not p or not (c["bid"] and p["bid"] and c["ask"] and p["ask"]):
                continue
            pts.append((K, (c["bid"] + c["ask"]) / 2 - (p["bid"] + p["ask"]) / 2))
        if len(pts) < 5:
            continue
        pts.sort()
        K = np.array([x[0] for x in pts]); cp = np.array([x[1] for x in pts])
        ks = K[np.argmin(np.abs(cp))]
        idx = np.argsort(np.abs(K - ks))[:15]
        b, a = np.polyfit(K[idx], cp[idx], 1)
        D = -b
        if not 0.8 < D < 1.02:
            continue
        return {"F": float(a / D), "D": float(D), "root": root}
    return None


def nasdaq_ndx_history(years: int = 5) -> list[tuple[str, float, float, float, float]]:
    since = (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 10)).isoformat()
    js = json.loads(_get(f"https://api.nasdaq.com/api/quote/NDX/historical?assetclass=index&fromdate={since}&limit=9999", timeout=30, headers=NASDAQ_HEADERS))
    out = []
    for r in js["data"]["tradesTable"]["rows"]:
        f = lambda s: float(str(s).replace(",", "").replace("$", ""))
        try:
            d = dt.datetime.strptime(r["date"], "%m/%d/%Y").date().isoformat()
            out.append((d, f(r["open"]), f(r["high"]), f(r["low"]), f(r["close"])))
        except (ValueError, KeyError):
            continue
    return sorted(out)


def cboe_spx_ohlc(years: int = 5) -> list[tuple[str, float, float, float, float]]:
    js = json.loads(_get("https://cdn.cboe.com/api/global/delayed_quotes/charts/historical/_SPX.json"))
    since = (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 10)).isoformat()
    out = []
    for r in js["data"]:
        if r["date"] >= since and float(r["open"]) > 0:
            out.append((r["date"], float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])))
    return sorted(out)


def calibrate(ohlc: list, iv: dict, last_days: int | None = None) -> dict | None:
    """Close inside +-k sigma and intraday touches of +k / -k sigma, sigma from the prior close's IV.

    Levels are exactly the ladder's: prior close x (1 +- k sigma), sigma = IV / 100 / sqrt(252), so all
    comparisons use simple returns (H / C0 - 1 >= k sigma etc.)."""
    rows = ohlc[-(last_days + 1):] if last_days else ohlc
    ins = np.zeros(len(K_LEVELS)); up = np.zeros(len(K_LEVELS)); dn = np.zeros(len(K_LEVELS))
    n, rv, ivv = 0, 0.0, 0.0
    for prev, cur in zip(rows[:-1], rows[1:]):
        v = iv.get(prev[0])
        if not v or v <= 0:
            continue
        s = v / 100 / math.sqrt(252)
        c0 = prev[4]
        rc, rh, rl = cur[4] / c0 - 1, cur[2] / c0 - 1, cur[3] / c0 - 1
        k = np.array(K_LEVELS) * s
        ins += np.abs(rc) <= k; up += rh >= k; dn += rl <= -k
        n += 1; rv += rc * rc; ivv += s * s
    if n < 50:
        return None
    return {"n": n, "from": rows[0][0], "to": rows[-1][0], "inside": (ins / n).round(4).tolist(),
            "touch_up": (up / n).round(4).tolist(), "touch_dn": (dn / n).round(4).tolist(),
            "rv_over_iv": round(math.sqrt(rv / ivv), 4)}


def open_usable(c0: float, o: float, h: float, l: float) -> bool:
    """Vendor opens that equal the prior close to the cent, or lie outside the day's range, are fill-ins
    (checked against a second source: 20 such NDX sessions in 2021-2026, 1 SPX session)."""
    return abs(o - c0) > 1e-9 and l - 1e-9 <= o <= h + 1e-9


def study_block(spx_ohlc: list, ndx_ohlc: list, iv_hist: dict) -> dict | None:
    """Per-session data for the level study, one row per S&P 500 session: open / high / low / close relative
    to the prior close (basis points, 3 decimals = the indices' own 0.01-point precision) and the prior
    close's implied vols. The browser turns these into sigma units for any volatility index and level."""
    if not spx_ohlc:
        return None
    ndx = {r[0]: r for r in ndx_ohlc}
    dates, cols = [], {"ES": {"o": [], "h": [], "l": [], "c": []}, "NQ": {"o": [], "h": [], "l": [], "c": []}}
    ivs = {k: [] for k in ("VIX", "VIX1D", "VXN", "VIX3M")}
    bad_open = {"ES": 0, "NQ": 0}
    bp = lambda x, c0: round((x / c0 - 1) * 1e4, 3)
    for prev, cur in zip(spx_ohlc[:-1], spx_ohlc[1:]):
        dates.append(cur[0])
        for key, p, c in (("ES", prev, cur), ("NQ", ndx.get(prev[0]), ndx.get(cur[0]))):
            if not p or not c:
                for f in "ohlc":
                    cols[key][f].append(None)
                continue
            c0 = p[4]
            ok = open_usable(c0, c[1], c[2], c[3])
            bad_open[key] += not ok
            cols[key]["o"].append(bp(c[1], c0) if ok else None)
            cols[key]["h"].append(bp(c[2], c0)); cols[key]["l"].append(bp(c[3], c0)); cols[key]["c"].append(bp(c[4], c0))
        for k in ivs:
            v = (iv_hist.get(k) or {}).get(prev[0])
            ivs[k].append(v if v and v > 0 else None)
    return {"dates": dates, **cols, "iv": ivs, "open_excluded": bad_open,
            "note": "returns in basis points vs. the prior close; iv = prior close's index level"}


def build(snap: dict, quotes: dict, ndx_chain: dict | None, hist: dict) -> dict:
    t0 = dt.datetime.fromisoformat(snap["meta"]["valuation_time_et"])
    exp, code = front_quarterly(t0.date())
    spx, spot_spx = snap["spot"]["level"], snap["spot"]["level"]
    e_es = next((e for e in snap["expiries"] if e["expiry"] == exp.isoformat()), None)
    F_es = e_es["F"] if e_es else None
    ndx_q = quotes.get("NDX")
    spot_ndx = float(ndx_q["current_price"]) if ndx_q else None
    nf = parity_forward(ndx_chain["data"]["options"], exp) if ndx_chain else None
    dates, spxc, ndxc = hist["dates"], hist.get("SPX", []), hist.get("NDX", [])
    def prev_close(series):
        vals = [(d, v) for d, v in zip(dates, series) if v is not None and d < t0.date().isoformat()]
        return vals[-1][1] if vals else None
    e0 = snap["expiries"][0] if snap["expiries"] else {}
    contracts = {
        "ES": {"index": "SPX", "label": "E-mini S&P 500", "code": "ES" + code, "expiry": exp.isoformat(), "spot": spot_spx,
               "prev_close": prev_close(spxc), "fair": None if F_es is None else round(F_es, 2),
               "basis": None if F_es is None else round(F_es - spot_spx, 2),
               "vols": {k: float(quotes[k]["current_price"]) for k in ("VIX", "VIX1D") if k in quotes},
               "model_move": None if e0.get("rep_move") is None else round(e0["rep_move"] * 100, 4), "model_expiry": e0.get("expiry")},
        "NQ": {"index": "NDX", "label": "E-mini Nasdaq-100", "code": "NQ" + code, "expiry": exp.isoformat(), "spot": spot_ndx,
               "prev_close": prev_close(ndxc), "fair": None if not nf else round(nf["F"], 2),
               "basis": None if not (nf and spot_ndx) else round(nf["F"] - spot_ndx, 2),
               "vols": {k: float(quotes[k]["current_price"]) for k in ("VXN",) if k in quotes}},
    }
    return {"asof": snap["meta"]["valuation_time_et"], "contracts": contracts, "K": K_LEVELS,
            "theory_inside": [round(2 * norm.cdf(k) - 1, 4) for k in K_LEVELS],
            "theory_touch": [round(2 * (1 - norm.cdf(k)), 4) for k in K_LEVELS]}


def calibration_block(spx_ohlc: list, ndx_ohlc: list, iv_hist: dict) -> dict:
    out = {}
    for key, ohlc, ivname in (("ES_VIX", spx_ohlc, "VIX"), ("ES_VIX1D", spx_ohlc, "VIX1D"), ("NQ_VXN", ndx_ohlc, "VXN")):
        ivm = iv_hist.get(ivname) or {}
        if not ohlc or not ivm:
            continue
        full, last = calibrate(ohlc, ivm), calibrate(ohlc, ivm, last_days=252)
        if full:
            out[key] = {"full": full, "last_year": last}
    return out
