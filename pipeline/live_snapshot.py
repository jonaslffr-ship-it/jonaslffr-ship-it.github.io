#!/usr/bin/env python3
"""Live SPX volatility snapshot from Cboe delayed quotes (about 15 minutes delayed).

Fetches the SPX/SPXW option chain and the VIX family from Cboe's public delayed-quote
endpoints and publishes DERIVED analytics only:

  * forward, discount factor, implied rate and dividend yield per expiry (put-call parity regression)
  * own Black-76 implied vols from mid quotes, raw-SVI fit per expiry, fit quality
    (share of strikes whose model IV lies inside the bid-ask IV), butterfly check
  * model-free implied variance to each expiry (Cboe VIX methodology incl. zero-bid rule)
  * dealer gamma exposure by strike and across spot under two stated sign conventions, zero-gamma level
  * open interest / volume / premium shares by delta bucket and days-to-expiry class
  * VIX1D / VIX9D / VIX / VIX3M term structure and one year of daily history

Raw option quotes are never written to disk by this script. Set PUBLISH_POINTS = False to
drop the per-strike bid/ask IV points used for the smile charts if Cboe's terms require it.

Usage:  python pipeline/live_snapshot.py [--out data/live] [--chain cached.json]
"""
from __future__ import annotations

import argparse, csv, datetime as dt, io, json, math, os, re, sys, time, urllib.request
from zoneinfo import ZoneInfo

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import implied_range  # noqa: E402  (sibling module)

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr
from scipy.stats import norm

SQ2 = math.sqrt(2.0)
ncdf = lambda x: 0.5 * math.erfc(-x / SQ2)   # fast scalar normal CDF

VERSION = "1.0.0"
PUBLISH_POINTS = True
BASE = "https://cdn-api.cboe.com/api/global/delayed_quotes"
HIST = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
UA = {"User-Agent": "Mozilla/5.0 (volatility research snapshot; github.com/jonaslffr-ship-it)"}
NY = ZoneInfo("America/New_York")
YEAR = 365.0 * 24 * 3600          # calendar-time annualization, as in Cboe's VIX methodology
SYM = re.compile(r"^(SPXW?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
# per-index chain conventions: PM-settled root (weeklies/dailies), AM-settled monthly root, GEX strike bin,
# and the Cboe indices my model-free replication is checked against
INDEX = {
    "SPX": {"re": SYM, "pm": "SPXW", "am": "SPX", "bin": 25, "rep": (("VIX9D", 9), ("VIX", 30), ("VIX3M", 93))},
    "NDX": {"re": re.compile(r"^(NDXP?)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"), "pm": "NDXP", "am": "NDX", "bin": 100, "rep": (("VXN", 30),)},
}
TENORS = (9, 30, 60, 93, 182, 365)       # days: model-free vol term structure (VIX-style interpolation)
BUCKETS = [("Far OTM", 0.0, 0.10), ("OTM", 0.10, 0.40), ("ATM", 0.40, 0.60), ("ITM", 0.60, 0.90), ("Deep ITM", 0.90, 1.01)]
DTE_CLASSES = [("0DTE", 0, 0), ("1–7 d", 1, 7), ("8–45 d", 8, 45), ("> 45 d", 46, 100000)]


# ----------------------------------------------------------------------------- fetch
def get(url: str, tries: int = 3) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read()
        except Exception as e:  # network hiccups: retry with backoff
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
    raise RuntimeError("unreachable")


def quote(sym: str) -> dict:
    return json.loads(get(f"{BASE}/quotes/{sym}.json"))["data"]


def history(name: str, days: int = 260) -> list[tuple[str, float]]:
    rows = list(csv.reader(io.StringIO(get(HIST.format(name)).decode("utf-8", "replace"))))
    out = []
    for r in rows[1:]:
        try:
            d = dt.datetime.strptime(r[0], "%m/%d/%Y").date().isoformat()
            out.append((d, float(r[-1])))
        except (ValueError, IndexError):
            continue
    return out[-days:]


# ----------------------------------------------------------------------------- Black-76 helpers
def b76(F, K, T, D, sig, call):
    v = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return D * (F * norm.cdf(d1) - K * norm.cdf(d2)) if call else D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, D, call):
    intrinsic = D * max((F - K) if call else (K - F), 0.0)
    if not (price > intrinsic + 1e-10) or price >= D * (F if call else K):
        return float("nan")
    f = lambda s: b76(F, K, T, D, s, call) - price
    try:
        return brentq(f, 1e-4, 6.0, xtol=1e-10, maxiter=200)
    except ValueError:
        return float("nan")


def b76_vec(F, K, T, D, sig, call):
    v = sig * math.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    c = D * (F * ndtr(d1) - K * ndtr(d1 - v))
    return np.where(call, c, c - D * (F - K))          # put via parity


def iv_vec(price, F, K, T, D, call):
    """Vectorized Black-76 implied vol by bisection (64 steps -> ~1e-18 bracket); NaN if outside no-arbitrage bounds."""
    price = np.asarray(price, float); K = np.asarray(K, float); call = np.asarray(call, bool)
    intrinsic = D * np.maximum(np.where(call, F - K, K - F), 0.0)
    ok = np.isfinite(price) & (price > intrinsic + 1e-10) & (price < D * np.where(call, F, K))
    target = np.where(ok, price, 0.0)
    lo = np.full(price.shape, 1e-4); hi = np.full(price.shape, 6.0)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        up = b76_vec(F, K, T, D, mid, call) < target
        lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def svi_w(p, k):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def svi_g(p, k):
    a, b, rho, m, s = p
    x = k - m
    R = np.sqrt(x * x + s * s)
    w = a + b * (rho * x + R)
    w1 = b * (rho + x / R)
    w2 = b * s * s / R ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


def fit_svi(k, iv, iv_bid, iv_ask, T):
    """Raw SVI on total variance, residuals in IV units weighted by the bid-ask IV spread."""
    w_obs = iv * iv * T
    spread = np.clip(np.nan_to_num(iv_ask - iv_bid, nan=0.02), 0.003, 0.2)

    def res(p):
        wm = svi_w(p, k)
        pen = max(0.0, -(p[0] + p[1] * p[4] * math.sqrt(1 - p[2] ** 2))) * 1e3   # min variance >= 0
        return np.append((np.sqrt(np.maximum(wm, 1e-12) / T) - iv) / spread, pen)

    wmin, wmax = float(w_obs.min()), float(w_obs.max())
    lo = [-wmax, 1e-6, -0.999, -1.0, 1e-4]
    hi = [wmax, 5.0, 0.999, 1.0, 2.0]
    best = None
    for rho0 in (-0.7, -0.3, 0.0):
        for s0 in (0.05, 0.2):
            x0 = [max(wmin * 0.8, 1e-6), max((wmax - wmin) / 0.5, 1e-3), rho0, float(k[np.argmin(w_obs)]), s0]
            x0 = [min(max(x0[i], lo[i] + 1e-9), hi[i] - 1e-9) for i in range(5)]
            try:
                r = least_squares(res, x0, bounds=(lo, hi), max_nfev=4000)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    return None if best is None else best.x


# ----------------------------------------------------------------------------- core
def replication_check(exps, quotes, tenors=(("VIX9D", 9), ("VIX", 30), ("VIX3M", 93))):
    """Interpolate my model-free total variance to the index tenors (VIX method: bracketing PM expiries) and compare."""
    E = [e for e in exps if e["rep_var"] and e["settle"] == "PM"]
    out = []
    for name, days in tenors:
        if name not in quotes: continue
        t = days / 365
        lo = [e for e in E if e["T"] <= t]; hi = [e for e in E if e["T"] > t]
        if not lo or not hi: continue
        a, b = max(lo, key=lambda e: e["T"]), min(hi, key=lambda e: e["T"])
        v = (a["rep_var"] * (b["T"] - t) + b["rep_var"] * (t - a["T"])) / (b["T"] - a["T"]) / t
        out.append({"index": name, "days": days, "mine": round(math.sqrt(v) * 100, 2), "cboe": float(quotes[name]["current_price"])})
    return out


def mf_vol_at(exps, days):
    """Model-free vol (%) at a fixed tenor: linear interpolation of total variance between the bracketing
    PM expiries, exactly as the VIX interpolates its two near-term expiries."""
    E = sorted([e for e in exps if e["rep_var"] and e["rep_var"] > 0 and e["settle"] == "PM"], key=lambda e: e["T"])
    t = days / 365
    lo = [e for e in E if e["T"] <= t]; hi = [e for e in E if e["T"] > t]
    if not lo or not hi:
        return None
    a, b = lo[-1], hi[0]
    w = (a["rep_var"] * (b["T"] - t) + b["rep_var"] * (t - a["T"])) / (b["T"] - a["T"])
    return round(math.sqrt(w / t) * 100, 3) if w > 0 else None


def calendar_check(slices, tol=1e-6, n=61):
    """No calendar arbitrage <=> total implied variance w(k, T) is non-decreasing in T at every forward
    log-moneyness k (Gatheral & Jacquier 2014). slices: [(label, T, svi_params, (k_lo, k_hi))], checked for each
    consecutive pair on the strikes both expiries actually quote. Returns (pairs checked, violations).
    Each violation carries its size in vol points (how far the back expiry's IV at that strike would have to rise)
    and whether that is within the two fits' own RMSE, i.e. indistinguishable from fit noise."""
    S_ = sorted(slices, key=lambda x: x[1]); out, pairs = [], 0
    for A_, B_ in zip(S_[:-1], S_[1:]):
        (la, Ta, pa, ra), (lb, Tb, pb, rb) = A_[:4], B_[:4]
        k0, k1 = max(ra[0], rb[0]), min(ra[1], rb[1])
        if k1 <= k0 or Tb <= Ta:
            continue
        pairs += 1
        kg = np.linspace(k0, k1, n); wa, wb = svi_w(pa, kg), svi_w(pb, kg); dw = wb - wa; i = int(np.argmin(dw))
        if dw[i] < -tol:
            vp = (math.sqrt(max(wa[i], 0) / Tb) - math.sqrt(max(wb[i], 0) / Tb)) * 100
            noise = max(A_[4] if len(A_) > 4 else 0, B_[4] if len(B_) > 4 else 0) * 100
            out.append({"a": la, "b": lb, "k": round(float(kg[i]), 4), "dw": round(float(dw[i]), 7),
                        "share": round(float(np.mean(dw < -tol)), 3), "vol_pts": round(vp, 3), "fit_rmse_pts": round(noise, 3),
                        "within_fit_error": bool(vp <= noise)})
    return pairs, out


def term_block(exps):
    """Contango / backwardation of the model-free term structure, forward vols between consecutive expiries
    and the calendar-arbitrage check on the fitted smiles."""
    tenors = [{"days": d_, "vol": mf_vol_at(exps, d_)} for d_ in TENORS]
    tv = {x["days"]: x["vol"] for x in tenors}
    ratio = round(tv[93] / tv[30], 4) if tv.get(93) and tv.get(30) else None
    state = None if ratio is None else "contango" if ratio >= 1.01 else "backwardation" if ratio <= 0.99 else "flat"
    E = sorted(exps, key=lambda e: e["T"]); fwd = []
    for a, b in zip(E[:-1], E[1:]):
        wa, wb = a["atm_iv"] ** 2 * a["T"], b["atm_iv"] ** 2 * b["T"]
        fv = (wb - wa) / (b["T"] - a["T"]) if b["T"] > a["T"] else None
        mfv = None
        if a["rep_var"] and b["rep_var"] and b["T"] > a["T"]:
            mfv = (b["rep_var"] - a["rep_var"]) / (b["T"] - a["T"])
        fwd.append({"a": a["expiry"], "b": b["expiry"], "da": a["days"], "db": b["days"],
                    "atm": None if fv is None or fv <= 0 else round(math.sqrt(fv) * 100, 3), "atm_var": None if fv is None else round(fv, 6),
                    "mf": None if mfv is None or mfv <= 0 else round(math.sqrt(mfv) * 100, 3)})
    good = [e for e in E if e["fit_rmse"] <= 0.015]
    pairs, viol = calendar_check([(e["expiry"], e["T"], e["svi"], tuple(e["k_range"]), e["fit_rmse"]) for e in good])
    return {"tenors": tenors, "ratio_3m_1m": ratio, "state": state, "forward": fwd,
            "calendar": {"pairs": pairs, "violations": viol, "material": sum(1 for v in viol if not v["within_fit_error"]), "atm_negative_forward": sum(1 for f in fwd if f["atm_var"] is not None and f["atm_var"] < 0),
                         "slices": len(good), "note": "fitted SVI smiles, strikes quoted by both expiries; w(k,T) must not fall with T"}}


def market_open(now_et: dt.datetime) -> bool:
    return now_et.weekday() < 5 and dt.time(9, 30) <= now_et.time() <= dt.time(16, 15)


def snapshot(chain: dict, quotes: dict, hist: dict | None, ix: str = "SPX") -> dict:
    cfg = INDEX[ix]
    now = dt.datetime.now(dt.timezone.utc)
    spx = chain["data"]
    S = float(spx["current_price"])
    t0 = dt.datetime.fromisoformat(spx["last_trade_time"]).replace(tzinfo=NY)  # valuation time = last SPX print
    t0_date = t0.date()

    # group the chain by expiry; prefer PM-settled SPXW, keep AM-settled SPX only where no SPXW exists
    by = {}
    for o in chain["data"]["options"]:
        m = cfg["re"].match(o["option"])
        if not m:
            continue
        root, yy, mm, dd, cp, kk = m.groups()
        d = dt.date(2000 + int(yy), int(mm), int(dd))
        by.setdefault(d, {}).setdefault(root, []).append((cp, int(kk) / 1000.0, o))
    expiries = []
    for d in sorted(by):
        root = cfg["pm"] if cfg["pm"] in by[d] else cfg["am"]
        if root not in by[d]:
            continue
        t_exp = dt.datetime.combine(d, dt.time(16, 0) if root == cfg["pm"] else dt.time(9, 30), NY)
        T = (t_exp - t0).total_seconds() / YEAR
        if T <= 1 / (365 * 24):                        # < 1 hour left: skip
            continue
        rows = {}
        for cp, K, o in by[d][root]:
            rows.setdefault(K, {})[cp] = o
        extra = None                                   # AM-settled monthlies on a date that also has a PM expiry:
        if root == cfg["pm"] and cfg["am"] in by[d]:   # not used for the surface fit, but their open interest counts
            T_am = (dt.datetime.combine(d, dt.time(9, 30), NY) - t0).total_seconds() / YEAR
            if T_am > 1 / (365 * 24):
                am = {}
                for cp, K, o in by[d][cfg["am"]]:
                    am.setdefault(K, {})[cp] = o
                extra = (am, T_am)
        expiries.append((d, root, T, rows, extra))

    out_exp, surface, smiles = [], [], []
    gex_parts = []          # (K, T, r, q, sigma(K), OIc, OIp)
    buckets = {c[0]: {b[0]: {"oi": 0.0, "vol": 0.0, "prem": 0.0} for b in BUCKETS} for c in DTE_CLASSES}
    for d, root, T, rows, extra in expiries:
        Ks = np.array(sorted(rows))
        def mid(o):
            if not o or o["bid"] is None or o["ask"] is None: return float("nan")
            b, a = float(o["bid"]), float(o["ask"])
            return (a + b) / 2 if a > 0 and a >= b else float("nan")
        C = np.array([mid(rows[K].get("C")) for K in Ks]); P = np.array([mid(rows[K].get("P")) for K in Ks])
        both = np.isfinite(C) & np.isfinite(P)
        if both.sum() < 5:
            continue
        # forward & discount from put-call parity: C - P = D (F - K), regression on strikes nearest the money
        kstar = Ks[both][np.argmin(np.abs((C - P)[both]))]
        idx = np.where(both)[0]; idx = idx[np.argsort(np.abs(Ks[idx] - kstar))][:15]
        slope, icpt = np.polyfit(Ks[idx], (C - P)[idx], 1)
        D = -slope
        parity_ok = math.exp(-0.15 * T) < D < math.exp(0.02 * T) + 1e-3   # rates between -2 % and 15 %
        if parity_ok:
            F = icpt / D
        else:                                            # regression unusable: parity at the single strike, D = 1
            D, F = 1.0, float(kstar + (C - P)[Ks == kstar][0])
        if not (0.5 * S < F < 1.5 * S):
            continue
        r = -math.log(D) / T; q = r - math.log(F / S) / T

        # OTM implied vols (mid, bid, ask) — puts below F, calls above
        cm = Ks >= F
        side = lambda K, c: rows[K].get("C" if c else "P") or {}
        bids = np.array([float(side(K, c).get("bid") or 0) for K, c in zip(Ks, cm)])
        asks = np.array([float(side(K, c).get("ask") or 0) for K, c in zip(Ks, cm)])
        mids = np.where(cm, C, P)
        sel = (bids > 0) & np.isfinite(mids)
        if sel.sum() < 8:
            continue
        iv_m = iv_vec(mids[sel], F, Ks[sel], T, D, cm[sel])
        iv_b = iv_vec(bids[sel], F, Ks[sel], T, D, cm[sel]); iv_a = iv_vec(asks[sel], F, Ks[sel], T, D, cm[sel])
        good = np.isfinite(iv_m)
        pts = np.column_stack([np.log(Ks[sel] / F), iv_m, iv_b, iv_a, Ks[sel]])[good]
        if len(pts) < 8:
            continue
        atm_guess = float(np.interp(0.0, pts[:, 0], pts[:, 1]))
        keep = (pts[:, 0] > -7 * atm_guess * math.sqrt(T) - 0.02) & (pts[:, 0] < 4 * atm_guess * math.sqrt(T) + 0.02)
        pts = pts[keep]
        if len(pts) < 8:
            continue
        k, iv, ivb, iva = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        p = fit_svi(k, iv, ivb, iva, T)
        if p is None:
            continue
        iv_fit = np.sqrt(np.maximum(svi_w(p, k), 1e-12) / T)
        inside = float(np.mean((iv_fit >= np.nan_to_num(ivb, nan=0.0) - 1e-9) & (iv_fit <= np.nan_to_num(iva, nan=9.0) + 1e-9)))
        rmse = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))
        kg = np.linspace(k.min() - 0.05, k.max() + 0.05, 200)
        bfly_ok = bool(np.all(svi_g(p, kg) >= -1e-9))
        atm_iv = float(math.sqrt(max(svi_w(p, 0.0), 1e-12) / T))

        # model-free implied variance (Cboe VIX methodology, zero-bid truncation)
        K0 = Ks[Ks <= F].max() if np.any(Ks <= F) else Ks.min()
        def q_at(K, side):
            o = rows[K].get(side); m_ = mid(o)
            return (m_, float(o["bid"]) if o and o["bid"] else 0.0)
        use = {}
        i0 = int(np.where(Ks == K0)[0][0])
        cm, _ = q_at(K0, "C"); pm, _ = q_at(K0, "P")
        if np.isfinite(cm) and np.isfinite(pm):
            use[K0] = (cm + pm) / 2
            for step, side in ((-1, "P"), (1, "C")):
                zeros, i = 0, i0 + step
                while 0 <= i < len(Ks):
                    m_, bid = q_at(Ks[i], side)
                    if bid <= 0 or not np.isfinite(m_):
                        zeros += 1
                        if zeros >= 2: break
                    else:
                        zeros = 0; use[Ks[i]] = m_
                    i += step
        rep_var = float("nan")
        if len(use) >= 5:
            uk = np.array(sorted(use)); uq = np.array([use[x] for x in uk])
            dK = np.gradient(uk)
            rep_var = float(2 / D * np.sum(dK / uk ** 2 * uq) - (F / K0 - 1) ** 2)
        straddle = float(np.interp(F, Ks[both], (C + P)[both]))
        days = (d - t0_date).days
        out_exp.append({
            "expiry": d.isoformat(), "root": root, "settle": "PM" if root == cfg["pm"] else "AM", "days": days, "T": round(T, 6),
            "F": round(F, 2), "D": round(D, 6), "r": round(r, 4), "q": round(q, 4), "atm_iv": round(atm_iv, 5),
            "rep_var": round(rep_var, 8) if np.isfinite(rep_var) else None,
            "rep_move": round(math.sqrt(rep_var), 6) if np.isfinite(rep_var) and rep_var > 0 else None,
            "straddle_move": round(straddle / (math.sqrt(2 / math.pi) * F), 6),
            "svi": [round(float(x), 6) for x in p], "fit_inside": round(inside, 3), "fit_rmse": round(rmse, 5), "n": int(len(k)), "parity_ok": bool(parity_ok),
            "k_range": [round(float(k.min()), 4), round(float(k.max()), 4)], "butterfly_ok": bfly_ok,
        })
        if PUBLISH_POINTS:
            smiles.append({"expiry": d.isoformat(), "days": days,
                           "pts": [[round(float(math.exp(a) * 100), 2), round(float(b) * 100, 3),
                                    None if not np.isfinite(c) else round(float(c) * 100, 3),
                                    None if not np.isfinite(e) else round(float(e) * 100, 3)] for a, b, c, e in zip(k, iv, ivb, iva)]})

        # per-strike inputs for gamma exposure and moneyness buckets (sigma from the SVI fit, clamped to quoted range);
        # includes AM-settled monthlies that share the date, valued with their own (shorter) time to settlement
        cls = next(c[0] for c in DTE_CLASSES if c[1] <= days <= c[2])
        for rws, TT in [(rows, T)] + ([extra] if extra else []):
          for K in sorted(rws):
            kk = min(max(math.log(K / F), k.min()), k.max())
            sig = math.sqrt(max(svi_w(p, kk), 1e-10) / T)
            oc, op = rws[K].get("C"), rws[K].get("P")
            oic = float(oc["open_interest"] or 0) if oc else 0.0
            oip = float(op["open_interest"] or 0) if op else 0.0
            gex_parts.append((K, TT, r, q, sig, oic, oip))
            v = sig * math.sqrt(TT); d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * TT) / v
            for o, dl in ((oc, math.exp(-q * TT) * ncdf(d1)), (op, -math.exp(-q * TT) * ncdf(-d1))):
                if not o or not math.isfinite(dl): continue
                ad = min(abs(dl), 1.0); b = next(x[0] for x in BUCKETS if x[1] <= ad < x[2])
                cell = buckets[cls][b]
                cell["oi"] += float(o["open_interest"] or 0); cell["vol"] += float(o["volume"] or 0)
                m_ = mid(o)
                if np.isfinite(m_): cell["prem"] += float(o["volume"] or 0) * m_ * 100

    # ---- dealer gamma exposure (two conventions), by strike at spot and profile across spot
    G = np.array(gex_parts)
    Kx, Tx, rx, qx, sx, oic, oip = (G[:, i] for i in range(7))
    def gamma_at(Sp):
        v = sx * np.sqrt(Tx); d1 = (np.log(Sp / Kx) + (rx - qx + 0.5 * sx * sx) * Tx) / v
        return np.exp(-qx * Tx) * norm.pdf(d1) / (Sp * v)
    def gex(Sp, conv):
        g = gamma_at(Sp) * 100 * Sp * Sp * 0.01          # $ per 1 % move per contract
        return g * (oic - oip) if conv == "A" else -g * (oic + oip)
    grid = S * (1 + np.arange(-8.0, 5.01, 0.25) / 100)
    prof = {c: [float(gex(s_, c).sum()) for s_ in grid] for c in ("A", "B")}
    zero = None
    pa = prof["A"]
    for i in range(1, len(grid)):
        if (pa[i - 1] < 0) != (pa[i] < 0):
            z = grid[i - 1] + (grid[i] - grid[i - 1]) * (-pa[i - 1]) / (pa[i] - pa[i - 1])
            if zero is None or abs(z - S) < abs(zero - S): zero = z
    near = (Kx >= S * 0.9) & (Kx <= S * 1.08)
    gA, gB = gex(S, "A"), gex(S, "B")
    kbin = np.round(Kx / cfg["bin"]) * cfg["bin"]        # aggregate strikes into bins (SPX 25, NDX 100 points)
    by_strike = [[float(b_), float(gA[near & (kbin == b_)].sum()), float(gB[near & (kbin == b_)].sum())] for b_ in np.unique(kbin[near])]

    # ---- surface grid (SVI, inside each expiry's quoted range only)
    kf = np.linspace(80, 110, 31)
    surf_exp = [e for e in out_exp if e["days"] <= 400 and e["fit_rmse"] <= 0.015]   # well-fitted slices only (RMSE <= 1.5 vol pts)
    for e in surf_exp:
        kk = np.log(kf / 100)
        z = np.sqrt(np.maximum(svi_w(e["svi"], kk), 1e-12) / e["T"]) * 100
        z[(kk < e["k_range"][0] - 0.02) | (kk > e["k_range"][1] + 0.02)] = np.nan
        surface.append([None if not np.isfinite(v) else round(float(v), 3) for v in z])
    # calendar check on ATM total variance
    atm_w = [e["atm_iv"] ** 2 * e["T"] for e in out_exp]
    cal_pairs = [(out_exp[i - 1]["expiry"], out_exp[i]["expiry"]) for i in range(1, len(atm_w)) if atm_w[i] < atm_w[i - 1] - 1e-9]
    cal_viol = len(cal_pairs)

    # ---- pick smiles to publish: nearest expiry and the ones closest to 7/30/91/182/365 days
    chosen = []
    for target in (0, 7, 30, 91, 182, 365):
        cand = min(out_exp, key=lambda e: abs(e["days"] - target)) if out_exp else None
        if cand and cand["expiry"] not in chosen: chosen.append(cand["expiry"])
    smiles = [s_ for s_ in smiles if s_["expiry"] in chosen]

    vix = {k_: {"level": float(v["current_price"]), "change": float(v.get("price_change") or 0)} for k_, v in quotes.items()}
    now_et = now.astimezone(NY)
    return {
        "index": ix,
        "meta": {
            "version": VERSION, "generated_utc": now.isoformat(timespec="seconds"), "cboe_timestamp": chain.get("timestamp"),
            "valuation_time_et": t0.isoformat(timespec="minutes"), "market_open": market_open(now_et),
            "source": "Cboe delayed quotes (about 15 min delayed) and Cboe index history; derived analytics only",
            "git_sha": os.environ.get("GITHUB_SHA", "")[:12], "time_convention": "calendar time (minutes / 525,600), as in the VIX",
        },
        "spot": {"level": S, "change": float(spx.get("price_change") or 0), "change_pct": float(spx.get("price_change_percent") or 0), "iv30": float(spx.get("iv30") or 0)},
        "vix": vix,
        "expiries": out_exp,
        "surface": {"kf": [float(x) for x in kf], "expiry": [e["expiry"] for e in surf_exp], "days": [e["days"] for e in surf_exp], "T": [e["T"] for e in surf_exp], "iv": surface},
        "smiles": smiles,
        "checks": {"butterfly_violations": sum(1 for e in out_exp if not e["butterfly_ok"]), "calendar_violations_atm": cal_viol, "calendar_pairs": cal_pairs, "expiries": len(out_exp),
                   "replication_vs_cboe": replication_check(out_exp, quotes, cfg["rep"])},
        "term": term_block(out_exp),
        "gex": {"spot_grid": [round(float(x), 2) for x in grid], "A": [round(x) for x in prof["A"]], "B": [round(x) for x in prof["B"]],
                "zero_gamma_A": None if zero is None else round(float(zero), 2), "by_strike": [[k_, round(a_), round(b_)] for k_, a_, b_ in by_strike], "bin": cfg["bin"],
                "note": "A: dealers long calls, short puts. B: dealers short all options. Open interest is from the prior day (OCC)."},
        "buckets": {"classes": [c[0] for c in DTE_CLASSES], "buckets": [b[0] for b in BUCKETS],
                    "data": {c: {b: {k_: round(v, 0) for k_, v in buckets[c][b].items()} for b in buckets[c]} for c in buckets}},
        "history": hist,
    }


SERIES = ("SPX", "VIX1D", "VIX9D", "VIX", "VIX3M", "VVIX", "VXN")    # Cboe daily history CSVs
BAR_SERIES = SERIES + ("NDX",)                                   # Cboe 1-minute bars
INTRADAY_SESSIONS = 5


def point_from(snap: dict) -> dict:
    """One observation of every series this site charts (index levels + my own derived measures)."""
    e0 = snap["expiries"][0] if snap["expiries"] else {}
    v30 = next((c["mine"] for c in snap["checks"]["replication_vs_cboe"] if c["index"] == "VIX"), None)
    pt = {"SPX": round(snap["spot"]["level"], 2), **{k: v["level"] for k, v in snap["vix"].items()},
          "move": None if e0.get("rep_move") is None else round(e0["rep_move"] * 100, 4),
          "v30": v30, "zg": snap["gex"]["zero_gamma_A"]}
    return pt


def update_intraday(path: str, snap: dict) -> list:
    """Rolling intraday record (last INTRADAY_SESSIONS sessions). In CI the previous file is fetched from the
    deployed site first, so the record grows without committing every 30-minute snapshot."""
    rows = []
    if os.path.exists(path):
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            rows = []
    t = snap["meta"]["valuation_time_et"]
    if not rows or rows[-1]["t"] != t:
        rows.append({"t": t, **point_from(snap)})
    rows.sort(key=lambda r: r["t"])
    keep = sorted({r["t"][:10] for r in rows})[-INTRADAY_SESSIONS:]
    rows = [r for r in rows if r["t"][:10] in keep]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, separators=(",", ":"))
    return rows


def compile_archive(hdir: str) -> list:
    out = []
    if os.path.isdir(hdir):
        for fn in sorted(os.listdir(hdir)):
            if fn.endswith(".json"):
                try:
                    out.append(json.load(open(os.path.join(hdir, fn), encoding="utf-8")))
                except (ValueError, OSError):
                    continue
    return [{"d": r["date"], "SPX": r.get("spot"), **(r.get("vix") or {}), "move": None if r.get("next_move") is None else round(r["next_move"] * 100, 4),
             "v30": r.get("v30"), "zg": r.get("zero_gamma_A")} for r in out]


CHARTS = "https://cdn.cboe.com/api/global/delayed_quotes/charts/intraday/_{}.json"
BAR_SESSIONS = 5


def fetch_bars() -> dict | None:
    """Latest session's 1-minute bars for all six series from Cboe, sampled every 5 minutes on a fixed
    09:35-16:00 ET grid (last non-zero close at or before each slot; the first vol-index bar is often 0)."""
    slots = [f"{9 + (35 + 5 * i) // 60:02d}:{(35 + 5 * i) % 60:02d}" for i in range(78)]
    out, day = {"t": slots}, None
    for name in BAR_SERIES:
        try:
            rows = json.loads(get(CHARTS.format(name)))["data"]
        except Exception:
            rows = []
        pts = sorted((r["datetime"][11:16], float(r["price"]["close"])) for r in rows if r.get("price") and r["price"].get("close"))
        if not pts:
            if name == "SPX":
                return None
            out[name] = [None] * len(slots); continue
        day = day or rows[0]["datetime"][:10]
        vals, j, last = [], 0, None
        for sl in slots:
            while j < len(pts) and pts[j][0] <= sl:
                last = pts[j][1]; j += 1
            vals.append(None if last is None else round(last, 2))
        out[name] = vals
        if name in ("SPX", "NDX"):          # true 5-minute extremes from the 1-minute highs / lows (touch detection)
            hl = sorted((r["datetime"][11:16], float(r["price"]["high"]), float(r["price"]["low"])) for r in rows
                        if r.get("price") and r["price"].get("high") and r["price"].get("low") and float(r["price"]["low"]) > 0)
            hi, lo, j, prev = [], [], 0, "00:00"
            for sl in slots:
                bh, bl = None, None
                while j < len(hl) and hl[j][0] <= sl:
                    if hl[j][0] > prev:
                        bh = hl[j][1] if bh is None else max(bh, hl[j][1]); bl = hl[j][2] if bl is None else min(bl, hl[j][2])
                    j += 1
                hi.append(None if bh is None else round(bh, 2)); lo.append(None if bl is None else round(bl, 2)); prev = sl
            out[name + "_h"], out[name + "_l"] = hi, lo
    return {"date": day, **out}


def update_bars(path: str) -> list:
    sessions = []
    if os.path.exists(path):
        try:
            sessions = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError):
            sessions = []
    try:
        cur = fetch_bars()
    except Exception as e:  # keep the restored sessions if Cboe's chart endpoint hiccups
        print("intraday bars unavailable:", e); cur = None
    if cur:
        sessions = [x for x in sessions if x["date"] != cur["date"]] + [cur]
    sessions = sorted(sessions, key=lambda x: x["date"])[-BAR_SESSIONS:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, separators=(",", ":"))
    return sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "live"))
    ap.add_argument("--chain", help="use a cached chain JSON instead of downloading (testing)")
    a = ap.parse_args()
    chain = json.load(open(a.chain, encoding="utf-8")) if a.chain else json.loads(get(f"{BASE}/options/_SPX.json"))
    quotes = {s.strip("_"): quote(s) for s in ("_VIX1D", "_VIX9D", "_VIX", "_VIX3M", "_VVIX", "_VXN", "_NDX")}
    # five years of daily closes, aligned on S&P 500 trading days (VIX1D exists only since 2022)
    hs = {n: dict(history(n, days=1400)) for n in SERIES}
    try:
        ndx_ohlc = implied_range.nasdaq_ndx_history()           # Nasdaq's own API (Cboe does not serve NDX history)
    except Exception as e:
        print("NDX history unavailable:", e); ndx_ohlc = []
    hs["NDX"] = {r[0]: r[4] for r in ndx_ohlc}
    dates = sorted(hs["SPX"])[-1260:]
    hist = {"dates": dates, **{n: [hs[n].get(d) for d in dates] for n in (*SERIES, "NDX")}}
    snap = snapshot(chain, quotes, hist)
    try:
        ndx_chain = json.loads(get(f"{BASE}/options/_NDX.json"))
    except Exception as e:
        print("NDX chain unavailable:", e); ndx_chain = None
    ndx_snap = None
    if ndx_chain:
        try:
            ndx_snap = snapshot(ndx_chain, {k: quotes[k] for k in ("VXN",) if k in quotes}, None, ix="NDX")
            ndx_snap.pop("history", None)
        except Exception as e:  # the SPX page must not fail because of the second index
            print("NDX snapshot failed:", e); ndx_snap = None
    try:
        spx_ohlc = implied_range.cboe_spx_ohlc()
    except Exception as e:
        print("SPX OHLC unavailable:", e); spx_ohlc = []
    rng = implied_range.build(snap, quotes, ndx_chain, hist, ndx_snap)
    rng["calibration"] = implied_range.calibration_block(spx_ohlc, ndx_ohlc, hs)
    snap["range"] = rng
    os.makedirs(a.out, exist_ok=True)
    study = implied_range.study_block(spx_ohlc, ndx_ohlc, hs)
    if study:
        sb = json.dumps({"generated_utc": snap["meta"]["generated_utc"], **study}, separators=(",", ":"), allow_nan=False)
        with open(os.path.join(a.out, "study.js"), "w", encoding="utf-8") as f: f.write("window.__STUDY__=" + sb + ";" + chr(10))
    hdir = os.path.join(a.out, "history")
    # end-of-day archive (small summary), one file per trading date
    now_et = dt.datetime.now(NY)
    if snap["meta"]["valuation_time_et"][:10] == now_et.date().isoformat() and now_et.time() >= dt.time(16, 15):
        day = snap["meta"]["valuation_time_et"][:10]
        os.makedirs(hdir, exist_ok=True)
        e0 = snap["expiries"][0] if snap["expiries"] else {}
        pt = point_from(snap)
        summ = {"date": day, "spot": snap["spot"]["level"], "vix": {k_: v["level"] for k_, v in snap["vix"].items()},
                "next_expiry": e0.get("expiry"), "next_move": e0.get("rep_move"), "zero_gamma_A": snap["gex"]["zero_gamma_A"],
                "v30": pt["v30"], "atm_term": [[e["days"], e["atm_iv"]] for e in snap["expiries"] if e["days"] <= 400]}
        with open(os.path.join(hdir, f"{day}.json"), "w", encoding="utf-8") as f: json.dump(summ, f, separators=(",", ":"))
    if ndx_snap:
        nb = json.dumps(ndx_snap, separators=(",", ":"), allow_nan=False)
        with open(os.path.join(a.out, "ndx.js"), "w", encoding="utf-8") as f: f.write("window.__NDX__=" + nb + ";" + chr(10))
        snap["ndx_meta"] = {"generated_utc": ndx_snap["meta"]["generated_utc"], "valuation_time_et": ndx_snap["meta"]["valuation_time_et"], "expiries": len(ndx_snap["expiries"])}
    snap["intraday"] = update_intraday(os.path.join(a.out, "intraday.json"), snap)
    snap["bars"] = update_bars(os.path.join(a.out, "bars.json"))
    snap["archive"] = compile_archive(hdir)
    body = json.dumps(snap, separators=(",", ":"), allow_nan=False)
    with open(os.path.join(a.out, "latest.json"), "w", encoding="utf-8") as f: f.write(body)
    with open(os.path.join(a.out, "latest.js"), "w", encoding="utf-8") as f: f.write("window.__LIVE__=" + body + ";" + chr(10))
    e0 = snap["expiries"][0] if snap["expiries"] else {}
    print(f"SPX {snap['spot']['level']:.2f} | expiries {len(snap['expiries'])} | next {e0.get('expiry')} move ±{(e0.get('rep_move') or 0)*100:.2f}% "
          f"| zero-gamma(A) {snap['gex']['zero_gamma_A']} | intraday {len(snap['intraday'])} pts | bars {len(snap['bars'])} sessions | archive {len(snap['archive'])} d | range ES {snap['range']['contracts']['ES']['fair']} NQ {snap['range']['contracts']['NQ']['fair']} | calib {list(snap['range']['calibration'])} | NDX {'ok ' + str(len(ndx_snap['expiries'])) + ' expiries, VXN check ' + str(ndx_snap['checks']['replication_vs_cboe']) if ndx_snap else 'n/a'} | term {snap['term']['state']} cal-viol {len(snap['term']['calendar']['violations'])} | {len(body)/1024:.0f} KB")


if __name__ == "__main__":
    main()
