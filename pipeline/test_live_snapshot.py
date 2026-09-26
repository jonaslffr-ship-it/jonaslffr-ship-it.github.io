"""Known-answer tests for live_snapshot.py: a synthetic Black–Scholes chain in Cboe's JSON format.
Run:  python pipeline/test_live_snapshot.py   (exit code 1 on failure; used as a gate in CI)"""
import datetime as dt, math, os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import live_snapshot as ls
import implied_range as ir

S, SIG, R, Q = 5000.0, 0.20, 0.04, 0.012
T0 = dt.datetime(2026, 9, 25, 16, 0)            # Friday 16:00 ET = valuation time
EXPIRIES = [dt.date(2026, 9, 30), dt.date(2026, 10, 30), dt.date(2026, 12, 31), dt.date(2027, 6, 30)]


def bs(K, T, call):
    d1 = (math.log(S / K) + (R - Q + 0.5 * SIG ** 2) * T) / (SIG * math.sqrt(T)); d2 = d1 - SIG * math.sqrt(T)
    if call: return S * math.exp(-Q * T) * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * math.exp(-Q * T) * norm.cdf(-d1)


def synthetic_chain():
    opts = []
    for d in EXPIRIES:
        T = (dt.datetime.combine(d, dt.time(16, 0)) - T0).total_seconds() / ls.YEAR
        lo, hi = S * math.exp(-6 * SIG * math.sqrt(T)), S * math.exp(4.5 * SIG * math.sqrt(T))   # wide enough: no truncation
        for K in np.arange(math.floor(lo / 5) * 5, hi + 0.1, 5.0):
            for cp in "CP":
                px = bs(K, T, cp == "C")
                half = max(0.05, 0.004 * px)
                bid = round(px - half, 2) if px - half >= 0.05 else 0.0
                opts.append({"option": f"SPXW{d:%y%m%d}{cp}{int(K * 1000):08d}", "bid": bid, "ask": round(px + half, 2),
                             "open_interest": 100.0, "volume": 10.0})
    return {"timestamp": "2026-09-25 20:00:00", "data": {"current_price": S, "price_change": 0, "price_change_percent": 0, "iv30": 20,
                                                          "last_trade_time": T0.isoformat(), "options": opts}}


def level_study_checks():
    """Known answer for the implied-range calibration and the level study: sessions of a driftless arithmetic
    Brownian motion whose daily sigma equals the implied one (IV / sqrt 252). Then: close inside +-k sigma ->
    2 Phi(k) - 1, touch -> 2 (1 - Phi(k)) (less a small discrete-monitoring bias), a touched level closes back
    inside half the time, realized / implied = 1, and injected vendor-style bad opens are excluded."""
    rng = np.random.default_rng(7)
    n, steps, iv = 6000, 780, 16.0
    sig = iv / 100 / math.sqrt(252)
    ohlc, c0, d0, iv_map = [], 5000.0, dt.date(2000, 1, 3), {}
    ohlc.append((d0.isoformat(), c0, c0, c0, c0))
    for i in range(1, n + 1):
        path = c0 * (1 + sig * np.cumsum(rng.standard_normal(steps)) / math.sqrt(steps))
        o, h, l, c = float(path[0]), float(path.max()), float(path.min()), float(path[-1])
        if i % 1000 == 1: o = c0                                   # vendor fill-in: open = prior close
        if i % 1000 == 2: o = h + 5.0                              # vendor error: open outside the range
        iv_map[ohlc[-1][0]] = iv
        ohlc.append(((d0 + dt.timedelta(days=i)).isoformat(), o, h, l, c)); c0 = c
    cal = ir.calibrate(ohlc, iv_map)
    st = ir.study_block(ohlc, ohlc, {"VIX": iv_map, "VXN": iv_map})
    fails = []
    K = np.array(ir.K_LEVELS)
    th_in, th_touch = 2 * norm.cdf(K) - 1, 2 * (1 - norm.cdf(K))
    if np.max(np.abs(np.array(cal["inside"]) - th_in)) > 0.02: fails.append("calibration: inside vs normal")
    for side in ("touch_up", "touch_dn"):
        if np.max(np.abs(np.array(cal[side]) - th_touch)) > 0.035: fails.append(f"calibration: {side} vs reflection principle")
    if abs(cal["rv_over_iv"] - 1) > 0.03: fails.append("calibration: realized / implied != 1")
    h = np.array(st["ES"]["h"]) / 1e4; l = np.array(st["ES"]["l"]) / 1e4; c = np.array(st["ES"]["c"]) / 1e4
    for j, k in enumerate(ir.K_LEVELS):
        up, dn = h >= k * sig, l <= -k * sig
        if up.sum() != round(cal["touch_up"][j] * cal["n"]) or dn.sum() != round(cal["touch_dn"][j] * cal["n"]):
            fails.append(f"study vs calibration touch counts at {k} sigma")
        held = (np.sum(up & (c < k * sig)) + np.sum(dn & (c > -k * sig))) / (up.sum() + dn.sum())
        if abs(held - 0.5) > 0.035: fails.append(f"held share at {k} sigma = {held:.3f}, expected 0.5")
        print(f"  level {k:4}σ  inside {cal['inside'][j]:.3f} (th {th_in[j]:.3f})  touch up/dn {cal['touch_up'][j]:.3f}/{cal['touch_dn'][j]:.3f} (th {th_touch[j]:.3f})  held {held:.3f} (th 0.5)")
    if st["open_excluded"]["ES"] != 12: fails.append(f"bad opens excluded: {st['open_excluded']['ES']}, expected 12")
    print(f"  realized/implied {cal['rv_over_iv']:.3f} · n {cal['n']} · bad opens excluded {st['open_excluded']['ES']}")
    return fails


def main():
    quotes = {n: {"current_price": 20.0, "price_change": 0} for n in ("VIX1D", "VIX9D", "VIX", "VIX3M", "VVIX")}
    snap = ls.snapshot(synthetic_chain(), quotes, {"dates": [], "VIX": []})
    fails = []
    assert len(snap["expiries"]) == len(EXPIRIES), f"expected {len(EXPIRIES)} expiries, got {len(snap['expiries'])}"
    for e in snap["expiries"]:
        T = e["T"]; Fth = S * math.exp((R - Q) * T)
        checks = {
            "forward": abs(e["F"] - Fth) < 0.5,
            "rate": abs(e["r"] - R) < 0.002 or T < 0.05,           # short maturities: rate is not identified
            "atm_iv": abs(e["atm_iv"] - SIG) < 0.002,
            "replicated variance": abs(e["rep_var"] / (SIG ** 2 * T) - 1) < 0.02,
            "svi rmse": e["fit_rmse"] < 0.002,
            "butterfly": e["butterfly_ok"],
        }
        bad = [k for k, ok in checks.items() if not ok]
        print(f"{e['expiry']}  T={T:.4f}  F={e['F']:.2f} (th {Fth:.2f})  r={e['r']:.4f}  atm={e['atm_iv']:.4f}  "
              f"repVar/σ²T={e['rep_var'] / (SIG ** 2 * T):.4f}  rmse={e['fit_rmse']:.5f}  {'OK' if not bad else 'FAIL ' + ','.join(bad)}")
        fails += [f"{e['expiry']}: {b}" for b in bad]
    bk = snap["buckets"]["data"]
    tot_oi = sum(v["oi"] for c in bk.values() for v in c.values())
    n_opts = len(synthetic_chain()["data"]["options"])
    if abs(tot_oi - 100.0 * n_opts) > 1: fails.append("bucket OI does not add up")
    print("zero-gamma (A):", snap["gex"]["zero_gamma_A"], "| buckets OI total", tot_oi)
    print("level study (synthetic Brownian sessions):")
    fails += level_study_checks()
    if fails:
        print("FAILED:", fails); sys.exit(1)
    print("all live-snapshot checks passed")


if __name__ == "__main__":
    main()
