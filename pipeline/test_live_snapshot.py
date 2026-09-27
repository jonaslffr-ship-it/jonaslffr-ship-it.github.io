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


def bs(K, T, call, sig=SIG):
    SIG = sig
    d1 = (math.log(S / K) + (R - Q + 0.5 * SIG ** 2) * T) / (SIG * math.sqrt(T)); d2 = d1 - SIG * math.sqrt(T)
    if call: return S * math.exp(-Q * T) * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * math.exp(-Q * T) * norm.cdf(-d1)


def synthetic_chain(root="SPXW", sig_of=lambda i: SIG):
    opts = []
    for i, d in enumerate(EXPIRIES):
        T = (dt.datetime.combine(d, dt.time(16, 0)) - T0).total_seconds() / ls.YEAR
        sg = sig_of(i)
        lo, hi = S * math.exp(-6 * sg * math.sqrt(T)), S * math.exp(4.5 * sg * math.sqrt(T))   # wide enough: no truncation
        for K in np.arange(math.floor(lo / 5) * 5, hi + 0.1, 5.0):
            for cp in "CP":
                px = bs(K, T, cp == "C", sg)
                half = max(0.05, 0.004 * px)
                bid = round(px - half, 2) if px - half >= 0.05 else 0.0
                opts.append({"option": f"{root}{d:%y%m%d}{cp}{int(K * 1000):08d}", "bid": bid, "ask": round(px + half, 2),
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


def term_checks():
    """Term structure and calendar arbitrage: flat vol -> flat, forward vol = sigma, no violations; a falling vol curve
    whose total variance still rises -> backwardation without arbitrage; an inversion strong enough to make total
    variance fall -> flagged. Plus the same known answers for an NDX-style chain (NDXP roots, VXN check)."""
    fails = []
    q = {n: {"current_price": 20.0, "price_change": 0} for n in ("VIX1D", "VIX9D", "VIX", "VIX3M", "VVIX")}
    flat = ls.snapshot(synthetic_chain(), q, None)["term"]
    vols = [x["vol"] for x in flat["tenors"] if x["vol"]]
    fw = [f["atm"] for f in flat["forward"] if f["atm"]]
    print(f"  flat 20 %: tenors {vols} · state {flat['state']} · forward ATM vols {fw} · calendar {flat['calendar']['pairs']} pairs, {len(flat['calendar']['violations'])} violations")
    if not vols or max(abs(v - 20) for v in vols) > 0.5: fails.append("flat: model-free tenors != 20 %")
    if flat["state"] != "flat": fails.append("flat: state")
    if not fw or max(abs(v - 20) for v in fw) > 0.3: fails.append("flat: forward vols != 20 %")
    if flat["calendar"]["violations"]: fails.append("flat: calendar violations")
    back = ls.snapshot(synthetic_chain(sig_of=lambda i: [0.30, 0.26, 0.22, 0.18][i]), q, None)["term"]
    print(f"  backwardation 30/26/22/18 %: state {back['state']} ratio {back['ratio_3m_1m']} · violations {len(back['calendar']['violations'])} · forward ATM {[f['atm'] for f in back['forward']]}")
    if back["state"] != "backwardation": fails.append("backwardation not detected")
    if back["calendar"]["violations"]: fails.append("backwardation wrongly flagged as calendar arbitrage")
    arb = ls.snapshot(synthetic_chain(sig_of=lambda i: [0.20, 0.40, 0.20, 0.20][i]), q, None)["term"]
    bad = [(v["a"], v["b"]) for v in arb["calendar"]["violations"]]
    print(f"  inversion 20/40/20/20 %: violations {bad} · negative ATM forward variances {arb['calendar']['atm_negative_forward']}")
    if (EXPIRIES[1].isoformat(), EXPIRIES[2].isoformat()) not in bad: fails.append("calendar arbitrage (w falls from expiry 2 to 3) not flagged")
    v23 = next(v for v in arb["calendar"]["violations"] if v["a"] == EXPIRIES[1].isoformat())
    T2, T3 = arb["forward"][1]["da"], arb["forward"][1]["db"]
    print(f"  size of the inversion: {v23['vol_pts']} vol pts (fit RMSE {v23['fit_rmse_pts']}) -> material: {not v23['within_fit_error']}; material count {arb['calendar']['material']}")
    if v23["within_fit_error"] or arb["calendar"]["material"] < 1: fails.append("a 20-vol-point inversion must count as material")
    # the check itself on two hand-made SVI slices: identical -> ok; back slice with lower total variance -> flagged
    p = [0.01, 0.1, -0.5, 0.0, 0.1]
    ok = ls.calendar_check([("a", 0.1, p, (-0.2, 0.2)), ("b", 0.2, [0.02, 0.1, -0.5, 0.0, 0.1], (-0.2, 0.2))])
    no = ls.calendar_check([("a", 0.1, p, (-0.2, 0.2)), ("b", 0.2, [0.005, 0.1, -0.5, 0.0, 0.1], (-0.2, 0.2))])
    print(f"  hand-made slices: increasing -> {ok}, decreasing -> {len(no[1])} violation(s)")
    if ok[1] or not no[1]: fails.append("calendar_check on hand-made slices")
    # NDX-style chain: same answers through the NDX configuration
    nq = {"VXN": {"current_price": 20.0, "price_change": 0}}
    nd = ls.snapshot(synthetic_chain(root="NDXP"), nq, None, ix="NDX")
    e = nd["expiries"]; rc = nd["checks"]["replication_vs_cboe"]
    print(f"  NDX config: {len(e)} expiries · ATM IVs {[round(x['atm_iv'], 4) for x in e]} · VXN check {rc} · GEX bin {nd['gex']['bin']}")
    if len(e) != len(EXPIRIES) or max(abs(x["atm_iv"] - SIG) for x in e) > 0.002: fails.append("NDX config: ATM IV")
    if not rc or abs(rc[0]["mine"] - 20) > 0.5: fails.append("NDX config: VXN replication")
    if nd["gex"]["bin"] != 100 or nd["index"] != "NDX": fails.append("NDX config: metadata")
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
    print("term structure, calendar arbitrage, NDX configuration:")
    fails += term_checks()
    if fails:
        print("FAILED:", fails); sys.exit(1)
    print("all live-snapshot checks passed")


if __name__ == "__main__":
    main()
