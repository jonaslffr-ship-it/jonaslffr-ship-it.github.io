"""Known-answer tests for live_snapshot.py: a synthetic Black–Scholes chain in Cboe's JSON format.
Run:  python pipeline/test_live_snapshot.py   (exit code 1 on failure; used as a gate in CI)"""
import datetime as dt, math, os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import live_snapshot as ls

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
    if fails:
        print("FAILED:", fails); sys.exit(1)
    print("all live-snapshot checks passed")


if __name__ == "__main__":
    main()
