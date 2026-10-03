# jonaslffr-ship-it.github.io

Personal research site of Jonas Löffler — volatility forecasting, options and honest backtesting.

**Live:** https://jonaslffr-ship-it.github.io/

## What is here

A single, dependency-free `index.html` (no build step): a dark-mode app with tabs (Apple HIG + Material 3 design language). Every tool runs in the browser on simulated or parametric data; the 3D surfaces use a small custom canvas renderer (no WebGL library).

| Tab | What it does |
|---|---|
| Overview | Positioning, live market strip (SPX, VIX family, implied move), research highlights, tool launcher, research log |
| Live Market | Real S&P 500 **and Nasdaq-100** data (Cboe delayed quotes, ~15 min), switchable: SVI surface in 3D with the at-the-money line, smiles with a smoothing spline through the quotes against the five-parameter SVI fit (share of strikes inside the bid–ask IV for both), model-free implied moves checked against VIX9D/VIX/VIX3M or VXN; term structure (contango / backwardation from the model-free curve, forward vols between expiries, calendar-arbitrage check on the fitted smiles with a size-vs-fit-error verdict); dealer gamma under two conventions with zero-gamma level, moneyness shares, per-expiry forward/rates |
| Implied Range | ES & NQ σ-ladder (0.25σ–2σ) from VIX, VIX1D, VXN or the chain’s model-free move, anchored at the futures’ fair value (index + put–call-parity basis), prior close or a custom price, with a live formula of what each control changes; the last five sessions against their implied range (5-minute highs/lows, first touch of every level, drag-to-zoom, enlarge, auto-update while the market is open); the math in eight steps (√time, reflection principle, the 50 % benchmark, parity, VRP) with a random-walk simulator; five-year calibration; a level lab (touched → held or broke, Wilson intervals, regime / term-structure / weekday filters, gap opens excluded) and the distribution of daily moves in σ and implied vs. subsequently realized vol — all recomputed in the browser from a per-session file |
| Vol Surface | SSVI surface (Gatheral & Jacquier 2014) in 3D — implied vol, total variance, Dupire local vol; butterfly / calendar / Gatheral–Jacquier checks; smiles, ATM term structure, risk-neutral density |
| Options & Greeks | Strategy builder with a term-structure slider for calendars and diagonals (back − front vol, sticky expiry) and a live calendar-arbitrage check (w = σ²T must rise with maturity); today’s spot drawn as a line on every 3D surface; (30 strategies in seven groups — incl. broken-wing butterflies, jade lizard, ratio/backspreads, calendars/diagonals, 0DTE structures and stock + options such as covered call or collar; editable legs incl. an underlying leg); P&L and 13 Black–Scholes greeks to third order as 3D surfaces over spot × vol or spot × time; per-leg finite-difference check |
| Dealer Gamma | Toy dealer-hedging feedback σ_eff = σ₀ / (1 + λΓ_D): gamma flip, GEX by strike, vol multiplier, Monte Carlo paths with common random numbers, three positioning conventions |
| 0DTE Variance | Known-answer test of the Cboe variance-replication estimator on a synthetic 0DTE chain; implied-move and event-day-move calculators |
| Realized Vol | Range estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang, RV) vs. a known truth incl. overnight gaps and microstructure noise; HAR forecasting lab with QLIKE and Diebold–Mariano |
| Backtest Lab | CSCV / probability of backtest overfitting and the deflated Sharpe ratio; bar-touch vs. tick-level fills on a random walk |
| Research · About | Working-paper results (forecast ladder, Sharpe intervals), Research 2 design, research standards, toolkit, teaching |

Images in `assets/`: profile picture (`avatar.jpg/.webp`, re-encoded without metadata) and the social preview card `og.jpg` (1200×630, used by LinkedIn/WhatsApp/X link previews).

Deep links: `#live` (alias `#market`), `#range` (aliases `#levels`, `#ranges`, `#em`; sections `#rg-math`, `#rg-lab`, `#rg-dist`), `#surface` (alias `#vol`), `#greeks`, `#dealer` (alias `#lab`), `#odte`, `#realized`, `#backtest` (aliases `#overfit`, `#backtests`), `#research`, `#f1`, `#f2`, `#standards`, `#about` (alias `#teaching`).

Only external resource at runtime: KaTeX 0.18.9 from jsDelivr (pinned, SRI hashes) for formula rendering. Live data is served from the same origin.

## Configuration

Links that are not public yet stay hidden. Fill them in the `SITE` object at the top of the main `<script>` in `index.html`:

```js
const SITE = {
  email: '', linkedin: '', orcid: '', cv: 'cv.pdf',
  f1: { paper: '', code: '', doi: '' },
  liveData: '',   // 'data/latest.json' once the forecast workflow runs
};
```

## Live market data

`pipeline/live_snapshot.py` downloads the SPX/SPXW chain (~29,000 contracts), the NDX/NDXP chain (~15,000; written to `data/live/ndx.js`, loaded only when the Nasdaq-100 is selected) and the VIX family from Cboe’s public delayed-quote endpoints and writes `data/live/latest.json` and `latest.js` (~95 KB, derived analytics only — forwards, SVI fits and smile splines, model-free variances, gamma aggregates, bucket shares, index levels). Cboe does not send CORS headers, so browsers cannot fetch it directly; the GitHub Action does it server-side.

- History & live chart: five years of Cboe daily closes (S&P 500, VIX1D, VIX9D, VIX, VIX3M, VVIX), Cboe’s 1-minute bars of the latest session sampled every 5 minutes (`data/live/bars.json`, last five sessions), a rolling intraday record of the 30-minute snapshots (last five sessions, `data/live/intraday.json` — each run restores it from the deployed site, so it grows without a commit per snapshot) and an end-of-day archive of my own measures (implied move, 30-day model-free vol, zero-gamma level) compiled from `data/live/history/`. My measures are drawn over their official Cboe reference series (VIX1D/√252, VIX, S&P 500), which carry the full history.
- Level study (`data/live/study.js`, ~120 KB, loaded only by the Implied Range tab): one row per S&P 500 session of the last five years with open / high / low / close relative to the prior close (basis points) and the prior close’s VIX, VIX1D, VXN and VIX3M. S&P 500 OHLC from Cboe, Nasdaq-100 OHLC from Nasdaq; both were cross-checked against a second source (high, low and close agree; 20 Nasdaq-100 and 1 S&P 500 vendor opens equal the prior close or lie outside the day’s range and are treated as missing). The browser computes every statistic from this file; the pipeline’s own calibration and an independent pandas recomputation agree exactly.
- Intraday bars keep the true 5-minute high and low of the S&P 500 and Nasdaq-100 (from Cboe’s 1-minute bars), so level touches in the session chart are not missed between samples.
- The page re-checks for a new snapshot every 2 minutes while the market is open and swaps it in without a reload.
- Runs whose option quotes come from Cboe's overnight session (20:15–09:30 ET on trading nights) publish nothing: those quotes price the overnight market while the index print is still the prior close, so every implied vol would use the wrong time to expiry. The last snapshot stays.
- Data gate before every publish (real data, not the synthetic test chain): the nearest forward must sit within ±0.25 % of spot (catches quotes and index print from different sessions; ±2 % between 16:00 and 16:15 ET when both come from the same session, because the index print is frozen while the options trade on), the spline must keep a median of at least 80 % of strikes inside the bid–ask, and the replicated 30-day vol must lie within max(2.5 points, 12 % of the index level) of Cboe's VIX (VXN for the Nasdaq-100). A failed gate publishes nothing; the verdict is logged in the workflow run and stored in `checks.gate`.
- `pipeline/test_live_snapshot.py` — known-answer tests on a synthetic Black–Scholes chain (forward, rate, ATM IV, replicated variance, SVI fit, spline inside the bid–ask and free of butterfly arbitrage, bucket totals), the NDX configuration, the term structure (flat → flat, backwardation without arbitrage, an inversion that makes total variance fall → flagged with its size in vol points) and on synthetic Brownian sessions for the level study (inside → 2Φ(k) − 1, touch → 2(1 − Φ(k)), touched → held 50 %, realized ÷ implied = 1, bad opens excluded); runs before every snapshot and blocks publishing on failure.
- `.github/workflows/live-data.yml` — every 15 minutes during US trading hours and a few times after the close: test → snapshot → deploy to Pages; after 16:15 ET it commits a small end-of-day summary to `data/live/history/` (never from a pre-open run, which would carry pre-market VIX quotes).
- Local refresh: `pip install -r pipeline/requirements.txt && python pipeline/live_snapshot.py`.
- Licensing: only derived analytics are published. `PUBLISH_POINTS` in the pipeline switches off the per-strike bid/ask IV points used for the smile chart, should Cboe’s terms require it.

## Live forecast tile

When `SITE.liveData` is set, the hero tile reads `data/latest.json` (schema in `data/latest.example.json`) and shows the pre-open forecast with its commit time and hash. Until then it states honestly that the live forecast is planned — nothing is backfilled.

## Deploy (GitHub Pages)

1. Create a public repository named exactly `jonaslffr-ship-it.github.io`.
2. Push this folder (index.html, `.nojekyll`, `pipeline/`, `data/`, `.github/`) to `main`.
3. Settings → Pages → Source: **GitHub Actions** (the `live-data` workflow builds and deploys the site, including fresh data).

## License

Text © 2026 Jonas Löffler, CC BY 4.0 · code MIT.
Research and education — not investment advice. No licensed raw data is published here.
