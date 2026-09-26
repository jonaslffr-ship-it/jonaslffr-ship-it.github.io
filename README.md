# jonaslffr-ship-it.github.io

Personal research site of Jonas Löffler — volatility forecasting, options and honest backtesting.

**Live:** https://jonaslffr-ship-it.github.io/

## What is here

A single, dependency-free `index.html` (no build step): a dark-mode app with tabs (Apple HIG + Material 3 design language). Every tool runs in the browser on simulated or parametric data; the 3D surfaces use a small custom canvas renderer (no WebGL library).

| Tab | What it does |
|---|---|
| Overview | Positioning, live market strip (SPX, VIX family, implied move), research highlights, tool launcher, research log |
| Live Market | Real SPX data (Cboe delayed quotes, ~15 min): SVI surface in 3D, smiles with bid–ask fit quality, model-free implied moves checked against VIX9D/VIX/VIX3M, VIX term structure and 12-month history, dealer gamma under two conventions with zero-gamma level, moneyness shares, per-expiry forward/rates |
| Vol Surface | SSVI surface (Gatheral & Jacquier 2014) in 3D — implied vol, total variance, Dupire local vol; butterfly / calendar / Gatheral–Jacquier checks; smiles, ATM term structure, risk-neutral density |
| Options & Greeks | Strategy builder (10 presets, editable legs); P&L and 13 Black–Scholes greeks to third order as 3D surfaces over spot × vol or spot × time; per-leg finite-difference check |
| Dealer Gamma | Toy dealer-hedging feedback σ_eff = σ₀ / (1 + λΓ_D): gamma flip, GEX by strike, vol multiplier, Monte Carlo paths with common random numbers, three positioning conventions |
| 0DTE Variance | Known-answer test of the Cboe variance-replication estimator on a synthetic 0DTE chain; implied-move and event-day-move calculators |
| Realized Vol | Range estimators (Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang, RV) vs. a known truth incl. overnight gaps and microstructure noise; HAR forecasting lab with QLIKE and Diebold–Mariano |
| Backtest Lab | CSCV / probability of backtest overfitting and the deflated Sharpe ratio; bar-touch vs. tick-level fills on a random walk |
| Research · About | Working-paper results (forecast ladder, Sharpe intervals), Research 2 design, research standards, toolkit, teaching |

Images in `assets/`: profile picture (`avatar.jpg/.webp`, re-encoded without metadata) and the social preview card `og.jpg` (1200×630, used by LinkedIn/WhatsApp/X link previews).

Deep links: `#surface`, `#greeks`, `#dealer` (alias `#lab`), `#odte`, `#realized`, `#backtest`, `#research`, `#f1`, `#f2`, `#standards`, `#about`.

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

`pipeline/live_snapshot.py` downloads the SPX/SPXW chain (~29,000 contracts) and the VIX family from Cboe’s public delayed-quote endpoints and writes `data/live/latest.json` and `latest.js` (~95 KB, derived analytics only — forwards, SVI fits, model-free variances, gamma aggregates, bucket shares, index levels). Cboe does not send CORS headers, so browsers cannot fetch it directly; the GitHub Action does it server-side.

- History & live chart: five years of Cboe daily closes (S&P 500, VIX1D, VIX9D, VIX, VIX3M, VVIX), Cboe’s 1-minute bars of the latest session sampled every 5 minutes (`data/live/bars.json`, last five sessions), a rolling intraday record of the 30-minute snapshots (last five sessions, `data/live/intraday.json` — each run restores it from the deployed site, so it grows without a commit per snapshot) and an end-of-day archive of my own measures (implied move, 30-day model-free vol, zero-gamma level) compiled from `data/live/history/`. My measures are drawn over their official Cboe reference series (VIX1D/√252, VIX, S&P 500), which carry the full history.
- `pipeline/test_live_snapshot.py` — known-answer tests on a synthetic Black–Scholes chain (forward, rate, ATM IV, replicated variance, SVI fit, bucket totals); runs before every snapshot and blocks publishing on failure.
- `.github/workflows/live-data.yml` — every 30 minutes during US trading hours (plus once after the close): test → snapshot → deploy to Pages; after the close it commits a small end-of-day summary to `data/live/history/`.
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
