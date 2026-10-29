# election-trading-strategy

A pre-registered study of how election-night prediction markets price partial vote counts, and a rule-based strategy that trades the gap between the market price and a county-level nowcast.

**Status:** pre-registered before the November 3, 2026 U.S. general election. Hypotheses, strategy rules and model inputs are frozen in release `v1.0-prereg` (see [Pre-registration integrity](#pre-registration-integrity)). Results will be added after the election without changing anything in that release.

## The idea

On election night, prices on markets like Kalshi move as county results come in. The running vote count is a biased estimate of the final result: which counties and ballot types report first is not random. Some traders appear to react to that raw count rather than to what is still left to count.

This project asks:

1. **How much of election-night price formation comes from traders reacting to the raw tally?** (the "naive share")
2. **How quickly do prices converge to an expectation-adjusted fair value after each batch of results?**
3. **Is the gap tradable after fees and data latency for a small participant?**

The primary output is a measurement of the market. Profit and loss is reported but treated as a noisy, secondary test, because one night cannot separate a real edge from luck.

## Markets

| Market | Role | Why |
|---|---|---|
| Texas U.S. Senate (Kalshi) | Primary: live trading | Plurality winner, no runoff, nearly all ballots in on election night, staggered county reporting, deepest market |
| Georgia Governor (Kalshi) | Secondary: same pipeline, record-only / paper trading | Close race, fast county reporting, no Libertarian on the ballot (runoff very unlikely), thinner market |

Markets were chosen with explicit criteria: simple winner rule, ballots essentially complete on election night, county-by-county reporting through a public feed, a predictable reporting order, a race that stays uncertain after the first big drop, and enough liquidity. See `PREREGISTRATION.md`.

## Method

1. **Data:** historical county results (2018–2024) split into early and Election Day votes, plus 2026 early-vote turnout by county published before election night.
2. **County model:** each county's expected vote share and turnout, the early-vs-Election-Day gap, and how counties tend to move together (urban, suburban, border/Hispanic, rural).
3. **Nowcast:** each reported batch is compared with its expectation. The surprise updates a Kalman filter over statewide, regional and county swings, which re-projects the uncounted vote into a win probability every minute.
4. **Market recording:** order books and trades from Kalshi, plus timestamped results from state and county sites.
5. **Strategy:** take liquidity only when the edge clears fees and a safety buffer, skipping any minute in which the price just jumped.
6. **Analysis:** estimate the naive share from the price path, run event studies around each batch, compute trade markouts and P&L.

## Strategy rules (frozen)

- Enter only when fair value minus the ask (or bid minus fair value) exceeds the Kalshi taker fee plus **3¢**.
- **Move filter:** skip entries and model-driven exits in any minute where the mid moved more than **1¢**, since the move may reflect data not yet processed.
- **Sizing:** quarter-Kelly, at most 50% of the bankroll in the market.
- **Exit:** sell when the bid, net of fees, exceeds fair value, or when the price reaches **97¢** (settlement is weeks away, so capital lockup matters).
- **Kill switch:** stop trading on a stale data feed, a parse anomaly, or a model–market gap too large to be believable.

## Pre-election simulation

Before the election, the full pipeline was run on 2,000 synthetic election nights per state (data pull, model fit, nowcast, market with informed and naive traders, strategy, metrics). Key takeaways, all from synthetic data with assumed spreads and reporting times:

- The strategy roughly breaks even when naive traders account for about 13–15% of price formation in Texas and about 20% in Georgia.
- At a 20% naive share, mean P&L per night on $100 was +$0.59 ± 0.33 in Texas and −$0.03 ± 0.39 in Georgia.
- One night's P&L has a standard deviation near $10 on $100, so it cannot prove an edge. One night's price path estimates the naive share to roughly ±7–9 percentage points.
- Data latency, not fees, is the main cost. Resting maker orders lost money in every scenario, so the strategy takes liquidity.

Code and figures are in `simulation/`.

## Repository layout

- Top level: this README, `PREREGISTRATION.md`, `RUNBOOK.md`, `config.yaml`, and the live pipeline scripts (`fit_counties.py`, `nowcast.py`, `strategy.py`, `record_markets.py`, `scrape_results.py`, `parse_results.py`, `run_live.py`, `estimate_naive.py`, `analyze.py`)
- `scrape_results.py` only saves raw bytes (UTC receive time, SHA-256, headers). `parse_results.py` turns them into snapshots and batches, and the same code serves live trading and the post-election re-parse. `run_live.py --replay` runs the whole pipeline on saved or synthetic files.
- `tests/`: parser tests against saved real pages (`tests/samples/`), plus `make_replay.py` for synthetic nights in the current format
- `simulation/`: Phase 1 simulation and the Texas vs. Georgia comparison, with figures and results
- `data/`: `data/fitted/` holds the frozen county expectations (committed); `data/raw/` and `data/live/` hold downloads and election-night logs (not committed until after the election)
- `paper/`: paper draft and phase memos

## Running the simulations

```bash
pip install -r requirements.txt
cd simulation
python run.py TX 2000 5      # Texas Senate: 2,000 nights over 5 history draws
python run.py GA 2000 5      # Georgia Governor
python report.py             # figures + summary table
```

## Pre-registration integrity

Git commit dates can be edited, so the timestamp of record is external:

- GitHub release: `v1.0-prereg` (link added at release)
- Archived copy: Zenodo DOI (added at release)
- SHA-256 of the release archive: (added at release)

Anything changed after the freeze is listed in a deviations section of the paper.

## Project phases

1. **Markup:** thesis, first simulation, planned conclusions and deliverables (done)
2. **Pre-registration and data audit:** market selection, Texas vs. Georgia comparison, frozen hypotheses and rules (in progress)
3. **Model fit on real county data**
4. **Live infrastructure:** recorders, scrapers, execution
5. **Calibration and rehearsal**
6. **Election night:** November 3, 2026
7. **Analysis**
8. **Write-up**

## Disclosure

The author trades his own money in these markets. This repository is research, not investment advice. Prediction-market trading involves risk of loss.

## Author

Isaac A. Zhang, NYU (Mathematics and Computer Science)