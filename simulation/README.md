# Simulations (synthetic data only)

Two pre-election simulation studies. Nothing here uses real election results: county sizes and leans are rough, real-shaped approximations, and spreads, reporting times and early-vs-Election-Day gaps are assumptions. The purpose is to test the pipeline and the strategy before the election, not to forecast the races.

## Study 1 (Phase 1): single-state strategy test

Files: `phase1_sim.py`, `phase1_strategy.py`, `phase1_run.py`, `phase1_plot_night.py`, `phase1_estimate_w.py`

- `phase1_sim.py`: a Texas-shaped world with 20 reporting units, latent swings (statewide, four county types, early-vs-Election-Day gap, county effects) and batch-by-batch reporting. Includes the Kalman nowcast and the minute-by-minute fair value (our model runs 1 minute late).
- `phase1_strategy.py`: the market (informed traders plus naive traders who extrapolate the raw count with a 5-minute lag, and follow the media call once the race is called) and two strategy families: taking liquidity (with an optional move filter) and resting maker orders.
- `phase1_run.py`: 2,000 nights, four versions of our model (good, poor, identical to the informed traders', zero-latency), naive share 0–40%, five strategies.
- `phase1_estimate_w.py`: tests whether one night's price path recovers the naive share.

Key findings: taking liquidity with a 1¢ move filter is the only variant near break-even when there are no naive traders. Resting maker orders lose money in every scenario. Latency, not fees, is the main cost.

Run: `python phase1_run.py 2000`, then `python phase1_plot_night.py` and `python phase1_estimate_w.py`.

## Study 2: Texas Senate vs. Georgia Governor

Files: `world.py`, `model.py`, `market.py`, `sim_strategy.py`, `estimate_naive.py`, `run.py`, `report.py`

The full pipeline, per state:

1. **Data pull** (`world.py`): synthetic 2018–2024 county history split into early and Election Day votes, plus 2026 early-vote turnout per county known before election night. Texas has 60 reporting units, Georgia 55: the largest counties individually, small counties pooled.
2. **Fit** (`model.py`): our county leans, Election Day gaps and early-vote shares, estimated from that history. Repeated over 5 independent history draws.
3. **Nowcast** (`model.py`): a Kalman filter over [statewide swing, four county-type swings, Election Day gap, county effects]. Each batch's surprise updates the state, the uncounted vote is projected, and the win probability comes from a Gaussian on the final margin. The statewide level is calibrated so the pre-results probability equals the market price.
4. **Market** (`market.py`): informed + naive traders; Georgia gets wider spreads and less depth (its Kalshi market is about 7x smaller).
5. **Strategy** (`sim_strategy.py`): taker with fee + 3¢ buffer, 1¢ move filter, quarter-Kelly, 50% cap, exit at fair value or 97¢. Run at $100 and $10,000.
6. **Metrics** (`run.py`, `report.py`): P&L, risk, how long the race stays uncertain, model calibration, naive-share estimator precision.

Run: `python run.py TX 2000 5`, `python run.py GA 2000 5`, `python run.py GA 2000 5 TX` (Georgia with Texas-level liquidity), then `python report.py`.

Key findings (2,000 nights per state, 95% intervals):

| | Texas Senate | Georgia Governor |
|---|---|---|
| P&L per night on $100, no naive traders | −$0.46 ± 0.34 | −$1.18 ± 0.40 |
| P&L on $100, 20% naive | +$0.59 ± 0.33 | −$0.03 ± 0.39 |
| P&L on $100, 40% naive | +$3.62 ± 0.39 | +$3.67 ± 0.47 |
| Median minutes until the race is effectively decided | 5 | 9 |
| Naive-share estimate when the truth is 20% | 22% ± 7 | 21% ± 9 |

Texas was chosen as the primary market: lower break-even naive share (about 13–15% vs. about 20%), deeper liquidity, slightly more precise measurement. Georgia runs the same pipeline in record-only or paper mode.

Outputs: `results_TX.json`, `results_GA.json`, `results_GA_liqTX.json`, `summary_table.md`, `figures/`.

## Re-run with real-data assumptions (v2, 2026-10-29)

Files: `run_v2.py`, `report_v2.py`, `results_*_v1.json` / `results_*_v2.json`, `summary_table_v2.md`, `figures/v2_*.png`

`run_v2.py` mirrors `run.py` step for step. `--scenario v1` reproduces `results_TX.json` and `results_GA.json` exactly. `--scenario v2` changes only the assumptions that the real 2018–2024 county data contradicted:

- real county structure from `../data/fitted/`;
- Election-Day-vs-early gap sd 5.5 points (was 2);
- 2026-10-29 prices (TX 0.685, GA 0.615);
- 20 history draws with clustered standard errors;
- the exact ">1c" move filter.

It also adds latency and filter sensitivity (paired on the same nights) and evaluates two H1 decision rules.

`world.py` and `sim_strategy.py` carry small opt-in additions (`World.from_fitted`, `SIG["ed_year"]`, `filter_rule` / `filter_window`, clustered `summarize`). Their defaults reproduce the original exactly. `estimate_naive.py` is now a thin wrapper around the repository's single estimator (`../estimate_naive.py`), which was checked to be bit-identical to the original.

Run: `python run_v2.py TX 2000 20 --scenario v2`, `python run_v2.py GA 2000 20 --scenario v2`, `python run_v2.py GA 2000 20 TX --scenario v2`, then `python report_v2.py`.
