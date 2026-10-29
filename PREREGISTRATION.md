# Pre-registration

**Status: DRAFT, not frozen.** It is frozen by publishing GitHub release `v1.0-prereg` (and the Zenodo DOI) before polls open on November 3, 2026, 7:00 a.m. ET. After that, this file does not change; every departure is listed under [Deviations](#deviations) in the paper.

## 1. Questions and hypotheses

Primary result: the **naive share** *w*, the fraction of price formation that follows the raw running tally rather than an expectation-adjusted fair value. It is reported with its interval whatever the P&L.

| | Hypothesis | Test | Decision rule |
|---|---|---|---|
| H1 | Naive trading exists | Estimator of §4 on the primary market | Lower end of the 90% interval > 0.10 |
| H2 | The price gap moves with the naive signal | Regress (mid − fair) on (naive − fair), minute data, Newey–West SE (lag 10) | Slope > 0 at 5% |
| H3 | Prices overshoot and reverse after misleading batches | Event study around batches whose raw-tally move and fair-value move have opposite signs: mid change at +1, +2, +5, +10, +30 min | Mean reversal from +2 to +30 min of the opposite sign to the +0 to +2 move, 5% |
| H4 | Strategy trades have positive markouts net of fees | Markout = (fair value at t + k − fill price) × side − fee, k = 1, 5, 15 min | Mean markout > 0 at k = 5, one-sided 5% |

The 0.10 threshold in H1 exists because the estimator is biased upward near zero. In simulation it averaged 7–8% when the true share was 0, since our model's own errors look like naive trading. One night's P&L (sd ≈ $10 on $100) is reported but is **not** a test of anything.

## 2. Markets

| Market | Ticker (YES = Democrat) | Role |
|---|---|---|
| Texas U.S. Senate (Talarico v. Paxton) | `SENATETX-26-D` | Primary for measurement. **Record-only on the night** (no live trades) |
| Georgia Governor (Jackson v. Bottoms) | `GOVPARTYGA-26-D` | Secondary: live paper trading through the full pipeline |

**Why Texas is record-only.** The 2026 Texas SOS results page reports statewide totals only, with no county breakdown and no early/ED split. County detail comes from the counties' own feeds (Harris JSON; Clarity ENR for Tarrant, Bexar, Denton, El Paso, Dallas, Travis, Fort Bend, Williamson, Hidalgo; Collin's posted reports), which together cast 61.6% of the 2026 Texas early vote. These are captured raw with receive timestamps on the night, and their parsers are written after the election. Parsers only extract vote counts by county, vote type and reporting status, and are tested against the captured files. Fair value, the naive signal and the paper strategy for Texas are then computed from the captured files in receive-time order, so nothing is used before it was received. Counties without a captured feed enter through the statewide total, as the difference between the statewide count and the sum of the captured counties.

Recorded alongside, for controls: `SENATETX-26-R`, `GOVPARTYGA-26-R`, `CONTROLS-2026-D`, `CONTROLH-2026-D`.

Selection criteria: a simple winner rule; ballots essentially all in on election night; staggered county reporting through a public feed; a predictable reporting-order swing; a race still uncertain after the first big drop; enough liquidity. Excluded: Nevada (mail accepted for 4 days), Maine and Alaska (ranked-choice voting).
**Known exception:** Georgia requires a majority, with a runoff if nobody passes 50%. A runoff is unlikely in a two-candidate race. The Georgia market resolves on the eventual winner, and a runoff is reported as such.

## 3. Fair value (nowcast)

- Inputs: `data/fitted/{tx,ga}_counties.csv` and `{tx,ga}_types.csv`, produced by `fit_counties.py` from the files in `data/SOURCES.md`, with SHA-256 in §8.
- Model: `nowcast.py`, a Kalman filter over [statewide swing, four type swings, ED-vs-early gap, county effects] with prior SDs `nowcast.sig` in `config.yaml`, and a Gaussian on the final two-party margin.
- **History weights:** each past cycle's weight in the county fit is 2 for a midterm and 1 for a presidential year, and the oldest cycle is halved (Texas 2020/2022/2024 = 1/2/1; Georgia 2018/2020/2022/2024 = 1/1/2/1). 2026 is a midterm. Texas 2018 is excluded for data quality (`data/SOURCES.md`).
- **ED-vs-early gap uncertainty:** prior sd 0.055 of two-party share. This is the cross-cycle sd of the statewide Election Day minus early gap in the official results (Texas 2020–24: −8.8, −3.9, +2.1 points, sd 5.4; Georgia 2018–24: −3.4, −14.4, −13.8, −3.3, sd 6.2). The simulation's original 0.02 understated it.
- **Anchor:** the statewide level is set so that, before results, the model's win probability equals the **last Kalshi mid before that state's first results batch**. Out-of-state news before the first batch is therefore absorbed by the anchor and not counted as naive trading.

## 4. Naive-share estimator (`estimate_naive.py`, the only implementation)

- Naive signal: naive_t = Φ((raw_{t−ℓ} − 0.5) / (0.06·√(1 − frac_{t−ℓ}) + 0.003)). Before any count it equals the anchor price; after the first projection of the race by AP or a major network it equals the called side (1 or 0).
- Price model: target_t = (1 − w)·fair_t + w·naive_t; m_t = m_{t−1} + κ·(target_t − m_{t−1}); m_0 = anchor.
- Fit: grid search, w ∈ {0, 0.02, …, 0.60}, ℓ ∈ {1, …, 10} minutes, κ ∈ {0.3, 0.5, 0.8}, minimizing Σ(m_t − mid_t)², with minute data.
- `fair` is our model's fair value using information through minute t, recomputed after the night from the captured files with no look-ahead. `raw` and `frac` come from the same files.
- **Window:** from the minute of the state's first results batch to the earlier of the first media call and +180 minutes.
- **Interval:** 90% residual block bootstrap (blocks of 10 minutes, 500 draws, refit each). It is reported next to the simulation-based sd of the estimator at the estimated w (simulation, `estimate_naive.py` via `simulation/`). **[OPEN: which one decides H1; see §9.]**

## 5. Strategy (`strategy.py`, port of `simulation/sim_strategy.py:run_taker`)

- **Entry:** buy YES when fair − ask − 0.07·ask·(1 − ask) − 0.03 > 0. Buy NO when (1 − fair) − (1 − bid) − fee − 0.03 > 0.
- **Move filter:** no entries and no model-driven exits in a decision tick where the mid moved by more than 1¢ since the previous tick. **[OPEN: "more than" vs. "1¢ or more"; see §9.]**
- **Sizing:** quarter-Kelly, f = (q − px)/(1 − px) of equity; at most 50% of equity in the market; never flip sides in one tick.
- **Exit:** sell when the bid net of the per-contract fee exceeds fair value (YES; NO symmetrically), or when the held contract's bid reaches 97¢.
- **Orders:** taking liquidity only (immediate-or-cancel). Decision tick every 60 s.
- **Kill switch:** trips on a count that decreases (parse anomaly), 3 consecutive parse failures, no new results file for 10 minutes while votes are outstanding, or |fair − mid| > 25¢. It stays tripped, and the open position is then handled by hand per RUNBOOK.md.
- **Live vs. paper:** no live trading in either market. Georgia is paper-traded live by `run_live.py`. Texas is recorded only, and its paper strategy is computed after the election from the captured files in receive-time order (§2). If the Georgia feed is missing on the night (unconfirmed slug), Georgia is also evaluated after the fact the same way.

## 6. What is logged

Raw results bytes with UTC receive time and SHA-256 (`scrape_results.py`). Every Kalshi WebSocket message with exchange `ts_ms`, `seq` and local receive time, plus REST snapshots every 30 s (`record_markets.py`). One decision record per state per tick (`run_live.py`). The full logs are published as the dataset after the election.

## 7. Analyses beyond the hypotheses (exploratory, labeled as such)

Latency (county sites vs. state portal vs. price moves); speed of absorption per batch; Georgia vs. Texas comparison; sensitivity of *w* to the window end.

## 8. Frozen inputs

Filled in at the freeze with `shasum -a 256 config.yaml data/fitted/*.csv`. This memo can't contain its own hash, so the release zip's SHA-256 goes in the README and the Zenodo record.

| File | SHA-256 |
|---|---|
| `config.yaml` | (at freeze) |
| `data/fitted/tx_counties.csv` | (at freeze) |
| `data/fitted/tx_types.csv` | (at freeze) |
| `data/fitted/ga_counties.csv` | (at freeze) |
| `data/fitted/ga_types.csv` | (at freeze) |

Timestamp procedure: connect the repo to Zenodo **before** publishing the release, because Zenodo does not archive releases retroactively, and its GitHub integration requires a public repo. If the repo stays private until after the election, upload the release zip to Zenodo by hand instead.

## 9. Open before freeze

1. ~~Texas county feed~~ Decided 2026-10-29: Texas record-only, county feeds captured (§2).
2. ~~Simulation assumptions vs. real data~~ Decided 2026-10-29: real county structure, gap sd 0.055, history weights (§3). Re-run results are in `simulation/summary_table_v2.md`.
3. ~~Move-filter wording~~ Decided 2026-10-29: "more than 1¢" as written (`gt`). The simulation cannot tell the two apart, because its mid is continuous and exact 1¢ ties never occur.
3a. **Filter window vs. latency (proposed).** In v2, a 5-minute data latency with a 1-minute filter costs about $2 per night, which wipes out the edge. Blocking for as many minutes as the latency removes most of that loss at low naive shares. Proposal: window = measured median latency from the Nov 1–2 dry run, rounded up, with a floor of 1 and a cap of 5.
4. **Which rule decides H1 (§4).** v2 results (`simulation/summary_table_v2.md`): rule A (estimate above the simulated null's 95th percentile) has the right size but almost no power in Texas (13% at a true 40%). Rule B (bootstrap lower bound > 10%) has good power but rejects too often at a true 0% (8% TX, 15% GA, against a nominal 5%).
5. Final 2026 early-vote pull (TX Oct 31) and the Georgia early-vote file (manual).
6. Georgia results slug for 2026.

## Deviations

None yet. After the freeze, any change to code, inputs or rules is listed here with its date and reason.
