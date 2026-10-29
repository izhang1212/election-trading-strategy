# Simulation re-run with real-data assumptions (v2)

Synthetic data only. P&L per night on $100. **±** = 95% interval; in v2 standard errors are **clustered by history draw** (nights scored with the same fitted model are not independent). The original study reported iid errors; its clustered errors are shown in brackets. v2 changes: real county structure from `data/fitted/`, Election-Day-vs-early gap sd 5.5 points (was 2), year noise 2 points (was 1), 2026-10-29 prices, 20 history draws, exact 'more than 1c' move filter. Market, strategy, estimator and metric definitions are unchanged (`market.py`, `sim_strategy.py`, `run.py`).

## Texas Senate

Reproduction check: `run_v2.py --scenario v1` reproduces `results_TX.json` **exactly** (every P&L figure identical).

| Metric | Original study | v2 (recalibrated) |
|---|---|---|
| Prior / simulated D win rate | 0.62 / 60.7% | 0.685 / 69.2% |
| Median minutes until decided (p25–p75) | 5 (4–8) | 5 (3–14) |
| Nights decided within 15 / 30 min | 88% / 91% | 75% / 77% |
| Brier, our model at +15 / +30 min | 0.054 / 0.045 | 0.083 / 0.081 |
| Mean P&L, naive 0% | $-0.46 ± 0.34 [± 0.32] | $-0.04 ± 0.78 |
| Mean P&L, naive 10% | $-0.38 ± 0.32 [± 0.27] | $-0.31 ± 0.55 |
| Mean P&L, naive 20% | $+0.59 ± 0.33 [± 0.26] | $+0.50 ± 0.50 |
| Mean P&L, naive 30% | $+1.88 ± 0.35 [± 0.37] | $+1.73 ± 0.50 |
| Mean P&L, naive 40% | $+3.62 ± 0.39 [± 0.55] | $+3.24 ± 0.60 |
| Break-even naive share | 14% | 14% |
| Naive-share estimate, true 0%: median (p10–p90); mean ± sd | 4% (0–20); 8 ± 9 | 4% (0–36); 12 ± 15 |
| Naive-share estimate, true 10%: median (p10–p90); mean ± sd | 10% (8–24); 14 ± 8 | 12% (6–38); 17 ± 14 |
| Naive-share estimate, true 20%: median (p10–p90); mean ± sd | 20% (18–30); 22 ± 7 | 20% (14–44); 25 ± 13 |
| Naive-share estimate, true 30%: median (p10–p90); mean ± sd | 30% (28–38); 32 ± 5 | 30% (24–48); 33 ± 11 |
| Naive-share estimate, true 40%: median (p10–p90); mean ± sd | 40% (38–46); 41 ± 5 | 40% (34–56); 43 ± 8 |
| H1 rule A, w-hat above the simulated null's 95th pct (28% / 52%): % of nights rejecting at true 0/10/20/30/40% | 4 / 6 / 10 / 79 / 99 | 4 / 5 / 7 / 8 / 13 |
| H1 rule B, bootstrap lower 90% bound > 10% (40 nights): % of nights rejecting at true 0/10/20/30/40% | 20 / 62 / 88 / 95 / 98 | 8 / 48 / 90 / 98 / 98 |

**TX v2: effect of filter rule and latency** (P&L change vs. exact '>1c' filter at 1-min latency, same nights; ± 95%, clustered):

| Variant | naive 0% | naive 20% | naive 40% |
|---|---|---|---|
| original float filter | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| '1c or more' filter | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| latency 2 min, filter 1 min | -0.06 ± 0.08 | -0.08 ± 0.10 | +0.01 ± 0.10 |
| latency 2 min, filter 2 min | +0.29 ± 0.16 | -0.01 ± 0.12 | -0.54 ± 0.16 |
| latency 3 min, filter 1 min | -0.32 ± 0.13 | -0.24 ± 0.14 | -0.11 ± 0.15 |
| latency 3 min, filter 3 min | +0.31 ± 0.32 | +0.01 ± 0.20 | -1.07 ± 0.18 |
| latency 5 min, filter 1 min | -2.66 ± 0.32 | -2.19 ± 0.26 | -2.03 ± 0.20 |
| latency 5 min, filter 5 min | +0.27 ± 0.49 | -0.07 ± 0.26 | -1.60 ± 0.21 |

## Georgia Governor

Reproduction check: `run_v2.py --scenario v1` reproduces `results_GA.json` **exactly** (every P&L figure identical).

| Metric | Original study | v2 (recalibrated) |
|---|---|---|
| Prior / simulated D win rate | 0.48 / 49.6% | 0.615 / 60.2% |
| Median minutes until decided (p25–p75) | 9 (7–18) | 13 (8–39) |
| Nights decided within 15 / 30 min | 72% / 84% | 52% / 62% |
| Brier, our model at +15 / +30 min | 0.102 / 0.069 | 0.133 / 0.123 |
| Mean P&L, naive 0% | $-1.18 ± 0.40 [± 0.59] | $-0.74 ± 0.56 |
| Mean P&L, naive 10% | $-1.14 ± 0.38 [± 0.53] | $-0.74 ± 0.53 |
| Mean P&L, naive 20% | $-0.03 ± 0.39 [± 0.51] | $+0.50 ± 0.54 |
| Mean P&L, naive 30% | $+1.62 ± 0.43 [± 0.54] | $+2.33 ± 0.51 |
| Mean P&L, naive 40% | $+3.67 ± 0.47 [± 0.61] | $+4.76 ± 0.63 |
| Break-even naive share | 20% | 16% |
| Naive-share estimate, true 0%: median (p10–p90); mean ± sd | 2% (0–20); 7 ± 11 | 6% (0–26); 10 ± 11 |
| Naive-share estimate, true 10%: median (p10–p90); mean ± sd | 10% (4–24); 12 ± 10 | 12% (8–30); 16 ± 10 |
| Naive-share estimate, true 20%: median (p10–p90); mean ± sd | 20% (14–30); 21 ± 9 | 22% (16–36); 25 ± 9 |
| Naive-share estimate, true 30%: median (p10–p90); mean ± sd | 30% (24–38); 31 ± 7 | 32% (26–44); 34 ± 8 |
| Naive-share estimate, true 40%: median (p10–p90); mean ± sd | 40% (36–48); 41 ± 6 | 42% (36–52); 43 ± 7 |
| H1 rule A, w-hat above the simulated null's 95th pct (34% / 34%): % of nights rejecting at true 0/10/20/30/40% | 5 / 4 / 6 / 14 / 90 | 3 / 7 / 12 / 32 / 94 |
| H1 rule B, bootstrap lower 90% bound > 10% (40 nights): % of nights rejecting at true 0/10/20/30/40% | 8 / 30 / 88 / 95 / 98 | 15 / 45 / 98 / 100 / 100 |

**GA v2: effect of filter rule and latency** (P&L change vs. exact '>1c' filter at 1-min latency, same nights; ± 95%, clustered):

| Variant | naive 0% | naive 20% | naive 40% |
|---|---|---|---|
| original float filter | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| '1c or more' filter | +0.00 ± 0.00 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| latency 2 min, filter 1 min | -0.11 ± 0.07 | -0.05 ± 0.08 | +0.03 ± 0.10 |
| latency 2 min, filter 2 min | +0.10 ± 0.10 | +0.09 ± 0.17 | +0.10 ± 0.21 |
| latency 3 min, filter 1 min | -0.51 ± 0.12 | -0.37 ± 0.12 | -0.13 ± 0.20 |
| latency 3 min, filter 3 min | +0.22 ± 0.20 | +0.05 ± 0.23 | -0.48 ± 0.27 |
| latency 5 min, filter 1 min | -2.85 ± 0.34 | -1.85 ± 0.36 | -1.06 ± 0.48 |
| latency 5 min, filter 5 min | +0.36 ± 0.41 | -0.11 ± 0.40 | -2.15 ± 0.45 |

Georgia v2 with Texas-level liquidity: naive 0%: $-0.50 ± 0.58; naive 20%: $+0.97 ± 0.58; naive 40%: $+5.55 ± 0.67

