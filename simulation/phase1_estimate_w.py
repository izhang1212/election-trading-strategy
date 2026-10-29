"""Can we back out the naive-trader share (w) from ONE night's logged data?

Logged on the night: market mid each minute, our model's fair value (info through t), the raw running
count / fraction reported (hence the naive raw-count signal), and the media call time.
Estimator: grid-search (w, naive lag, market speed kappa) to minimise squared error between the observed
mid and the mid implied by   target = (1-w)*our_fair + w*naive_signal(lag),  mid_t = mid_{t-1} + kappa*(target - mid_{t-1}).
Note we must use OUR fair value as a proxy for the informed traders' - a real-world source of bias.
"""
import numpy as np
from scipy.stats import norm
from phase1_strategy import build_market

d = np.load("nowcast_ours_good_lat0_2000.npz")
NN, TMAX = 300, 150
P_S, P_OUR, RAW, FRAC, FLAG, PRIOR = (d[k][:NN] for k in ("P_S", "P_O", "RAW", "FRAC", "FLAG", "PRIOR"))

def called_mask(P):
    sure = (P > 0.995) | (P < 0.005); acc = np.zeros(P.shape[0]); run = np.zeros_like(P)
    for t in range(P.shape[1]):
        acc = np.where(sure[:, t], acc + 1, 0); run[:, t] = acc
    return np.maximum.accumulate(run >= 15, axis=1)

CALLED = called_mask(P_S)            # the media call is observable on the night

def naive_signal(lag):
    raw_l = np.full_like(RAW, np.nan); raw_l[:, lag:] = RAW[:, :-lag]
    frac_l = np.zeros_like(FRAC); frac_l[:, lag:] = FRAC[:, :-lag]
    sig = 0.06 * np.sqrt(np.clip(1 - frac_l, 0, 1)) + 0.003
    pn = np.where(np.isnan(raw_l), PRIOR[:, None], norm.cdf((np.nan_to_num(raw_l, nan=0.5) - 0.5) / sig))
    return np.where(CALLED, (P_S > 0.5).astype(float), pn)

W_GRID = np.round(np.arange(0, 0.61, 0.02), 2); LAGS = range(1, 11); KAPPAS = [0.3, 0.5, 0.8]

def estimate(mid):
    """Returns per-night w-hat (n,) and the chosen lag."""
    n = mid.shape[0]; best = np.full(n, np.inf); w_hat = np.zeros(n); lag_hat = np.zeros(n)
    for lag in LAGS:
        pn = naive_signal(lag)[:, :TMAX]
        for kap in KAPPAS:
            for w in W_GRID:
                tgt = (1 - w) * P_OUR[:, :TMAX] + w * pn
                m = np.empty_like(tgt); m[:, 0] = PRIOR
                for t in range(1, TMAX):
                    m[:, t] = m[:, t - 1] + kap * (tgt[:, t] - m[:, t - 1])
                sse = ((m - mid[:, :TMAX]) ** 2).sum(1)
                better = sse < best
                best = np.where(better, sse, best); w_hat = np.where(better, w, w_hat); lag_hat = np.where(better, lag, lag_hat)
    return w_hat, lag_hat

print("true w | mean w-hat | single-night sd | 90% of nights within | median lag-hat")
for w_true in (0.0, 0.1, 0.2, 0.3, 0.4):
    mid, *_ = build_market(P_S, RAW, FRAC, FLAG, PRIOR, w_true, np.random.default_rng(99))
    w_hat, lag_hat = estimate(mid)
    err = w_hat - w_true
    print(f"{w_true:.2f}   | {w_hat.mean():.3f}      | {w_hat.std():.3f}           | +/-{np.percentile(np.abs(err), 90):.3f}"
          f"               | {np.median(lag_hat):.0f}", flush=True)
