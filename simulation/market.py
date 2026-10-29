"""Stage 4: the market. Mid price = smoothed blend of the informed fair value (instant) and a naive
raw-count extrapolation (lagged); after a media call the naive traders follow the call. Spreads widen after batches.
Liquidity differs by state (Georgia's market is ~7x smaller than Texas's)."""
import numpy as np
from scipy.stats import norm


def build_market(R, w, cfg, rng, lag=5, kappa=0.5, noise=0.004):
    P_S, RAW, FRAC, FLAG, PRIOR = R["P_S"], R["RAW"], R["FRAC"], R["FLAG"], R["PRIOR"]
    n, T1 = P_S.shape
    raw_l = np.full_like(RAW, np.nan); raw_l[:, lag:] = RAW[:, :-lag]
    frac_l = np.zeros_like(FRAC); frac_l[:, lag:] = FRAC[:, :-lag]
    sig_n = 0.06 * np.sqrt(np.clip(1 - frac_l, 0, 1)) + 0.003
    p_naive = np.where(np.isnan(raw_l), PRIOR[:, None], norm.cdf((np.nan_to_num(raw_l, nan=0.5) - 0.5) / sig_n))
    sure = (P_S > 0.995) | (P_S < 0.005)
    acc = np.zeros(n); run = np.zeros_like(P_S)
    for t in range(T1):
        acc = np.where(sure[:, t], acc + 1, 0); run[:, t] = acc
    called = np.maximum.accumulate(run >= 15, axis=1)
    p_naive = np.where(called, (P_S > 0.5).astype(float), p_naive)
    target = (1 - w) * P_S + w * p_naive
    mid = np.empty((n, T1)); mid[:, 0] = PRIOR
    for t in range(1, T1):
        mid[:, t] = np.clip(mid[:, t - 1] + kappa * (target[:, t] - mid[:, t - 1]) + rng.normal(0, noise, n), 0.015, 0.985)
    recent = FLAG | np.c_[np.zeros((n, 1), bool), FLAG[:, :-1]]
    hs = cfg["spread_base"] + cfg["spread_post"] * recent
    bid = np.clip(np.floor((mid - hs) * 100 + 1e-9) / 100, 0.01, 0.98)
    ask = np.clip(np.ceil((mid + hs) * 100 - 1e-9) / 100, 0.02, 0.99)
    ask = np.maximum(ask, bid + 0.01)
    return dict(mid=mid, bid=bid, ask=ask, p_naive=p_naive, called=called)
