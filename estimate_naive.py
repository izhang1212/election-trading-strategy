"""Naive-share estimator: the ONLY implementation. simulation/ imports this file, so the simulated validation
in the paper is literally the code run on the real night.

Model of the market mid:
  target_t = (1 - w) * fair_t + w * naive_t(lag)
  m_t      = m_{t-1} + kappa * (target_t - m_{t-1}),   m_0 = prior
naive_t(lag) = Phi((raw_{t-lag} - 0.5) / (a*sqrt(1 - frac_{t-lag}) + b)), the win probability implied by the
raw running count; before any count it is the prior; after a media call it is the called side (0/1).
(w, lag, kappa) are chosen by grid search to minimise sum_t (m_t - mid_t)^2 over the first tmax minutes.

All inputs are plain arrays, one row per night (a single night may be passed as 1-D):
  fair  : our fair value with information through t (no trading latency)
  raw   : raw running two-party D share of the counted vote (NaN before the first count)
  frac  : fraction of the expected total vote counted
  call  : NaN before a media call, then 1.0 (D called) or 0.0 (R called)
  mid   : market mid each minute
  prior : market price at t=0 (scalar or one per night)

Usage on the real night: python estimate_naive.py data/live/analysis/minute_panel.csv
"""
import os, sys
import numpy as np
from scipy.stats import norm

_here = os.path.dirname(os.path.abspath(__file__))
try:
    import yaml
    _E = yaml.safe_load(open(os.path.join(_here, "config.yaml")))["estimator"]
except Exception:   # simulation/ must run without the live config; these equal the frozen config values
    _E = dict(w_grid=[0.0, 0.60, 0.02], lags=[1, 10], kappas=[0.3, 0.5, 0.8], tmax=180,
              naive_sigma=[0.06, 0.003], bootstrap=dict(n=500, block_minutes=10, level=0.90), h1_lower_bound=0.10)
W_GRID = np.round(np.arange(_E["w_grid"][0], _E["w_grid"][1] + _E["w_grid"][2] / 2, _E["w_grid"][2]), 2)
LAGS = range(_E["lags"][0], _E["lags"][1] + 1)
KAPPAS = tuple(_E["kappas"])
TMAX = _E["tmax"]
SIG_A, SIG_B = _E["naive_sigma"]


def _2d(*xs):
    return [np.atleast_2d(np.asarray(x, float)) for x in xs]


def naive_signal(raw, frac, call, prior, lag):
    raw, frac, call = _2d(raw, frac, call)
    prior = np.broadcast_to(np.asarray(prior, float), (raw.shape[0],))
    raw_l = np.full_like(raw, np.nan); raw_l[:, lag:] = raw[:, :-lag]
    frac_l = np.zeros_like(frac); frac_l[:, lag:] = frac[:, :-lag]
    sig = SIG_A * np.sqrt(np.clip(1 - frac_l, 0, 1)) + SIG_B
    pn = np.where(np.isnan(raw_l), prior[:, None], norm.cdf((np.nan_to_num(raw_l, nan=0.5) - 0.5) / sig))
    return np.where(np.isnan(call), pn, call)


def model_path(fair, pn, prior, w, kappa):
    tgt = (1 - w) * fair + w * pn
    m = np.empty_like(tgt); m[:, 0] = prior
    for t in range(1, tgt.shape[1]):
        m[:, t] = m[:, t - 1] + kappa * (tgt[:, t] - m[:, t - 1])
    return m


def _loss(r, kind):
    if kind == "sse":
        return r ** 2
    if kind == "abs":
        return np.abs(r)
    d = 0.02                                   # huber: quadratic within 2c, linear beyond
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r ** 2, d * (a - 0.5 * d))


def _window_path(F, pn, M, w, kap, start):
    """Partial-adjustment path that runs from each night's window start, initialised at the observed mid."""
    tgt = (1 - w) * F + w * pn
    m = np.empty_like(tgt); m[:, 0] = M[:, 0]
    for t in range(1, tgt.shape[1]):
        prop = m[:, t - 1] + kap * (tgt[:, t] - m[:, t - 1])
        m[:, t] = np.where(t <= start, M[:, t], prop)
    return m


def estimate(fair, raw, frac, call, mid, prior, tmax=TMAX, w_grid=W_GRID, lags=LAGS, kappas=KAPPAS,
             start=None, end=None, loss="sse", return_grid=False):
    """Grid search per night. Returns dict of (n,) arrays: w, lag, kappa, sse.
    Defaults reproduce the frozen estimator. Options (estimator study, simulation/estimator_study.py):
      start : per-night first minute of the window (e.g. the first results batch); the path is initialised at the
              observed mid there and the window runs tmax minutes from it
      end   : per-night exclusive last minute (e.g. the first media call)
      loss  : "sse" (default), "abs" or "huber"
      return_grid : also return the per-night loss for every (lag, kappa, w), shape (n, lags, kappas, w)."""
    fair, mid = _2d(fair, mid)
    n = mid.shape[0]
    prior = np.broadcast_to(np.asarray(prior, float), (n,))
    best = np.full(n, np.inf); w_hat = np.zeros(n); lag_hat = np.zeros(n); kap_hat = np.zeros(n)
    lags, kappas, w_grid = list(lags), list(kappas), list(w_grid)
    grid = np.empty((n, len(lags), len(kappas), len(w_grid))) if return_grid else None
    if start is None and end is None:
        T = min(tmax, mid.shape[1]); F = fair[:, :T]; M = mid[:, :T]; keep = None
    else:
        s = np.zeros(n, int) if start is None else np.asarray(start, int)
        e = np.full(n, mid.shape[1]) if end is None else np.asarray(end, int)
        T = min(int(s.max()) + tmax, mid.shape[1]); F = fair[:, :T]; M = mid[:, :T]
        t = np.arange(T)[None]
        keep = (t > s[:, None]) & (t < np.minimum(s + tmax, e)[:, None])
    for i, lag in enumerate(lags):
        pn = naive_signal(raw, frac, call, prior, lag)[:, :T]
        for j, kap in enumerate(kappas):
            for k, w in enumerate(w_grid):
                if keep is None:
                    r = model_path(F, pn, prior, w, kap) - M
                    L = np.nansum(_loss(r, loss), axis=1)
                else:
                    r = _window_path(F, pn, M, w, kap, s) - M
                    L = np.nansum(np.where(keep, _loss(r, loss), 0.0), axis=1)
                if return_grid:
                    grid[:, i, j, k] = L
                better = L < best
                best = np.where(better, L, best); w_hat = np.where(better, w, w_hat)
                lag_hat = np.where(better, lag, lag_hat); kap_hat = np.where(better, kap, kap_hat)
    out = dict(w=w_hat, lag=lag_hat, kappa=kap_hat, sse=best)
    if return_grid:
        out["grid"] = grid; out["n_minutes"] = (np.full(n, T) if keep is None else keep.sum(1))
    return out


def estimate_diff(fair, raw, frac, call, mid, prior, start, end=None, H=5, tmax=TMAX, w_grid=W_GRID, lags=LAGS,
                  return_grid=False):
    """Change-based estimator (estimator study): over non-overlapping H-minute blocks from `start`,
        d_mid = a * [(1 - w) * d_fair + w * d_naive(lag)] + e,
    with a (speed/scale) profiled out in closed form. A persistent level gap between our fair value and the informed
    price cancels in the differences; the naive signal's jumps at each batch remain. Returns w, lag, sse
    (and the per-(lag, w) loss grid and number of blocks for pooling)."""
    fair, mid = _2d(fair, mid)
    n, T0 = mid.shape
    s = np.asarray(start, int); e = np.full(n, T0) if end is None else np.asarray(end, int)
    stop = np.minimum(np.minimum(s + tmax, e), T0 - 1)
    nb = np.maximum((stop - s) // H, 0)
    B = int(nb.max()) if n else 0
    rows = np.arange(n)[:, None]
    idx0 = np.clip(s[:, None] + H * np.arange(B)[None], 0, T0 - 1)
    idx1 = np.clip(idx0 + H, 0, T0 - 1)
    valid = np.arange(B)[None] < nb[:, None]
    dm = np.where(valid, mid[rows, idx1] - mid[rows, idx0], 0.0)
    df = np.where(valid, fair[rows, idx1] - fair[rows, idx0], 0.0)
    lags, w_grid = list(lags), np.asarray(w_grid)
    grid = np.empty((n, len(lags), len(w_grid)))
    yy = (dm ** 2).sum(1)
    for i, lag in enumerate(lags):
        pn = naive_signal(raw, frac, call, prior, lag)
        dn = np.where(valid, pn[rows, idx1] - pn[rows, idx0], 0.0)
        for k, w in enumerate(w_grid):
            x = (1 - w) * df + w * dn
            xx = (x ** 2).sum(1); xy = (x * dm).sum(1)
            grid[:, i, k] = yy - np.where(xx > 0, xy ** 2 / np.maximum(xx, 1e-15), 0.0)
    flat = grid.reshape(n, -1); j = flat.argmin(1)
    out = dict(w=w_grid[j % len(w_grid)], lag=np.asarray(lags)[j // len(w_grid)], sse=flat.min(1))
    if return_grid:
        out["grid"] = grid; out["n_blocks"] = nb
    return out


def pooled(grids, n_minutes, w_grid=W_GRID):
    """One naive share across several markets (e.g. TX and GA on the same night): each market keeps its own lag,
    kappa and noise level; w maximises the summed concentrated likelihood sum_m n_m * log(min_{lag,kappa} loss_m(w)).
    grids: list of (n, lags, kappas, w) arrays from estimate(..., return_grid=True); returns (n,) w-hat."""
    total = sum(nm[:, None] * np.log(np.maximum(g.min(axis=(1, 2)), 1e-12)) for g, nm in zip(grids, n_minutes))
    return np.asarray(w_grid)[np.argmin(total, axis=1)]


def bootstrap(fair, raw, frac, call, mid, prior, tmax=TMAX, n_boot=None, block=None, level=None, seed=0):
    """Pre-registered interval for one night: residual block bootstrap around the fitted path.
    Fit (w, lag, kappa); resample the residual series in blocks of `block` minutes; add to the fitted path;
    refit. Returns point estimate, percentile interval and the bootstrap draws."""
    b = _E["bootstrap"]; n_boot = n_boot or b["n"]; block = block or b["block_minutes"]; level = level or b["level"]
    fair, raw, frac, call, mid = _2d(fair, raw, frac, call, mid)
    T = min(tmax, mid.shape[1])
    est = estimate(fair, raw, frac, call, mid, prior, tmax)
    pn = naive_signal(raw, frac, call, prior, int(est["lag"][0]))[:, :T]
    fit = model_path(fair[:, :T], pn, np.atleast_1d(prior), est["w"][0], est["kappa"][0])[0]
    resid = mid[0, :T] - fit
    rng = np.random.default_rng(seed)
    starts = np.arange(0, T - block + 1)
    draws = np.empty((n_boot, T))
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, T // block + 1)])[:T]
        draws[i] = np.clip(fit + resid[idx], 0, 1)
    # refit all bootstrap paths at once (same fair / naive inputs, one row per draw)
    rep = lambda x: np.repeat(x[:, :T], n_boot, axis=0)
    w_b = estimate(rep(fair), rep(raw), rep(frac), rep(call), draws, prior, tmax)["w"]
    lo, hi = np.quantile(w_b, [(1 - level) / 2, 1 - (1 - level) / 2])
    return dict(w=float(est["w"][0]), lag=int(est["lag"][0]), kappa=float(est["kappa"][0]),
                lo=float(lo), hi=float(hi), level=level, draws=w_b,
                h1_reject=bool(lo > _E["h1_lower_bound"]))


if __name__ == "__main__":
    import pandas as pd
    d = pd.read_csv(sys.argv[1])   # columns: fair, raw, frac, call, mid (one row per minute from t=0)
    r = bootstrap(d.fair.values, d.raw.values, d.frac.values, d.call.values, d.mid.values, d.mid.values[0])
    print({k: v for k, v in r.items() if k != "draws"})
