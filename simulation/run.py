"""End-to-end pipeline for one state:  python run.py TX 2000 [history_draws=5] [liquidity_from=TX|GA]
Nights are split across several independent history draws, so results don't hinge on one lucky/unlucky model fit.
  1 data pull   : synthetic 2018-2024 county history + 2026 early-vote turnout feed
  2 fit         : our county leans / ED offsets / early shares from that history
  3 nowcast     : minute-by-minute fair value on each simulated election night (1-min latency)
  4 market      : informed + naive traders, state-specific spreads and depth
  5 strategy    : taker with move filter, quarter Kelly, fees; $100 and $10,000 bankrolls
  6 metrics     : P&L, risk, trading window, model calibration, naive-share estimator precision
"""
import sys, json, time
import numpy as np
from world import World
from model import fit_from_history, smart_params, Nowcast, run_night
from market import build_market
from sim_strategy import run_taker, summarize
from estimate_naive import estimate

code = sys.argv[1]; NIGHTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
LIQ = sys.argv[4] if len(sys.argv) > 4 else code
TAG = code if LIQ == code else f"{code}_liq{LIQ}"
T = 420
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4]
BANKROLLS = [100, 10_000]
t0 = time.time()

w = World(code)
perfect = Nowcast(w, dict(lean=w.lean, L=w.L, ed_off=w.ed_off, ef=w.ef_mean), w.cfg["prior_p"])
perfect.calibrate(w.V * w.ef_mean); w.shift = perfect.shift

# 1-3: for each history draw -> data pull, fit, then nowcast that draw's share of nights
from world import STATES
mcfg = dict(w.cfg); mcfg.update({k: STATES[LIQ][k] for k in ("spread_base", "spread_post", "depth")})
keys = ["P_S", "P_O", "P_I", "RAW", "FRAC", "FLAG"]
R = {k: [] for k in keys}; R["PRIOR"] = []
Y = []; final = []; nb = []; fits = []
rel_true = w.lean - np.average(w.lean, weights=w.V)
for dsd in range(DRAWS):
    rng = np.random.default_rng(7 + 1000 * dsd)
    hist = w.history(rng)
    fit = fit_from_history(w, hist)
    fits.append(dict(lean_err_sd=float(np.std(fit["lean"] - np.average(fit["lean"], weights=w.V) - rel_true)),
                     ed_off_hat=fit["ed_off"].round(4).tolist(), ef_err_sd=float(np.std(fit["ef"] - w.ef_mean))))
    smart = Nowcast(w, smart_params(w, rng), w.cfg["prior_p"])
    ours = Nowcast(w, fit, w.cfg["prior_p"])
    for k in range(NIGHTS // DRAWS):
        nt = w.night(rng)
        smart.calibrate(nt["early_known"]); ours.calibrate(nt["early_known"])
        r = run_night(w, nt, smart, ours, T)
        for kk in keys: R[kk].append(r[kk])
        R["PRIOR"].append(r["PRIOR"]); Y.append(nt["outcome"]); final.append(nt["final_share"]); nb.append(nt["n_batches"])
fit_quality = dict(ed_off_true=w.ed_off.tolist(), draws=fits)
NIGHTS = len(Y)
R = {k: np.array(v) for k, v in R.items()}
Y = np.array(Y, float); final = np.array(final); margin = 200 * (final - 0.5)
print(f"{TAG}: nowcasts done {time.time()-t0:.0f}s", flush=True)

# diagnostics: model accuracy and how long the race stays tradeable
diag = dict(race=w.cfg["race"], nights=NIGHTS, units=w.N, prior_p=w.cfg["prior_p"], D_win_rate=float(Y.mean()),
            margin_sd=float(margin.std()), batches_per_night=float(np.mean(nb)), fit=fit_quality)
for tt in (15, 30, 60, 120):
    diag[f"brier_smart_t{tt}"] = float(((R["P_S"][:, tt] - Y) ** 2).mean())
    diag[f"brier_ours_t{tt}"] = float(((R["P_I"][:, tt] - Y) ** 2).mean())
ps = R["P_S"]
out = (ps < 0.10) | (ps > 0.90)
dec_t = np.where(out.any(1), out.argmax(1), T)
diag["min_until_decided_median"] = float(np.median(dec_t)); diag["min_until_decided_p25"] = float(np.percentile(dec_t, 25))
diag["min_until_decided_p75"] = float(np.percentile(dec_t, 75))
diag["share_decided_within_15min"] = float((dec_t <= 15).mean())
diag["share_decided_within_30min"] = float((dec_t <= 30).mean())
diag["share_never_decided_by_2h"] = float((dec_t > 120).mean())
diag["batches_before_decided"] = float(np.mean([R["FLAG"][i, :dec_t[i] + 1].sum() for i in range(NIGHTS)]))
p = R["P_I"][:, 30]; bins = np.clip((p * 5).astype(int), 0, 4)
diag["reliability_ours_t30"] = [[f"{b/5:.1f}-{(b+1)/5:.1f}", int((bins == b).sum()),
                                 round(float(Y[bins == b].mean()), 3) if (bins == b).any() else None] for b in range(5)]

# 4-6: markets, strategy, metrics
buckets = {"D by >6": margin > 6, "D by 2-6": (margin > 2) & (margin <= 6), "D by <2": (margin > 0) & (margin <= 2),
           "R by <2": (margin <= 0) & (margin > -2), "R by 2-6": (margin <= -2) & (margin > -6), "R by >6": margin <= -6}
results = []; by_bucket = {}; est = {}
for wt in WEIGHTS:
    mkt = build_market(R, wt, mcfg, np.random.default_rng(99))
    gap = np.abs(mkt["mid"] - R["P_S"])
    for br in BANKROLLS:
        res = run_taker(R["P_O"], mkt, Y, br, mcfg["depth"])
        s = summarize(res, br); s.update(state=code, w=wt, bankroll=br,
                                         mean_max_mispricing=float(gap[:, 1:120].max(1).mean()),
                                         minutes_mispriced_5c=float((gap[:, 1:240] > 0.05).sum(1).mean()))
        results.append(s)
        if br == 100:
            by_bucket[str(wt)] = {name: dict(n=int(m.sum()), mean_pnl=float(res["pnl"][m].mean()),
                                             se=float(res["pnl"][m].std(ddof=1) / np.sqrt(max(m.sum(), 2))))
                                  for name, m in buckets.items() if m.sum() > 1}
        print(f"{code} w={wt:.1f} ${br:>6}: mean {s['mean_pnl']:+9.2f} (se {s['se']:.2f}) ret {s['mean_ret_pct']:+.2f}% "
              f"win {s['win_rate']:.2f} fills {s['fills']:.1f} fees {s['fees']:.2f}", flush=True)
    if wt in (0.0, 0.2, 0.4):
        m300 = {k: v[:300] for k, v in R.items()}
        wh = estimate(m300, mkt["mid"][:300], mkt["called"][:300])
        est[str(wt)] = dict(mean=float(wh.mean()), sd=float(wh.std()), within_10pts=float((np.abs(wh - wt) <= 0.10).mean()))
        print(f"{code} estimator w={wt}: mean {wh.mean():.3f} sd {wh.std():.3f}", flush=True)
        if wt == 0.2:
            np.savez_compressed(f"sample_{TAG}.npz", P_S=R["P_S"][:200], P_O=R["P_O"][:200], RAW=R["RAW"][:200],
                                FRAC=R["FRAC"][:200], mid=mkt["mid"][:200], bid=mkt["bid"][:200], ask=mkt["ask"][:200],
                                Y=Y[:200], margin=margin[:200])

json.dump(dict(diag=diag, results=results, by_bucket=by_bucket, estimator=est), open(f"results_{TAG}.json", "w"),
          indent=1, default=float)
print(f"{code}: done {time.time()-t0:.0f}s")
