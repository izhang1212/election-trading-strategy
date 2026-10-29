"""Re-run of the Texas vs. Georgia comparison with assumptions recalibrated to real data.
  python run_v2.py TX 2000 5  --scenario v1     # = run.py exactly (check: matches results_TX.json)
  python run_v2.py TX 2000 20 --scenario v2     # recalibrated
  python run_v2.py GA 2000 20 TX --scenario v2  # Georgia with Texas-level liquidity
Mirrors run.py step for step (same world calibration, seeds, horizon, market, strategy and metrics); v2 changes:
  * units, leans, sizes, early shares and type-level ED gaps from data/fitted/ (official 2018-2024 results);
  * ED-vs-early gap sd SIG g = 0.055 (cross-cycle sd: TX 0.054, GA 0.062) and year noise 0.02 (were 0.02 / 0.01);
  * priors at 2026-10-29 Kalshi prices: TX 0.685, GA 0.615 (were 0.62 / 0.48);
  * the exact "more than 1c" move filter (run.py used a float comparison, see sim_strategy.py).
Both scenarios add: standard errors clustered by history draw (nights sharing one fitted model are not
independent), a sensitivity grid (filter rule; latency 1/2/3/5 min; filter window 1 min vs. = latency), the full
distribution of naive-share estimates at w = 0..0.4, and the size/power of two H1 decision rules.
Output: results_<TAG>_<scenario>.json (never overwrites run.py's results_<TAG>.json).
"""
import argparse, json, os, time
import numpy as np
from world import World, STATES, SIG
from model import fit_from_history, smart_params, Nowcast, run_night
from market import build_market
from sim_strategy import run_taker, summarize
from estimate_naive import estimate, _shared

ap = argparse.ArgumentParser()
ap.add_argument("code"); ap.add_argument("nights", type=int); ap.add_argument("draws", type=int)
ap.add_argument("liq", nargs="?"); ap.add_argument("--scenario", default="v2", choices=["v1", "v2"])
ap.add_argument("--boot_nights", type=int, default=40)
ap.add_argument("--save_nights", type=int, default=0, help="save the first N nights' arrays to nights_<TAG>_<scenario>.npz and exit")
a = ap.parse_args()
code, NIGHTS, DRAWS = a.code, a.nights, a.draws
LIQ = a.liq or code
TAG = code if LIQ == code else f"{code}_liq{LIQ}"
HERE = os.path.dirname(os.path.abspath(__file__))
T = 420
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4]
BANKROLLS = [100, 10_000]
PRIORS_V2 = {"TX": 0.685, "GA": 0.615}
RULE = None if a.scenario == "v1" else "gt"
t0 = time.time()

if a.scenario == "v2":
    SIG.update(g=0.055, ed_year=0.02)
    w = World.from_fitted(code, os.path.join(HERE, "..", "data", "fitted"), n_units={"TX": 60, "GA": 55}[code],
                          prior_p=PRIORS_V2[code])
else:
    w = World(code)
perfect = Nowcast(w, dict(lean=w.lean, L=w.L, ed_off=w.ed_off, ef=w.ef_mean), w.cfg["prior_p"])
perfect.calibrate(w.V * w.ef_mean); w.shift = perfect.shift

mcfg = dict(w.cfg); mcfg.update({k: STATES[LIQ][k] for k in ("spread_base", "spread_post", "depth")})
keys = ["P_S", "P_O", "P_I", "RAW", "FRAC", "FLAG"]
R = {k: [] for k in keys}; R["PRIOR"] = []
Y = []; final = []; nb = []; fits = []; DRAW = []
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
        DRAW.append(dsd)
NIGHTS = len(Y)
R = {k: np.array(v) for k, v in R.items()}
Y = np.array(Y, float); final = np.array(final); margin = 200 * (final - 0.5); DRAW = np.array(DRAW)
print(f"{TAG} {a.scenario}: {w.N} units, nowcasts done {time.time()-t0:.0f}s", flush=True)
if a.save_nights:   # for estimator_study.py: same nights, same seeds as the full run
    n = a.save_nights
    np.savez_compressed(os.path.join(HERE, f"nights_{TAG}_{a.scenario}.npz"), **{k: v[:n] for k, v in R.items()},
                        Y=Y[:n], DRAW=DRAW[:n], spread_base=mcfg["spread_base"], spread_post=mcfg["spread_post"])
    raise SystemExit(0)

# diagnostics (definitions as run.py)
diag = dict(race=w.cfg["race"], scenario=a.scenario, nights=NIGHTS, draws=DRAWS, units=w.N, prior_p=w.cfg["prior_p"],
            D_win_rate=float(Y.mean()), margin_sd=float(margin.std()), batches_per_night=float(np.mean(nb)),
            sig=dict(SIG), fit=dict(ed_off_true=np.asarray(w.ed_off).round(4).tolist(), draws=fits))
for tt in (15, 30, 60, 120):
    diag[f"brier_smart_t{tt}"] = float(((R["P_S"][:, tt] - Y) ** 2).mean())
    diag[f"brier_ours_t{tt}"] = float(((R["P_I"][:, tt] - Y) ** 2).mean())
ps = R["P_S"]
out = (ps < 0.10) | (ps > 0.90)
dec_t = np.where(out.any(1), out.argmax(1), T)
diag.update(min_until_decided_median=float(np.median(dec_t)), min_until_decided_p25=float(np.percentile(dec_t, 25)),
            min_until_decided_p75=float(np.percentile(dec_t, 75)), share_decided_within_15min=float((dec_t <= 15).mean()),
            share_decided_within_30min=float((dec_t <= 30).mean()), share_never_decided_by_2h=float((dec_t > 120).mean()))


def lagged(L):
    if L == 1:
        return R["P_O"]
    P = np.empty_like(R["P_I"]); P[:, :L] = R["P_O"][:, :1]; P[:, L:] = R["P_I"][:, :-L]
    return P


# markets, strategy, metrics, sensitivity, estimator
results, sens, est = [], [], {}
for wt in WEIGHTS:
    mkt = build_market(R, wt, mcfg, np.random.default_rng(99))
    gap = np.abs(mkt["mid"] - R["P_S"])
    for br in BANKROLLS:
        res = run_taker(R["P_O"], mkt, Y, br, mcfg["depth"], filter_rule=RULE)
        s = summarize(res, br); s["se_cluster"] = summarize(res, br, cluster=DRAW)["se"]
        s.update(state=code, w=wt, bankroll=br, mean_max_mispricing=float(gap[:, 1:120].max(1).mean()),
                 minutes_mispriced_5c=float((gap[:, 1:240] > 0.05).sum(1).mean()))
        results.append(s)
        if br == 100:
            print(f"{TAG} {a.scenario} w={wt:.1f}: mean {s['mean_pnl']:+.2f} (se iid {s['se']:.2f}, clustered {s['se_cluster']:.2f})",
                  flush=True)
    for rule in (None, "gt", "ge"):
        for L in (1, 2, 3, 5):
            for win in sorted({1, L}):
                if rule != "gt" and (L, win) != (1, 1):
                    continue
                res = run_taker(lagged(L), mkt, Y, 100, mcfg["depth"], filter_rule=rule, filter_window=win)
                sens.append(dict(w=wt, filter_rule=rule or "float", latency=L, window=win, mean_pnl=float(res["pnl"].mean()),
                                 se_cluster=summarize(res, 100, cluster=DRAW)["se"], pnl=res["pnl"].round(4).tolist()))
    m300 = {k: v[:300] for k, v in R.items()}
    wh = estimate(m300, mkt["mid"][:300], mkt["called"][:300])
    est[str(wt)] = dict(mean=float(wh.mean()), sd=float(wh.std()), within_10pts=float((np.abs(wh - wt) <= 0.10).mean()),
                        median=float(np.median(wh)), w_hat=wh.round(3).tolist())
    # H1 rule B (bootstrap lower 90% bound > 0.10) on the first boot_nights nights
    rej = []
    for i in range(min(a.boot_nights, 300)):
        call_i = np.where(mkt["called"][i], (R["P_S"][i] > 0.5).astype(float), np.nan)
        b = _shared.bootstrap(R["P_I"][i], R["RAW"][i], R["FRAC"][i], call_i, mkt["mid"][i], R["PRIOR"][i], n_boot=100, seed=i)
        rej.append(b["h1_reject"])
    est[str(wt)]["bootstrap_rule_reject_rate"] = float(np.mean(rej))
    print(f"{TAG} {a.scenario} estimator w={wt}: mean {wh.mean():.3f} sd {wh.std():.3f} median {np.median(wh):.3f}; "
          f"bootstrap-rule rejects {np.mean(rej):.0%}", flush=True)

null = np.array(est["0.0"]["w_hat"])
h1 = dict(null_q90=float(np.quantile(null, 0.90)), null_q95=float(np.quantile(null, 0.95)))
h1["rule_A_reject_rate"] = {k: float((np.array(v["w_hat"]) > h1["null_q95"]).mean()) for k, v in est.items()}
h1["rule_B_reject_rate"] = {k: v["bootstrap_rule_reject_rate"] for k, v in est.items()}
json.dump(dict(diag=diag, results=results, sensitivity=sens, estimator=est, h1=h1),
          open(os.path.join(HERE, f"results_{TAG}_{a.scenario}.json"), "w"), indent=1, default=float)
print(f"{TAG} {a.scenario}: done {time.time()-t0:.0f}s", flush=True)
