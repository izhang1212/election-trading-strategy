"""Does modelling a statewide early-vote shift make our fair value closer to the informed traders'?

Real early shares of the two-party vote move statewide between cycles (TX 87/68/81% in 2020/22/24; GA 54/81/64/77%
in 2018-24), and 2026 early turnout is far above 2022's. run_v2's world and model both treat Election Day volume
as independent by county. Here the v2 world adds a statewide early-share shift (SIG ef_common, sd 0.05); the
informed traders model it (correlated ED-volume uncertainty + learning it from reporting counties); our model is
run three ways on the same nights:
  base     : as frozen (independent county ED volume)
  common   : + correlated ED-volume uncertainty (relative sd 0.25 for not-yet-reporting counties)
  learn    : + learn the statewide ED-volume ratio from counties already reporting ED votes
Also run on a world without the shift (TX), to check the fix costs little when there is no surge.
Metrics: Brier vs. outcome, gap to the informed fair value, naive-share estimator tail at w = 0 and spread at
w = 0.2 (frozen estimator), P&L change vs. base (paired, clustered by history draw).
Output: surge_study.md / .json. Usage: python surge_study.py
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PRIORS = {"TX": 0.685, "GA": 0.615}
VARIANTS = {"base": {}, "common": dict(ed_common=0.25), "learn": dict(ed_common=0.25, ed_learn=True)}
SMART = dict(ed_common=0.25, ed_learn=True)
DRAWS, PER, T = 10, 100, 420


def simulate(args):
    state, surge = args
    from world import World, SIG, STATES
    from model import fit_from_history, smart_params, Nowcast, run_night
    SIG.update(g=0.055, ed_year=0.02, ef_common=surge)
    w = World.from_fitted(state, os.path.join(HERE, "..", "data", "fitted"), n_units={"TX": 60, "GA": 55}[state],
                          prior_p=PRIORS[state])
    perfect = Nowcast(w, dict(lean=w.lean, L=w.L, ed_off=w.ed_off, ef=w.ef_mean), w.cfg["prior_p"])
    perfect.calibrate(w.V * w.ef_mean); w.shift = perfect.shift
    out = {v: {k: [] for k in ("P_S", "P_O", "P_I", "RAW", "FRAC", "FLAG", "PRIOR")} for v in VARIANTS}
    Y, DRAW = [], []
    for d in range(DRAWS):
        rng = np.random.default_rng(7 + 1000 * d)
        fit = fit_from_history(w, w.history(rng))
        sp = dict(smart_params(w, rng), **SMART)
        nights = [w.night(rng) for _ in range(PER)]
        for v, extra in VARIANTS.items():
            smart = Nowcast(w, sp, w.cfg["prior_p"]); ours = Nowcast(w, dict(fit, **extra), w.cfg["prior_p"])
            for nt in nights:
                smart.calibrate(nt["early_known"]); ours.calibrate(nt["early_known"])
                r = run_night(w, nt, smart, ours, T)
                for k in out[v]:
                    out[v][k].append(r[k])
        Y += [nt["outcome"] for nt in nights]; DRAW += [d] * PER
    return (state, surge), dict(R={v: {k: np.array(x) for k, x in o.items()} for v, o in out.items()},
                                Y=np.array(Y, float), DRAW=np.array(DRAW), cfg={k: STATES[state][k] for k in ("spread_base", "spread_post", "depth")})


def clustered(d, cl):
    g = np.unique(cl); G = len(g)
    r = np.array([(d[cl == k] - d.mean()).sum() for k in g])
    return np.sqrt(G / (G - 1) * (r ** 2).sum()) / len(d)


def main():
    t0 = time.time()
    jobs = [("TX", 0.05), ("GA", 0.05), ("TX", 0.0)]
    with ProcessPoolExecutor(3) as ex:
        sims = dict(ex.map(simulate, jobs))
    print(f"simulated {time.time() - t0:.0f}s", flush=True)
    import estimate_naive as E
    from market import build_market
    from sim_strategy import run_taker
    res, L = {}, ["# Early-vote surge study", "",
                  "v2 world plus a statewide early-share shift (sd 5 pts) unless noted; informed traders model it. "
                  "Our model three ways on the same nights and market. P&L change vs. base on $100, 95% clustered.", "",
                  "| World | Our model | Brier +15 / +30 / +60 | Mean abs gap to informed, first 60 min | Estimator at w=0: p90 / p95 / share >= 30% | at w=0.2: p10-p90 | P&L change at w = 0 / 0.2 / 0.4 |",
                  "|---|---|---|---|---|---|---|"]
    for (state, surge), S in sims.items():
        Y, DRAW, cfg = S["Y"], S["DRAW"], S["cfg"]
        base_pnl = {}
        for v in VARIANTS:
            R = S["R"][v]
            brier = [float(((R["P_I"][:, t] - Y) ** 2).mean()) for t in (15, 30, 60)]
            gap = float(np.abs(R["P_I"][:, :60] - R["P_S"][:, :60]).mean())
            est, dpnl = {}, {}
            for wt in (0.0, 0.2, 0.4):
                mkt = build_market(R, wt, dict(cfg), np.random.default_rng(99))
                pnl = run_taker(R["P_O"], mkt, Y, 100, cfg["depth"], filter_rule="gt")["pnl"]
                if v == "base":
                    base_pnl[wt] = pnl
                d = pnl - base_pnl[wt]; dpnl[wt] = (float(d.mean()), float(clustered(d, DRAW)))
                if wt < 0.3:
                    n = 300; call = np.where(mkt["called"][:n], (R["P_S"][:n] > 0.5).astype(float), np.nan)
                    est[wt] = E._shared.estimate(R["P_I"][:n], R["RAW"][:n], R["FRAC"][:n], call, mkt["mid"][:n], R["PRIOR"][:n])["w"]
            e0, e2 = est[0.0], est[0.2]
            res[f"{state}|{surge}|{v}"] = dict(brier=brier, gap=gap, est0=[float(np.quantile(e0, q)) for q in (0.9, 0.95)],
                                               est0_ge30=float((e0 >= 0.3).mean()),
                                               est2=[float(np.quantile(e2, q)) for q in (0.1, 0.9)], dpnl=dpnl)
            r = res[f"{state}|{surge}|{v}"]
            L.append(f"| {state} {'surge' if surge else 'no surge'} | {v} | " + " / ".join(f"{b:.3f}" for b in brier) +
                     f" | {100 * gap:.1f} pts | {100 * r['est0'][0]:.0f}% / {100 * r['est0'][1]:.0f}% / {100 * r['est0_ge30']:.0f}% | "
                     f"{100 * r['est2'][0]:.0f}-{100 * r['est2'][1]:.0f}% | " +
                     " / ".join(f"{m:+.2f} ± {1.96 * s:.2f}" for m, s in dpnl.values()) + " |")
    json.dump(res, open(os.path.join(HERE, "surge_study.json"), "w"), indent=1)
    open(os.path.join(HERE, "surge_study.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
