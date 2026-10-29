"""Which version of the naive-share estimator to freeze, and how H1 is decided.

Uses the v2 nights saved by `run_v2.py ... --save_nights 2000` and the original market (market.py, seed 99, as
run.py). For each estimator variant and true naive share w it records the distribution of w-hat. H1 rule A is
then calibrated without reusing nights: the 95th percentile of w-hat under w = 0 on nights 300-599 is the
threshold; size and power are measured on nights 0-299. Pooled = one w across the TX and GA nights paired by
index (independent worlds, same true w).

Selection criterion (fixed before looking at results): highest average power over w = 0.1-0.4 at 5% size,
with |median bias| <= 2 points at every w. Output: estimator_study.json, estimator_study.md
Usage: python estimator_study.py
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4)
N_EVAL, N_CAL = 300, 300
FINE_K = tuple(np.round(np.arange(0.2, 1.01, 0.1), 2))
VARIANTS = {   # name: (start at first batch, end at media call, loss, kappas)
    "base":            (False, False, "sse", None),
    "start":           (True, False, "sse", None),
    "start+end":       (True, True, "sse", None),
    "start+huber":     (True, False, "huber", None),
    "start+abs":       (True, False, "abs", None),
    "start+fineK":     (True, False, "sse", FINE_K),
    "start+huber+fineK": (True, False, "huber", FINE_K),
    "diff3":           (True, False, "diff", 3),       # change-based: H-minute blocks from the first batch
    "diff5":           (True, False, "diff", 5),
    "diff10":          (True, False, "diff", 10),
    "diff5+end":       (True, True, "diff", 5),
}


def load(state):
    d = np.load(os.path.join(HERE, f"nights_{state}_v2.npz"))
    return {k: d[k] for k in d.files}


def task(args):
    state, name, w, lo, hi = args
    import estimate_naive as E                       # simulation wrapper -> shared top-level estimator
    from market import build_market
    from world import STATES
    D = load(state); R = {k: D[k][lo:hi] for k in ("P_S", "P_O", "P_I", "RAW", "FRAC", "FLAG", "PRIOR")}
    cfg = dict(STATES[state], spread_base=float(D["spread_base"]), spread_post=float(D["spread_post"]))
    full = {k: D[k] for k in ("P_S", "P_O", "P_I", "RAW", "FRAC", "FLAG", "PRIOR")}
    mkt = build_market(full, w, cfg, np.random.default_rng(99))      # same market draw as run_v2.py
    mid, called = mkt["mid"][lo:hi], mkt["called"][lo:hi]
    st, en, loss, kap = VARIANTS[name]
    call = np.where(called, (R["P_S"] > 0.5).astype(float), np.nan)
    start = np.argmax(R["FLAG"], axis=1) if st else None
    end = np.where(called.any(1), np.argmax(called, axis=1), mid.shape[1]) if en else None
    if loss == "diff":
        r = E._shared.estimate_diff(R["P_I"], R["RAW"], R["FRAC"], call, mid, R["PRIOR"], start, end, H=kap, return_grid=True)
        return (state, name, w, lo), dict(w_hat=r["w"], grid=r["grid"].min(axis=1), n_minutes=r["n_blocks"])
    kw = dict(start=start, end=end, loss=loss, return_grid=True)
    if kap is not None:
        kw["kappas"] = kap
    r = E._shared.estimate(R["P_I"], R["RAW"], R["FRAC"], call, mid, R["PRIOR"], **kw)
    return (state, name, w, lo), dict(w_hat=r["w"], grid=r["grid"].min(axis=(1, 2)), n_minutes=r["n_minutes"])


def main():
    t0 = time.time()
    jobs = [(s, v, w, 0, N_EVAL) for s in ("TX", "GA") for v in VARIANTS for w in WEIGHTS] + \
           [(s, v, 0.0, N_EVAL, N_EVAL + N_CAL) for s in ("TX", "GA") for v in VARIANTS]
    res = {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for key, val in ex.map(task, jobs):
            res[key] = val
            print(f"{key} done {time.time() - t0:.0f}s", flush=True)
    import estimate_naive as E
    out = {}
    for v in VARIANTS:
        for state in ("TX", "GA", "pooled"):
            if state == "pooled":
                def wh(w, lo):
                    a, b = res[("TX", v, w, lo)], res[("GA", v, w, lo)]
                    tot = a["n_minutes"][:, None] * np.log(np.maximum(a["grid"], 1e-12)) + \
                          b["n_minutes"][:, None] * np.log(np.maximum(b["grid"], 1e-12))
                    return E.W_GRID[np.argmin(tot, axis=1)]
            else:
                def wh(w, lo, state=state):
                    return res[(state, v, w, lo)]["w_hat"]
            thr = float(np.quantile(wh(0.0, N_EVAL), 0.95))
            rows = {}
            for w in WEIGHTS:
                x = wh(w, 0)
                rows[str(w)] = dict(median=float(np.median(x)), p10=float(np.quantile(x, 0.1)), p90=float(np.quantile(x, 0.9)),
                                    mean=float(x.mean()), sd=float(x.std()), reject=float((x > thr).mean()))
            power = float(np.mean([rows[str(w)]["reject"] for w in WEIGHTS[1:]]))
            maxbias = float(max(abs(rows[str(w)]["median"] - w) for w in WEIGHTS))
            out[f"{v}|{state}"] = dict(variant=v, state=state, threshold=thr, rows=rows, avg_power=power, max_median_bias=maxbias)
    json.dump(out, open(os.path.join(HERE, "estimator_study.json"), "w"), indent=1)
    L = ["# Estimator study (v2 nights, original market)", "",
         "Rule A threshold = 95th percentile of w-hat at w = 0 on calibration nights 300-599; rejection rates on nights "
         "0-299. Pooled = TX and GA nights paired, one shared w. Selection rule (fixed in advance): highest average power "
         "over w = 10-40% subject to size and |median bias| <= 2 points.", "",
         "| Variant | Market | Threshold | Reject at 0 / 10 / 20 / 30 / 40% | Avg power | Median (p10-p90) at 20% | Max median bias |",
         "|---|---|---|---|---|---|---|"]
    for k, r in out.items():
        rr = r["rows"]
        L.append(f"| {r['variant']} | {r['state']} | {100 * r['threshold']:.0f}% | " +
                 " / ".join(f"{100 * rr[str(w)]['reject']:.0f}" for w in WEIGHTS) +
                 f" | {100 * r['avg_power']:.0f}% | {100 * rr['0.2']['median']:.0f}% ({100 * rr['0.2']['p10']:.0f}-{100 * rr['0.2']['p90']:.0f}) | "
                 f"{100 * r['max_median_bias']:.0f} pts |")
    open(os.path.join(HERE, "estimator_study.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
