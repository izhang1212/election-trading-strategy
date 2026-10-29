"""Calibrate H1 rule B: the residual-bootstrap lower bound of w (estimate_naive.bootstrap) is too liberal at the
nominal 10% cut-off. Find the cut-off c with 5% false positives under w = 0 on calibration nights 300-499, then
measure power on nights 0-99 at w = 0.1..0.4. Per state, v2 nights, original market (seed 99).
Output: boot_calibration.json. Usage: python boot_calibration.py
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4)


def lower_bound(args):
    state, w, i = args
    import estimate_naive as E
    from market import build_market
    from world import STATES
    D = np.load(os.path.join(HERE, f"nights_{state}_v2.npz"))
    R = {k: D[k] for k in ("P_S", "P_O", "P_I", "RAW", "FRAC", "FLAG", "PRIOR")}
    cfg = dict(STATES[state], spread_base=float(D["spread_base"]), spread_post=float(D["spread_post"]))
    mkt = build_market(R, w, cfg, np.random.default_rng(99))
    call = np.where(mkt["called"][i], (R["P_S"][i] > 0.5).astype(float), np.nan)
    b = E._shared.bootstrap(R["P_I"][i], R["RAW"][i], R["FRAC"][i], call, mkt["mid"][i], R["PRIOR"][i], n_boot=100, seed=i)
    return (state, w, i), (b["lo"], b["w"])


def main():
    t0 = time.time()
    jobs = [(s, 0.0, i) for s in ("TX", "GA") for i in range(300, 500)] + \
           [(s, w, i) for s in ("TX", "GA") for w in WEIGHTS for i in range(100)]
    res = {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
        for k, v in ex.map(lower_bound, jobs, chunksize=4):
            res[k] = v
    out = {}
    for s in ("TX", "GA"):
        null_lb = np.array([res[(s, 0.0, i)][0] for i in range(300, 500)])
        c = float(np.quantile(null_lb, 0.95))
        rows = {}
        for w in WEIGHTS:
            lb = np.array([res[(s, w, i)][0] for i in range(100)])
            rows[str(w)] = dict(reject_calibrated=float((lb > c).mean()), reject_nominal_0p10=float((lb > 0.10).mean()),
                                median_lb=float(np.median(lb)))
        out[s] = dict(cutoff_5pct=c, rows=rows,
                      avg_power_calibrated=float(np.mean([rows[str(w)]["reject_calibrated"] for w in WEIGHTS[1:]])))
        print(s, f"cut-off for 5% size: lower bound > {c:.3f}", {w: (round(r["reject_calibrated"], 2), round(r["reject_nominal_0p10"], 2))
                                                             for w, r in rows.items()}, flush=True)
    json.dump(out, open(os.path.join(HERE, "boot_calibration.json"), "w"), indent=1)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
