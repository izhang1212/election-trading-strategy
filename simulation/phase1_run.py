"""Run the Monte Carlo: simulate nights, build markets for several naive-trader weights, run the strategies."""
import time, json, sys
import numpy as np
from phase1_sim import gen_night, run_nowcasts, ONEHOT, L_TRUE, ED_OFF_TRUE
from phase1_strategy import build_market, run_taker, run_maker, summarize

NIGHTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
T = 360
BIASED_ED = ED_OFF_TRUE + np.array([0.015, 0.015, -0.01, 0.01])
MODELS = {  # name: (our model spec, our data latency in minutes)
    "ours_good": (dict(L=ONEHOT, ed_off=ED_OFF_TRUE, var_scale=1.0), 1),        # wrong loadings only
    "ours_poor": (dict(L=ONEHOT, ed_off=BIASED_ED, var_scale=0.5), 1),          # + biased ED offsets, overconfident
    "same_as_smart": (dict(L=L_TRUE, ed_off=ED_OFF_TRUE, var_scale=1.0), 1),    # our model == smart traders' model
    "ours_good_lat0": (dict(L=ONEHOT, ed_off=ED_OFF_TRUE, var_scale=1.0), 0),   # diagnostic: as fast as smart traders
}
WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4]

rng = np.random.default_rng(7)
nights = [gen_night(rng) for _ in range(NIGHTS)]
Y = np.array([nt["outcome"] for nt in nights], float)

t0 = time.time()
cache = {}
for name, (kw, lat) in MODELS.items():
    fn = f"nowcast_{name}_{NIGHTS}.npz"
    try:
        cache[name] = dict(np.load(fn)); print(f"loaded {fn}", flush=True); continue
    except FileNotFoundError:
        pass
    rows = [run_nowcasts(nt, T, kw, latency=lat) for nt in nights]
    cache[name] = dict(P_S=np.array([r[0] for r in rows]), P_O=np.array([r[1] for r in rows]),
                       RAW=np.array([r[2] for r in rows]), FRAC=np.array([r[3] for r in rows]),
                       FLAG=np.array([r[4] for r in rows]), PRIOR=np.array([r[5] for r in rows]))
    np.savez_compressed(fn, **cache[name])
    print(f"nowcasts {name}: {time.time()-t0:.0f}s", flush=True)

# ---- calibration / accuracy checks
calib = {"D_win_rate": float(Y.mean()), "prior_p": float(cache["ours_good"]["PRIOR"].mean()),
         "final_share_sd": float(np.std([nt["final_share"] for nt in nights]))}
for name in ("ours_good", "ours_poor"):
    for tt in (10, 20, 45, 90):
        calib[f"brier_{name}_t{tt}"] = float(((cache[name]["P_O"][:, tt] - Y) ** 2).mean())
for tt in (10, 20, 45, 90):
    calib[f"brier_smart_t{tt}"] = float(((cache["ours_good"]["P_S"][:, tt] - Y) ** 2).mean())
for name in ("ours_good", "ours_poor"):
    p = cache[name]["P_O"][:, 20]; bins = np.clip((p * 5).astype(int), 0, 4)
    calib[f"reliability_{name}_t20"] = [(f"{b/5:.1f}-{(b+1)/5:.1f}", int((bins == b).sum()),
                                        round(float(Y[bins == b].mean()), 3) if (bins == b).any() else None)
                                       for b in range(5)]
# minutes until the smart fair value first leaves (0.05, 0.95) -- how fast the race "resolves"
ps = cache["ours_good"]["P_S"]
res_t = np.array([np.argmax((row < 0.05) | (row > 0.95)) for row in ps])
calib["resolve_minutes_median"] = float(np.median(res_t)); calib["resolve_minutes_p90"] = float(np.percentile(res_t, 90))
json.dump(calib, open("phase1_calib.json", "w"), indent=1)
print(json.dumps(calib, indent=1), flush=True)

results = []
for name in MODELS:
    c = cache[name]
    for w in WEIGHTS:
        mid, bid, ask, p_naive = build_market(c["P_S"], c["RAW"], c["FRAC"], c["FLAG"], c["PRIOR"], w,
                                              np.random.default_rng(99))
        gap = float(np.abs(mid - c["P_S"])[:, 1:120].max(axis=1).mean())
        strats = {
            "taker": lambda: run_taker(c["P_O"], mid, bid, ask, Y),
            "taker+filter1c": lambda: run_taker(c["P_O"], mid, bid, ask, Y, move_filter=0.01),
            "taker+filter2c": lambda: run_taker(c["P_O"], mid, bid, ask, Y, move_filter=0.02),
            "maker_touch": lambda: run_maker(c["P_O"], mid, bid, ask, Y),
            "maker_deep6c": lambda: run_maker(c["P_O"], mid, bid, ask, Y, depth=0.06),
        }
        for strat, fn in strats.items():
            s = summarize(fn()); s.update(model=name, w=w, strategy=strat, mean_max_mispricing=gap)
            results.append(s)
            print(f"{name:15s} w={w:.1f} {strat:15s} mean ${s['mean_pnl']:+6.2f} (se {s['se']:.2f}) "
                  f"win {s['win_rate']:.2f} fills {s['fills']:4.1f} fees ${s['fees']:.2f}", flush=True)
        if name == "ours_good" and w == 0.2:
            np.savez_compressed("phase1_sample_nights.npz", P_S=c["P_S"][:300], P_O=c["P_O"][:300], mid=mid[:300],
                                bid=bid[:300], ask=ask[:300], p_naive=p_naive[:300], RAW=c["RAW"][:300], FRAC=c["FRAC"][:300], Y=Y[:300])

json.dump(results, open("phase1_results.json", "w"), indent=1, default=float)
print("done", round(time.time() - t0))
