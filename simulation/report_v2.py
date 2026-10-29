"""Tables and figures for the v2 re-run -> summary_table_v2.md, figures/v2_*.png.

Reads run.py's original results (results_<TAG>.json), run_v2.py's v1 check (results_<TAG>_v1.json, which must
reproduce the original exactly) and the recalibrated v2 runs (results_<TAG>_v2.json).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COL = {"orig": "#eb6834", "v2": "#2a78d6"}   # categorical slots 2 and 1 of the reference palette
WS = ("0.0", "0.1", "0.2", "0.3", "0.4")
NAME = {"TX": "Texas Senate", "GA": "Georgia Governor"}


def load(tag):
    f = os.path.join(HERE, f"results_{tag}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def pnl(r, w, key="se"):
    x = next(x for x in r["results"] if x["bankroll"] == 100 and abs(x["w"] - float(w)) < 1e-9)
    return x["mean_pnl"], x.get(key, x["se"])


def breakeven(r, key="mean_pnl"):
    ws = [float(w) for w in WS]; m = [pnl(r, w)[0] for w in WS]
    if m[0] > 0:
        return "below 0%"
    for i in range(len(ws) - 1):
        if m[i] <= 0 < m[i + 1]:
            return f"{100 * (ws[i] + (ws[i + 1] - ws[i]) * (-m[i]) / (m[i + 1] - m[i])):.0f}%"
    return "above 40%"


def paired(S, w, base, alt, draws):
    """Mean P&L difference alt - base on the same nights, se clustered by history draw."""
    a = np.array(next(s for s in S if s["w"] == w and (s["filter_rule"], s["latency"], s["window"]) == alt)["pnl"])
    b = np.array(next(s for s in S if s["w"] == w and (s["filter_rule"], s["latency"], s["window"]) == base)["pnl"])
    d = a - b; cl = np.repeat(np.arange(draws), len(d) // draws)
    g = np.array([(d[cl == k] - d.mean()).sum() for k in range(draws)])
    return d.mean(), np.sqrt(draws / (draws - 1) * (g ** 2).sum()) / len(d)


def table():
    L = ["# Simulation re-run with real-data assumptions (v2)", "",
         "Synthetic data only. P&L per night on $100. **±** = 95% interval; in v2 standard errors are **clustered by history "
         "draw** (nights scored with the same fitted model are not independent). The original study reported iid errors; its "
         "clustered errors are shown in brackets. v2 changes: real county structure from `data/fitted/`, Election-Day-vs-early "
         "gap sd 5.5 points (was 2), year noise 2 points (was 1), 2026-10-29 prices, 20 history draws, exact 'more than 1c' "
         "move filter. Market, strategy, estimator and metric definitions are unchanged (`market.py`, `sim_strategy.py`, `run.py`).", ""]
    for st in ("TX", "GA"):
        o, c, v = load(st), load(f"{st}_v1"), load(f"{st}_v2")
        if not (o and c and v):
            L.append(f"_{st}: results missing_"); continue
        same = all(abs(x["mean_pnl"] - y["mean_pnl"]) < 1e-9 for x, y in zip(o["results"], c["results"]))
        L += [f"## {NAME[st]}", "", f"Reproduction check: `run_v2.py --scenario v1` reproduces `results_{st}.json` "
              f"{'**exactly** (every P&L figure identical)' if same else '**NOT exactly**: investigate before using v2'}.", "",
              "| Metric | Original study | v2 (recalibrated) |", "|---|---|---|"]
        od, vd = o["diag"], v["diag"]
        L.append(f"| Prior / simulated D win rate | {od['prior_p']:.2f} / {od['D_win_rate']:.1%} | {vd['prior_p']:.3f} / {vd['D_win_rate']:.1%} |")
        L.append(f"| Median minutes until decided (p25–p75) | {od['min_until_decided_median']:.0f} ({od['min_until_decided_p25']:.0f}–"
                 f"{od['min_until_decided_p75']:.0f}) | {vd['min_until_decided_median']:.0f} ({vd['min_until_decided_p25']:.0f}–{vd['min_until_decided_p75']:.0f}) |")
        L.append(f"| Nights decided within 15 / 30 min | {od['share_decided_within_15min']:.0%} / {od['share_decided_within_30min']:.0%} | "
                 f"{vd['share_decided_within_15min']:.0%} / {vd['share_decided_within_30min']:.0%} |")
        L.append(f"| Brier, our model at +15 / +30 min | {od['brier_ours_t15']:.3f} / {od['brier_ours_t30']:.3f} | "
                 f"{vd['brier_ours_t15']:.3f} / {vd['brier_ours_t30']:.3f} |")
        for w in WS:
            m0, s0 = pnl(o, w); _, sc = pnl(c, w, "se_cluster"); m2, s2 = pnl(v, w, "se_cluster")
            L.append(f"| Mean P&L, naive {float(w):.0%} | ${m0:+.2f} ± {1.96 * s0:.2f} [± {1.96 * sc:.2f}] | ${m2:+.2f} ± {1.96 * s2:.2f} |")
        L.append(f"| Break-even naive share | {breakeven(o)} | {breakeven(v)} |")
        for w in WS:
            e1, e2 = c["estimator"][w], v["estimator"][w]
            q1, q2 = (np.quantile(e["w_hat"], [0.1, 0.5, 0.9]) for e in (e1, e2))
            L.append(f"| Naive-share estimate, true {float(w):.0%}: median (p10–p90); mean ± sd | {100 * q1[1]:.0f}% ({100 * q1[0]:.0f}–"
                     f"{100 * q1[2]:.0f}); {100 * e1['mean']:.0f} ± {100 * e1['sd']:.0f} | {100 * q2[1]:.0f}% ({100 * q2[0]:.0f}–"
                     f"{100 * q2[2]:.0f}); {100 * e2['mean']:.0f} ± {100 * e2['sd']:.0f} |")
        for rule, lab in (("rule_A_reject_rate", f"H1 rule A, w-hat above the simulated null's 95th pct ({100 * c['h1']['null_q95']:.0f}% / "
                                                  f"{100 * v['h1']['null_q95']:.0f}%)"),
                          ("rule_B_reject_rate", "H1 rule B, bootstrap lower 90% bound > 10% (40 nights)")):
            L.append(f"| {lab}: % of nights rejecting at true 0/10/20/30/40% | "
                     f"{' / '.join(f'{100 * c['h1'][rule][w]:.0f}' for w in WS)} | {' / '.join(f'{100 * v['h1'][rule][w]:.0f}' for w in WS)} |")
        S = v["sensitivity"]; D = v["diag"]["draws"]; base = ("gt", 1, 1)
        L += ["", f"**{st} v2: effect of filter rule and latency** (P&L change vs. exact '>1c' filter at 1-min latency, same nights; "
              "± 95%, clustered):", "", "| Variant | naive 0% | naive 20% | naive 40% |", "|---|---|---|---|"]
        for alt, lab in ((("float", 1, 1), "original float filter"), (("ge", 1, 1), "'1c or more' filter"),
                         (("gt", 2, 1), "latency 2 min, filter 1 min"), (("gt", 2, 2), "latency 2 min, filter 2 min"),
                         (("gt", 3, 1), "latency 3 min, filter 1 min"), (("gt", 3, 3), "latency 3 min, filter 3 min"),
                         (("gt", 5, 1), "latency 5 min, filter 1 min"), (("gt", 5, 5), "latency 5 min, filter 5 min")):
            cells = [paired(S, float(w), base, alt, D) for w in ("0.0", "0.2", "0.4")]
            L.append(f"| {lab} | " + " | ".join(f"{d:+.2f} ± {1.96 * s:.2f}" for d, s in cells) + " |")
        L.append("")
    liq = load("GA_liqTX_v2")
    if liq:
        L += ["Georgia v2 with Texas-level liquidity: " + "; ".join(
            f"naive {float(w):.0%}: ${pnl(liq, w, 'se_cluster')[0]:+.2f} ± {1.96 * pnl(liq, w, 'se_cluster')[1]:.2f}" for w in ("0.0", "0.2", "0.4")), ""]
    open(os.path.join(HERE, "summary_table_v2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def style(ax):
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)


def figures():
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True); fig.patch.set_facecolor(SURF)
    for ax, st in zip(axes, ("TX", "GA")):
        style(ax); ax.axhline(0, color=INK2, lw=1)
        for key, tag, lab in (("orig", st, "original study"), ("v2", f"{st}_v2", "recalibrated (v2)")):
            r = load(tag)
            if not r:
                continue
            x = np.array([100 * float(w) for w in WS])
            m = np.array([pnl(r, w)[0] for w in WS])
            s = np.array([pnl(r, w, "se_cluster")[1] if key == "v2" else pnl(load(f"{st}_v1"), w, "se_cluster")[1] for w in WS])
            ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=COL[key], alpha=0.15, lw=0)
            ax.plot(x, m, color=COL[key], lw=2, marker="o", ms=6, label=lab)
        ax.set_title(NAME[st], loc="left", color=INK, fontsize=11); ax.set_xlabel("True naive share (%)", color=INK2)
    axes[0].set_ylabel("Mean P&L per night on $100", color=INK2)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.suptitle("Strategy P&L vs. naive share (95% bands clustered by history draw)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "v2_pnl_vs_naive.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True); fig.patch.set_facecolor(SURF)
    for ax, st in zip(axes, ("TX", "GA")):
        style(ax); ax.axhline(0, color=INK2, lw=1)
        r = load(f"{st}_v2")
        if not r:
            continue
        S, D = r["sensitivity"], r["diag"]["draws"]
        for win_eq, col, lab in ((False, COL["orig"], "filter blocks 1 min"), (True, COL["v2"], "filter blocks = latency")):
            x, y, e = [], [], []
            for Lm in (1, 2, 3, 5):
                d, s = paired(S, 0.2, ("gt", 1, 1), ("gt", Lm, Lm if win_eq else 1), D) if Lm > 1 else (0.0, 0.0)
                x.append(Lm); y.append(d); e.append(s)
            y, e = np.array(y), np.array(e)
            ax.fill_between(x, y - 1.96 * e, y + 1.96 * e, color=col, alpha=0.15, lw=0)
            ax.plot(x, y, color=col, lw=2, marker="o", ms=6, label=lab)
        ax.set_title(f"{NAME[st]}, naive share 20%", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Our data latency (minutes)", color=INK2); ax.set_xticks([1, 2, 3, 5])
    axes[0].set_ylabel("P&L change vs. 1-min latency ($ per night)", color=INK2)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.suptitle("Latency cost, recalibrated simulation (paired on the same nights, 95% bands)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "v2_latency.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    table(); figures()
