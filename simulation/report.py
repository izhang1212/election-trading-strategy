"""Figures + summary table comparing Texas Senate and Georgia Governor."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"TX": "#2a78d6", "GA": "#eb6834"}
NAME = {"TX": "Texas Senate", "GA": "Georgia Governor"}
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
R = {c: json.load(open(f"results_{c}.json")) for c in ("TX", "GA")}


def style(ax):
    ax.set_axisbelow(True); ax.set_facecolor(BG); ax.grid(color=GRID, lw=0.8); ax.tick_params(colors=INK2)
    for s in ax.spines.values(): s.set_visible(False)


# ---- Figure 1: mean P&L vs naive share ($100), and return % at $10k
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2)); fig.patch.set_facecolor(BG)
for ax, br, ylab in ((axes[0], 100, r"Mean P&L per night (\$), \$100 bankroll"),
                     (axes[1], 10_000, r"Mean return per night (%), \$10,000 bankroll")):
    style(ax)
    for c in ("TX", "GA"):
        rows = [r for r in R[c]["results"] if r["bankroll"] == br]
        x = np.array([r["w"] for r in rows]) * 100
        if br == 100:
            y = np.array([r["mean_pnl"] for r in rows]); e = 1.96 * np.array([r["se"] for r in rows])
        else:
            y = np.array([r["mean_ret_pct"] for r in rows]); e = 1.96 * np.array([r["se"] for r in rows]) / br * 100
        ax.errorbar(x, y, yerr=e, color=C[c], lw=2, marker="o", ms=6, capsize=3, label=NAME[c])
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Naive traders' share of price formation (%)", color=INK2); ax.set_ylabel(ylab, color=INK2)
    ax.set_xlim(-3, 52)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Strategy P&L by market and naive-trader share (2,000 simulated nights each; bars = 95% CI)",
             color=INK, fontsize=11, x=0.01, ha="left")
fig.tight_layout(); fig.savefig("figures/fig_pnl_vs_naive.png", dpi=150)

# ---- Figure 2: P&L by outcome bucket at 20% naive
fig, ax = plt.subplots(figsize=(10, 4)); fig.patch.set_facecolor(BG); style(ax)
names = ["R by >6", "R by 2-6", "R by <2", "D by <2", "D by 2-6", "D by >6"]
xs = np.arange(len(names)); wbar = 0.38
for k, c in enumerate(("TX", "GA")):
    b = R[c]["by_bucket"]["0.2"]
    y = [b[n]["mean_pnl"] if n in b else np.nan for n in names]
    e = [1.96 * b[n]["se"] if n in b else 0 for n in names]
    ax.bar(xs + (k - 0.5) * wbar, y, wbar - 0.04, yerr=e, color=C[c], label=NAME[c], capsize=2,
           error_kw=dict(ecolor=INK2, lw=1))
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"{n} pts" for n in names], color=INK2)
ax.set_ylabel(r"Mean P&L per night (\$), \$100 bankroll", color=INK2)
ax.legend(frameon=False)
ax.set_title("Where the money is made: P&L by final margin (naive share 20%)", color=INK, fontsize=11, loc="left")
fig.tight_layout(); fig.savefig("figures/fig_pnl_by_margin.png", dpi=150)

# ---- Figure 3: how long the race stays uncertain (informed price inside 10-90%)
fig, ax = plt.subplots(figsize=(8, 4)); fig.patch.set_facecolor(BG); style(ax)
for c in ("TX", "GA"):
    s = np.load(f"sample_{c}.npz")
    ps = s["P_S"]; out = (ps < 0.10) | (ps > 0.90)
    dec = np.where(out.any(1), out.argmax(1), ps.shape[1])
    t = np.arange(0, 181)
    still = [(dec > tt).mean() * 100 for tt in t]
    ax.plot(t, still, color=C[c], lw=2, label=NAME[c])
ax.set_xlim(0, 90); ax.legend(frameon=False)
ax.set_xlabel("Minutes after first poll close (200 sample nights per state)", color=INK2)
ax.set_ylabel("Nights still uncertain (%)\n(informed price between 10% and 90%)", color=INK2)
ax.set_ylim(0, 100)
ax.set_title("Trading window: share of nights still undecided over time", color=INK, fontsize=11, loc="left")
fig.tight_layout(); fig.savefig("figures/fig_trading_window.png", dpi=150)

# ---- summary table
print("| Metric | Texas Senate | Georgia Governor |")
print("| --- | --- | --- |")
def row(label, f):
    print(f"| {label} | {f('TX')} | {f('GA')} |")
d = {c: R[c]["diag"] for c in R}
get = lambda c, w, br, k: [r for r in R[c]["results"] if r["w"] == w and r["bankroll"] == br][0][k]
row("Pre-election price (Dem)", lambda c: f"{d[c]['prior_p']:.0%}")
row("Simulated D win rate", lambda c: f"{d[c]['D_win_rate']:.1%}")
row("Batches per night", lambda c: f"{d[c]['batches_per_night']:.0f}")
row("Median minutes until decided (10/90%)", lambda c: f"{d[c]['min_until_decided_median']:.0f}")
row("Nights decided within 15 min", lambda c: f"{d[c]['share_decided_within_15min']:.0%}")
row("Nights still open after 2 h", lambda c: f"{d[c]['share_never_decided_by_2h']:.0%}")
row("Brier, our model at +15 / +30 min", lambda c: f"{d[c]['brier_ours_t15']:.3f} / {d[c]['brier_ours_t30']:.3f}")
for w in (0.0, 0.2, 0.4):
    row(f"Mean P&L, $100, naive {int(w*100)}%", lambda c, w=w: f"${get(c, w, 100, 'mean_pnl'):+.2f} ± {1.96*get(c, w, 100, 'se'):.2f}")
row("Win rate (nights traded), naive 20%", lambda c: f"{get(c, 0.2, 100, 'win_rate'):.0%}")
row("5th / 95th pct P&L, naive 20%", lambda c: f"${get(c, 0.2, 100, 'p5'):+.1f} / ${get(c, 0.2, 100, 'p95'):+.1f}")
row("Sharpe per night, naive 20%", lambda c: f"{get(c, 0.2, 100, 'sharpe_night'):.2f}")
row("Trades per night / fees, naive 20%", lambda c: f"{get(c, 0.2, 100, 'fills'):.1f} / ${get(c, 0.2, 100, 'fees'):.2f}")
row("Mean max drawdown, naive 20%", lambda c: f"{get(c, 0.2, 100, 'mdd_pct'):.1f}%")
row("Return at $10k, naive 20% / 40%", lambda c: f"{get(c, 0.2, 10000, 'mean_ret_pct'):+.2f}% / {get(c, 0.4, 10000, 'mean_ret_pct'):+.2f}%")
row("Minutes mispriced >5¢, naive 20%", lambda c: f"{get(c, 0.2, 100, 'minutes_mispriced_5c'):.0f}")
for w in ("0.0", "0.2", "0.4"):
    row(f"Naive-share estimate, true {int(float(w)*100)}% (mean ± sd)",
        lambda c, w=w: f"{100*R[c]['estimator'][w]['mean']:.0f}% ± {100*R[c]['estimator'][w]['sd']:.0f}")

# ---- liquidity swap: Georgia with Texas-level spreads and depth
try:
    G2 = json.load(open("results_GA_liqTX.json"))
    g2 = lambda w, br, k: [r for r in G2["results"] if r["w"] == w and r["bankroll"] == br][0][k]
    print("\nGeorgia with Texas-level liquidity:")
    for w in (0.0, 0.2, 0.4):
        print(f"  naive {int(w*100)}%: $100 mean ${g2(w,100,'mean_pnl'):+.2f} ± {1.96*g2(w,100,'se'):.2f};  $10k return {g2(w,10000,'mean_ret_pct'):+.2f}%")
except FileNotFoundError:
    pass
