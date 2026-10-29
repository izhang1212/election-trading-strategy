"""Plot one simulated night: fair value vs our model vs market, with the taker+filter strategy's trades."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase1_strategy import run_taker

d = np.load("phase1_sample_nights.npz")
P_S, P_O, mid, RAW, Y = d["P_S"], d["P_O"], d["mid"], d["RAW"], d["Y"]
bid, ask = d["bid"], d["ask"]
n, T1 = mid.shape

# pick a "red mirage" night: D wins, early raw count is R-leaning, and the strategy trades
best = None
for i in range(n):
    if Y[i] == 1 and np.nanmin(RAW[i, :30]) < 0.44 and P_S[i, 25] < 0.9:
        r = run_taker(P_O, mid, bid, ask, Y, trace_idx=i, move_filter=0.01)
        if 2 <= len(r["trace"]["trades"]) <= 6:
            best = (i, r); break
i, r = best
T = 100
t = np.arange(T)
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e1"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
for ax in (ax1, ax2):
    ax.set_facecolor("#fcfcfb"); ax.grid(color=grid, lw=0.8); ax.tick_params(colors=ink2)
    for s in ax.spines.values(): s.set_visible(False)
fig.patch.set_facecolor("#fcfcfb")
ax1.plot(t, P_S[i, :T] * 100, color="#1baf7a", lw=2, label="Informed fair value")
ax1.plot(t, P_O[i, :T] * 100, color="#2a78d6", lw=2, label="Our model (1-min lag)")
ax1.plot(t, mid[i, :T] * 100, color="#eb6834", lw=2, label="Market mid (20% naive)")
for (tt, what, C, px) in r["trace"]["trades"]:
    if tt < T:
        buy = what.startswith("buy")
        ax1.scatter(tt, (px if "YES" in what else 1 - px) * 100, s=70, zorder=5,
                    marker="^" if buy else "v", color=ink, edgecolor="#fcfcfb", lw=1.5)
        ax1.annotate(f"{what} ×{int(C)} @ {px*100:.0f}¢", (tt, (px if 'YES' in what else 1 - px) * 100),
                     xytext=(6, 8 if buy else -14), textcoords="offset points", fontsize=8.5, color=ink)
ax1.set_ylabel("Talarico win probability / price (¢)", color=ink2)
ax1.set_ylim(0, 100)
ax1.legend(frameon=False, loc="lower right", fontsize=9)
ax1.set_title(f"Sample simulated night #{i}: raw count leans R early, Talarico wins  |  night P&L ${r['pnl'][i]:+.2f}",
              color=ink, fontsize=11, loc="left")
ax2.plot(t, RAW[i, :T] * 100, color="#2a78d6", lw=2)
ax2.axhline(50, color=ink2, lw=1, ls="--")
ax2.set_ylabel("Raw Talarico share\nof counted vote (%)", color=ink2)
ax2.set_xlabel("Minutes after polls close", color=ink2)
fig.tight_layout()
fig.savefig("figures/phase1_sample_night.png", dpi=150)
print(i, r["pnl"][i], r["trace"]["trades"])
