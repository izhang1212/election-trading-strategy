"""
Synthetic election-night simulator + strategy backtest (toy, NOT real data).

World: Texas-shaped counties (sizes/leans roughly 2024-shaped, scaled to midterm turnout),
latent statewide / type / early-vs-Election-Day swings, batch-by-batch reporting.
Market: blend of a correctly specified "smart" nowcast and "naive" traders who extrapolate
the raw on-screen count with a lag. Our model is deliberately MISSPECIFIED relative to the smart
traders (wrong loadings, optionally wrong variances / ED offsets), so any profit has to come from
the naive-trader component. With naive weight w = 0 the correct result is a loss (null test).

Strategies (both start with $100, one market = Talarico YES/NO):
  taker: buy when p - ask - fee - buffer > 0, fractional Kelly size; exit when bid - fee > p or bid >= 0.97
  maker: rest orders at <= p - margin (improving the touch by 1c); fill only if price trades through us
"""
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------- world
TYPES = ["urban", "suburban", "border", "rural"]
# name, type, expected total votes (thousands), expected D share of EARLY vote (before uniform shift)
COUNTIES = [
    ("Harris", 0, 1160, 0.56), ("Dallas", 0, 640, 0.64), ("Travis", 0, 490, 0.73), ("Bexar", 0, 600, 0.57),
    ("Tarrant", 1, 675, 0.49), ("Collin", 1, 435, 0.46), ("Denton", 1, 360, 0.43), ("Fort Bend", 1, 300, 0.55),
    ("Williamson", 1, 255, 0.51), ("Montgomery", 1, 250, 0.28), ("Exurban ring", 1, 225, 0.27),
    ("Hidalgo", 2, 190, 0.52), ("El Paso", 2, 225, 0.62), ("Cameron", 2, 100, 0.49), ("Webb", 2, 70, 0.54),
    ("Rural South", 2, 190, 0.46),
    ("Rural East", 3, 675, 0.24), ("Rural Panhandle/West", 3, 525, 0.19), ("Rural Central", 3, 675, 0.27),
    ("Mid-size metros", 3, 525, 0.36),
]
N = len(COUNTIES)
CTYPE = np.array([c[1] for c in COUNTIES])
V = np.array([c[2] for c in COUNTIES], float)
D_EARLY_RAW = np.array([c[3] for c in COUNTIES])
EARLY_FRAC = 0.65
ED_OFF_TRUE = np.array([-0.04, -0.04, -0.03, -0.02])  # ED is redder than early vote, by type

# truth volatilities (vote-share points as fractions)
SIG = dict(s=0.03, f=0.025, g=0.02, e=0.02, b_early=0.003, b_ed=0.01)
SIG_TURN_S, SIG_TURN_I, SIG_EF = 0.05, 0.05, 0.04


def calibrate_shift(target=0.505):
    ed = D_EARLY_RAW + ED_OFF_TRUE[CTYPE]
    base = (V * (EARLY_FRAC * D_EARLY_RAW + (1 - EARLY_FRAC) * ed)).sum() / V.sum()
    return target - base


SHIFT = calibrate_shift()
D_EARLY = D_EARLY_RAW + SHIFT

# fixed "true" loadings: own type = 1 plus fixed cross-loadings the world has but our model ignores
_world = np.random.default_rng(12345)
ONEHOT = np.eye(4)[CTYPE]
L_TRUE = ONEHOT + _world.normal(0, 0.25, (N, 4)) * (1 - ONEHOT)


def gen_night(rng):
    s = rng.normal(0, SIG["s"]); f = rng.normal(0, SIG["f"], 4); g = rng.normal(0, SIG["g"])
    e = rng.normal(0, SIG["e"], N)
    tmult = 1 + rng.normal(0, SIG_TURN_S) + rng.normal(0, SIG_TURN_I, N)
    ef = np.clip(EARLY_FRAC + rng.normal(0, SIG_EF, N), 0.5, 0.8)
    swing = s + L_TRUE @ f + e
    early_share = D_EARLY + swing
    ed_share = D_EARLY + ED_OFF_TRUE[CTYPE] + swing + g
    batches = []  # (time, county, is_ed, votes, dshare, chunk_idx, n_chunks)
    n_chunks = np.zeros(N, int)
    for i in range(N):
        t_lo, t_hi = [(15, 50), (5, 30), (10, 40), (1, 20)][CTYPE[i]]
        if COUNTIES[i][0] == "El Paso":
            t_lo, t_hi = 60, 75  # Mountain time
        te = rng.uniform(t_lo, t_hi)
        ve = V[i] * tmult[i] * ef[i]
        batches.append((te, i, 0, ve, early_share[i] + rng.normal(0, SIG["b_early"]), 0, 1))
        vd = V[i] * tmult[i] * (1 - ef[i])
        k = (2 if V[i] > 500 else 1) if CTYPE[i] == 3 else int(np.clip(round(V[i] / 150), 2, 7))
        n_chunks[i] = k
        w = rng.dirichlet(np.full(k, 4.0))
        t = te + rng.uniform(30, 70)
        for j in range(k):
            batches.append((t, i, 1, vd * w[j], ed_share[i] + rng.normal(0, SIG["b_ed"]), j, k))
            t += rng.uniform(10, 35)
    batches.sort(key=lambda b: b[0])
    totD = sum(b[3] * b[4] for b in batches); tot = sum(b[3] for b in batches)
    return dict(batches=batches, n_chunks=n_chunks, outcome=int(totD > tot / 2), final_share=totD / tot)


# ---------------------------------------------------------------- nowcast (Kalman over latent swings)
class Nowcast:
    """State x = [s, f1..f4, g, e_1..e_N]. Observes batch residual vs expectation; projects outstanding vote."""

    def __init__(self, L, ed_off, var_scale=1.0):
        self.L, self.ed_off = L, ed_off
        sd = np.sqrt(var_scale)
        self.sb_early, self.sb_ed = SIG["b_early"] * sd, SIG["b_ed"] * sd
        prior = np.r_[SIG["s"], np.full(4, SIG["f"]), SIG["g"], np.full(N, SIG["e"])] * sd
        self.mu = np.zeros(6 + N); self.P = np.diag(prior ** 2)

    def h(self, i, is_ed):
        v = np.zeros(6 + N); v[0] = 1; v[1:5] = self.L[i]; v[5] = is_ed; v[6 + i] = 1
        return v

    def base(self, i, is_ed):
        return D_EARLY[i] + (self.ed_off[CTYPE[i]] if is_ed else 0.0)

    def update(self, i, is_ed, share):
        h = self.h(i, is_ed); r = share - self.base(i, is_ed)
        R = (self.sb_ed if is_ed else self.sb_early) ** 2
        Ph = self.P @ h; S = h @ Ph + R; K = Ph / S
        self.mu = self.mu + K * (r - h @ self.mu); self.P = self.P - np.outer(K, Ph)

    def prob(self, st):
        """st: reporting state dict. Returns P(D wins) from Gaussian projection of final margin."""
        Drep, Trep = st["Drep"], st["Trep"]
        obs_t = [st["early_votes"][i] / (V[i] * EARLY_FRAC) for i in range(N) if st["early_votes"][i] > 0]
        turn = (sum(obs_t) + 3.0) / (len(obs_t) + 3.0)  # shrunk turnout ratio for unreported counties
        mD, mT = Drep, Trep; w = np.zeros(6 + N); var_extra = 0.0
        for i in range(N):
            blocks = []
            if st["early_votes"][i] == 0:
                ve = V[i] * turn * EARLY_FRAC
                blocks.append((0, ve, 0.07, 1))
                ed_tot = V[i] * turn * (1 - EARLY_FRAC); sd_t = 0.07
            else:
                ed_tot = st["early_votes"][i] * (1 - EARLY_FRAC) / EARLY_FRAC; sd_t = 0.06
            nrep, nch = st["ed_chunks"][i], st["n_chunks"][i]
            if nrep > 0:
                ed_tot = st["ed_votes"][i] * nch / nrep; sd_t = 0.25 / np.sqrt(nrep)
            rem = ed_tot * (nch - nrep) / nch
            if rem > 0:
                blocks.append((1, rem, sd_t, nch - nrep))
            for is_ed, v, sdt, nc in blocks:
                h = self.h(i, is_ed); sh = self.base(i, is_ed) + h @ self.mu
                mD += v * sh; mT += v; w += v * h
                var_extra += (sdt * v * (sh - 0.5)) ** 2
                if is_ed:
                    var_extra += nc * (v / nc) ** 2 * self.sb_ed ** 2
        margin = mD - mT / 2
        var = w @ self.P @ w + var_extra
        if var < 1e-12:
            return 1.0 if margin > 0 else 0.0
        return float(np.clip(norm.cdf(margin / np.sqrt(var)), 1e-4, 1 - 1e-4))


def run_nowcasts(night, T, ours_kw, latency=1):
    """Minute-by-minute arrays: smart p (info through t), our p (info through t-1), raw share, frac reported."""
    smart = Nowcast(L_TRUE, ED_OFF_TRUE)
    ours = Nowcast(**ours_kw)
    st = dict(Drep=0.0, Trep=0.0, early_votes=np.zeros(N), ed_votes=np.zeros(N),
              ed_chunks=np.zeros(N, int), n_chunks=night["n_chunks"])
    p_s, p_info = np.empty(T + 1), np.empty(T + 1)
    raw, frac, flag = np.full(T + 1, np.nan), np.zeros(T + 1), np.zeros(T + 1, bool)
    cur_s, cur_o = smart.prob(st), ours.prob(st)
    prior_s, prior_o = cur_s, cur_o
    b = night["batches"]; k = 0
    for t in range(T + 1):
        got = False
        while k < len(b) and b[k][0] <= t:
            _, i, is_ed, v, sh, _, _ = b[k]
            smart.update(i, is_ed, sh); ours.update(i, is_ed, sh)
            st["Drep"] += v * sh; st["Trep"] += v
            if is_ed:
                st["ed_votes"][i] += v; st["ed_chunks"][i] += 1
            else:
                st["early_votes"][i] = v
            got = True; k += 1
        if got:
            cur_s, cur_o = smart.prob(st), ours.prob(st)
        flag[t] = got
        p_s[t], p_info[t] = cur_s, cur_o
        if st["Trep"] > 0:
            raw[t] = st["Drep"] / st["Trep"]
        frac[t] = st["Trep"] / V.sum()
    # we act on data `latency` minutes late (scrape + parse + model)
    p_o = np.r_[np.full(latency, prior_o), p_info[:len(p_info) - latency]] if latency else p_info
    return p_s, p_o, raw, frac, flag, prior_s
