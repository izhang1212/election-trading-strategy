"""Synthetic 'true worlds' for the Texas Senate and Georgia Governor races (toy, NOT real data).

Each state is a list of reporting units: the largest counties individually, smaller counties pooled into groups.
County sizes and leans are rough, real-shaped approximations for illustration only.

A world produces:
  * a synthetic HISTORY (2018-2024): county results split early vs Election Day, plus early-vote share per county
    -> this is what the 'data pull' stage would download;
  * a 2026 PRE-ELECTION feed: early-vote turnout per county (TX daily turnout / GA daily totals), known before polls close;
  * a 2026 ELECTION NIGHT: timestamped batches (early drop, then Election Day chunks with precinct-reporting %).
Vote shares are two-party Democratic shares; third-party votes are ignored (plurality race, no runoff).

Additions for the v2 re-run (defaults reproduce the original study exactly):
  * SIG["ef_common"]: sd of a statewide shift in every county's early share of the vote on election night (0 = off);
  * SIG["ed_year"]: year-to-year noise of the type-level ED gap in the synthetic history (was a literal 0.01);
  * World.calibrate_truth(): sets the true statewide level so the D win rate equals the market prior;
  * World.from_fitted(): builds the units from data/fitted/<state>_counties.csv (real leans, sizes, early
    shares and type-level ED gaps) instead of the hand-entered lists below.
"""
import numpy as np

N_TYPES = 4
# volatilities of the true world (two-party share points as fractions)
SIG = dict(s=0.03, f=0.025, g=0.02, e=0.02, b_early=0.003, b_ed=0.012, ed_year=0.01, ef_common=0.0)


def _tx_units():
    # types: 0 urban core, 1 suburban, 2 Hispanic / border, 3 rural & small metro
    big = [("Harris", 0, 1100, 0.55), ("Dallas", 0, 600, 0.64), ("Bexar", 0, 570, 0.58), ("Travis", 0, 470, 0.73),
           ("Tarrant", 1, 650, 0.47), ("Collin", 1, 400, 0.45), ("Denton", 1, 350, 0.42), ("Fort Bend", 1, 290, 0.54),
           ("Williamson", 1, 250, 0.50), ("Montgomery", 1, 250, 0.24), ("Brazoria", 1, 140, 0.38),
           ("Galveston", 1, 140, 0.35), ("Hays", 1, 100, 0.55),
           ("El Paso", 2, 200, 0.63), ("Hidalgo", 2, 160, 0.55), ("Cameron", 2, 85, 0.52), ("Webb", 2, 55, 0.55),
           ("Nueces", 2, 110, 0.44), ("Lubbock", 3, 100, 0.28), ("McLennan", 3, 85, 0.34)]
    r = np.random.default_rng(101)
    units = list(big)
    units += [(f"South Texas group {j+1}", 2, 40, r.uniform(0.40, 0.55)) for j in range(8)]
    units += [(f"Rural TX group {j+1}", 3, 70, r.uniform(0.15, 0.33)) for j in range(32)]
    return units


def _ga_units():
    # types: 0 urban core, 1 metro suburbs/exurbs, 2 Black Belt rural, 3 white rural
    big = [("Fulton", 0, 420, 0.72), ("DeKalb", 0, 300, 0.82), ("Clayton", 0, 95, 0.87), ("Chatham", 0, 110, 0.59),
           ("Richmond", 0, 65, 0.66), ("Muscogee", 0, 65, 0.61), ("Bibb", 0, 60, 0.61), ("Clarke", 0, 45, 0.70),
           ("Gwinnett", 1, 330, 0.59), ("Cobb", 1, 330, 0.56), ("Henry", 1, 110, 0.60), ("Douglas", 1, 60, 0.64),
           ("Rockdale", 1, 40, 0.73), ("Fayette", 1, 70, 0.42), ("Newton", 1, 45, 0.52), ("Cherokee", 1, 135, 0.28),
           ("Forsyth", 1, 125, 0.33), ("Paulding", 1, 75, 0.34), ("Columbia", 1, 70, 0.36), ("Hall", 1, 80, 0.26)]
    r = np.random.default_rng(202)
    units = list(big)
    units += [(f"Black Belt group {j+1}", 2, 40, r.uniform(0.50, 0.62)) for j in range(10)]
    units += [(f"Rural GA group {j+1}", 3, 55, r.uniform(0.18, 0.33)) for j in range(25)]
    return units


STATES = {
    "TX": dict(
        race="Texas Senate", units=_tx_units(), prior_p=0.62, poll_close_et="8:00 pm",
        early_frac=0.70, ed_off=np.array([-0.04, -0.04, -0.03, -0.02]),
        # minutes after first poll close: early-report window by type, ED start delay, ED chunk spacing
        early_win=[(0, 20), (2, 25), (5, 35), (3, 40)], ed_delay=[(50, 100), (40, 90), (40, 90), (25, 70)],
        ed_gap=(15, 40), late_units={"El Paso": 60}, slow_size=500,   # El Paso: Mountain time, an hour later
        spread_base=0.01, spread_post=0.02, depth=2000, kalshi_volume="$14M+"),
    "GA": dict(
        race="Georgia Governor", units=_ga_units(), prior_p=0.48, poll_close_et="7:00 pm",
        early_frac=0.65, ed_off=np.array([-0.05, -0.04, -0.03, -0.02]),
        early_win=[(20, 60), (5, 55), (5, 50), (5, 45)], ed_delay=[(45, 90), (35, 80), (30, 70), (25, 60)],
        ed_gap=(20, 50), late_units={}, slow_size=250,
        spread_base=0.015, spread_post=0.03, depth=400, kalshi_volume="$1.9M"),
}


class World:
    def __init__(self, code, seed=0):
        cfg = STATES[code]
        self.code, self.cfg = code, cfg
        u = cfg["units"]
        self.names = [x[0] for x in u]
        self.ctype = np.array([x[1] for x in u])
        self.V = np.array([x[2] for x in u], float)          # expected total votes (thousands)
        self.lean = np.array([x[3] for x in u], float)       # true historical lean (two-party D share)
        self.N = len(u)
        r = np.random.default_rng(1000 + seed)
        onehot = np.eye(N_TYPES)[self.ctype]
        self.L = onehot + r.normal(0, 0.25, (self.N, N_TYPES)) * (1 - onehot)   # true loadings (cross-loadings)
        self.ed_off = cfg["ed_off"]
        self.ef_mean = np.clip(cfg["early_frac"] + r.normal(0, 0.05, self.N), 0.45, 0.85)  # county early share
        self.shift = 0.0   # set by calibrate_truth()

    # ---------------- v2: real county structure ----------------
    @classmethod
    def from_fitted(cls, code, fitted_dir, seed=0, n_big=20, n_units=60, prior_p=None):
        """Units from fit_counties.py output: the n_big largest counties individually, the rest pooled within type
        into groups of similar size (n_units in total). Leans are the fitted relative early leans (the level is set
        by calibrate_truth), V = expected 2026 two-party votes in thousands, ef_mean = fitted early share,
        ed_off = fitted type-level ED gap."""
        import pandas as pd
        c = pd.read_csv(f"{fitted_dir}/{code.lower()}_counties.csv")
        t = pd.read_csv(f"{fitted_dir}/{code.lower()}_types.csv").sort_values("ctype")
        size = (c.early_known / c.ef) if c.early_known.notna().all() else c.twoparty_2022   # GA: 2026 early file pending
        c = c.assign(V=size / 1000).sort_values("V", ascending=False)
        big, rest = c.iloc[:n_big], c.iloc[n_big:]
        units = [(r.county, int(r.ctype), r.V, r.lean, r.ef) for r in big.itertuples()]
        n_left = n_units - n_big
        tot = rest.groupby("ctype").V.sum()
        for k, g in rest.groupby("ctype"):
            m = max(1, int(round(n_left * tot[k] / tot.sum())))
            g = g.sort_values("lean")                      # pool counties of similar lean together
            for j, part in enumerate(np.array_split(g, m)):
                w = part.V.values
                units.append((f"{code} type{k} group {j+1}", int(k), w.sum(), np.average(part.lean, weights=w),
                              np.average(part.ef, weights=w)))
        cfg = dict(STATES[code], units=[u[:4] for u in units], ed_off=t.ed_off.values,
                   prior_p=prior_p or STATES[code]["prior_p"],
                   slow_size=STATES[code]["slow_size"] * 1.0)
        self = cls.__new__(cls)
        self.code, self.cfg = code, cfg
        self.names = [u[0] for u in units]; self.ctype = np.array([u[1] for u in units])
        self.V = np.array([u[2] for u in units], float); self.lean = np.array([u[3] for u in units], float)
        self.N = len(units)
        r = np.random.default_rng(1000 + seed)
        onehot = np.eye(N_TYPES)[self.ctype]
        self.L = onehot + r.normal(0, 0.25, (self.N, N_TYPES)) * (1 - onehot)
        self.ed_off = cfg["ed_off"]
        self.ef_mean = np.clip(np.array([u[4] for u in units]), 0.45, 0.92)
        self.shift = 0.0
        return self

    def calibrate_truth(self, rng, n=2000):
        """Set the true statewide level so P(D wins) = prior_p. The final share is linear in the shift
        (both vote types move one-for-one and turnout does not depend on it), so one batch of nights suffices."""
        self.shift = 0.0
        fs = np.array([self.night(rng)["final_share"] for _ in range(n)])
        self.shift = 0.5 - np.quantile(fs, 1 - self.cfg["prior_p"])
        return self.shift

    # ---------------- synthetic history (what the data-pull stage downloads) ----------------
    def history(self, rng, years=(2018, 2020, 2022, 2024)):
        rows = []
        for y in years:
            c = rng.normal(0, 0.04)                                   # year's statewide environment
            f = rng.normal(0, SIG["f"], N_TYPES)
            e = rng.normal(0, SIG["e"], self.N)
            off_y = self.ed_off + rng.normal(0, SIG["ed_year"], N_TYPES)   # ED gap varies by year
            g = rng.normal(0, SIG["g"])
            early = self.lean + c + self.L @ f + e
            ed = early + off_y[self.ctype] + g + rng.normal(0, 0.008, self.N)
            ef = np.clip(self.ef_mean + rng.normal(0, 0.04, self.N), 0.4, 0.9)
            tot = self.V * (1 + rng.normal(0, 0.08, self.N))
            rows.append(dict(year=y, early_share=early, ed_share=ed, early_votes=tot * ef, ed_votes=tot * (1 - ef)))
        return rows

    # ---------------- 2026 election night ----------------
    def night(self, rng):
        cfg = self.cfg
        s = rng.normal(0, SIG["s"]); f = rng.normal(0, SIG["f"], N_TYPES); g = rng.normal(0, SIG["g"])
        e = rng.normal(0, SIG["e"], self.N)
        swing = s + self.L @ f + e
        early_share = self.lean + self.shift + swing
        ed_share = early_share + self.ed_off[self.ctype] + g
        tmult = 1 + rng.normal(0, 0.05) + rng.normal(0, 0.05, self.N)
        ef_shift = rng.normal(0, SIG["ef_common"]) if SIG["ef_common"] else 0.0   # v2: statewide early-share shift
        ef = np.clip(self.ef_mean + ef_shift + rng.normal(0, 0.04, self.N), 0.4, 0.9)
        early_votes = self.V * tmult * ef
        ed_votes = self.V * tmult * (1 - ef)
        # pre-election feed: early-vote turnout per county is published before election night (small error)
        early_known = early_votes * (1 + rng.normal(0, 0.01, self.N))
        batches = []   # (minute, unit, is_ed, votes, dshare, cumulative precinct fraction)
        n_chunks = np.zeros(self.N, int)
        for i in range(self.N):
            off = cfg["late_units"].get(self.names[i], 0)
            lo, hi = cfg["early_win"][self.ctype[i]]
            te = off + rng.uniform(lo, hi)
            batches.append((te, i, 0, early_votes[i], early_share[i] + rng.normal(0, SIG["b_early"]), 1.0))
            k = int(np.clip(round(self.V[i] / 120), 1, 9)) if self.ctype[i] != 3 or self.V[i] > 90 else rng.integers(1, 3)
            n_chunks[i] = k
            w = rng.dirichlet(np.full(k, 4.0))
            lo, hi = cfg["ed_delay"][self.ctype[i]]
            t = te + rng.uniform(lo, hi) * (1.6 if self.V[i] > cfg["slow_size"] else 1.0)    # big counties are slower
            cum = 0.0
            for j in range(k):
                cum += w[j]
                pfrac = 1.0 if j == k - 1 else float(np.clip(cum + rng.normal(0, 0.03), 0.02, 0.99))
                batches.append((t, i, 1, ed_votes[i] * w[j], ed_share[i] + rng.normal(0, SIG["b_ed"]), pfrac))
                t += rng.uniform(*cfg["ed_gap"])
        batches.sort(key=lambda b: b[0])
        totD = (early_votes * early_share + ed_votes * ed_share).sum(); tot = (early_votes + ed_votes).sum()
        return dict(batches=batches, early_known=early_known, outcome=int(totD > tot / 2),
                    final_share=totD / tot, n_batches=len(batches))
