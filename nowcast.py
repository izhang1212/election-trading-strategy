"""Kalman fair value from parsed batches, adapted from simulation/model.py:Nowcast for real county inputs.

State x = [s, f_1..f_4, g, e_1..e_N] (statewide swing, four county-type swings, Election-Day-vs-early gap,
county effects). Each batch's two-party D share is compared with the county's expectation for that vote
type; the surprise updates x; the outstanding vote is projected; P(D wins) = Phi(margin / sd).

Differences from simulation/model.py, all forced by real feeds:
  * inputs come from data/fitted/<state>_counties.csv and <state>_types.csv instead of a World object;
  * a county's early vote may arrive in several chunks (e.g. mail after in-person). The early block is
    treated as complete once counted early votes reach early_complete_frac of early_known; until then the
    remainder (early_known - counted) is projected at the updated early expectation;
  * when the feed gives no precinct-reporting fraction for an ED batch, it is estimated as
    counted ED votes / expected ED votes;
  * (= simulation/model.py with ed_common/ed_learn) outstanding Election Day volume has a common statewide
    component (relative sd ed_common), and the statewide ED-volume ratio is learned from counties already
    reporting ED votes. Real early shares move statewide between cycles (TX 87/68/81%, GA 54/81/64/77%).
With one early batch per county and a reported precinct fraction, the arithmetic is identical to model.py.
"""
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from parse_results import county_key

CFG = yaml.safe_load(open("config.yaml"))
SIG = CFG["nowcast"]["sig"]
ED_COMMON = CFG["nowcast"].get("ed_common", 0.0)     # see simulation/surge_study.py
ED_LEARN = CFG["nowcast"].get("ed_learn", False)
N_TYPES = 4


class Nowcast:
    def __init__(self, state, fitted_dir=None, early_complete_frac=0.95):
        d = fitted_dir or CFG["paths"]["fitted"]
        c = pd.read_csv(f"{d}/{state.lower()}_counties.csv")
        t = pd.read_csv(f"{d}/{state.lower()}_types.csv").sort_values("ctype")
        if c.early_known.isna().any():
            raise ValueError(f"{state}: early_known missing for {c.early_known.isna().sum()} counties; "
                             "load the 2026 early-vote file and re-run fit_counties.py")
        self.names = list(c.county); self.idx = {county_key(n): i for i, n in enumerate(self.names)}
        self.N = len(c); self.ct = c.ctype.values.astype(int)
        self.lean = c.lean.values; self.ef = c.ef.values; self.ed_off = t.ed_off.values
        self.early_known = c.early_known.values.astype(float)
        self.L = np.eye(N_TYPES)[self.ct]                       # pure type loadings, as in fit_from_history
        self.ecf = early_complete_frac
        n = 6 + self.N
        self.He = np.zeros((self.N, n)); self.He[:, 0] = 1; self.He[:, 1:5] = self.L
        self.He[np.arange(self.N), 6 + np.arange(self.N)] = 1
        self.Hd = self.He.copy(); self.Hd[:, 5] = 1
        self.P0 = np.diag(np.r_[SIG["s"], np.full(4, SIG["f"]), SIG["g"], np.full(self.N, SIG["e"])] ** 2)
        self.shift = 0.0
        self.reset()

    # ------------------------------------------------------------ as model.py
    def reset(self):
        Ek = self.early_known
        self.mu = np.zeros(6 + self.N); self.P = self.P0.copy()
        self.Ek = Ek
        self.ed_exp = Ek * (1 - self.ef) / self.ef
        self.er = np.zeros(self.N, bool); self.ev = np.zeros(self.N)
        self.dv = np.zeros(self.N); self.pf = np.zeros(self.N)
        self.Drep = 0.0; self.Trep = 0.0

    def calibrate(self, prior_p):
        """Choose the statewide level so the pre-results win probability equals the market price."""
        lo, hi = -1.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2; self.shift = mid; self.reset()
            if self.prob() < prior_p:
                lo = mid
            else:
                hi = mid
        self.shift = (lo + hi) / 2; self.reset()

    def base(self):
        return self.lean + self.shift

    def update(self, i, is_ed, votes, share, pfrac=None):
        h = self.Hd[i] if is_ed else self.He[i]
        expct = self.base()[i] + (self.ed_off[self.ct[i]] if is_ed else 0.0)
        R = SIG["b_ed"] ** 2 if is_ed else SIG["b_early"] ** 2
        Ph = self.P @ h; S = h @ Ph + R; K = Ph / S
        self.mu = self.mu + K * (share - expct - h @ self.mu); self.P = self.P - np.outer(K, Ph)
        self.Drep += votes * share; self.Trep += votes
        if is_ed:
            self.dv[i] += votes
            self.pf[i] = pfrac if pfrac is not None else min(self.dv[i] / max(self.ed_exp[i], 1.0), 0.99)
        else:
            self.ev[i] += votes
            self.er[i] = self.ev[i] >= self.ecf * self.Ek[i]

    def prob(self):
        b = self.base()
        sh_e = b + self.He @ self.mu
        sh_d = b + self.ed_off[self.ct] + self.Hd @ self.mu
        v_e = np.where(self.er, 0.0, np.maximum(self.Ek - self.ev, 0.0))
        known_early = np.where(self.er, self.ev, np.maximum(self.Ek, self.ev))
        ed_prior = known_early * (1 - self.ef) / self.ef
        rep = self.pf > 0; k = 0.0
        if ED_LEARN and rep.any():   # statewide ED-volume ratio from counties reporting ED votes, shrunk toward 1
            rho = (self.dv[rep] / np.maximum(self.pf[rep], 1e-6)).sum() / max(ed_prior[rep].sum(), 1e-9)
            k = ed_prior[rep].sum() / (ed_prior[rep].sum() + 0.2 * ed_prior.sum())
            ed_prior = ed_prior * (1 + k * (rho - 1))
        ed_tot = np.where(rep, self.dv / np.maximum(self.pf, 1e-6), ed_prior)
        v_d = np.maximum(ed_tot - self.dv, 0.0) * (self.pf < 1)
        sd_rel = np.where(self.pf > 0, 0.08 * (1 - self.pf), 0.10)
        mD = self.Drep + v_e @ sh_e + v_d @ sh_d
        mT = self.Trep + v_e.sum() + v_d.sum()
        wvec = v_e @ self.He + v_d @ self.Hd
        var = wvec @ self.P @ wvec + ((sd_rel * v_d * (sh_d - 0.5)) ** 2).sum() + ((0.01 * v_e * (sh_e - 0.5)) ** 2).sum() \
            + 0.5 * ((v_d * SIG["b_ed"]) ** 2).sum()
        if ED_COMMON:   # one statewide early-share shift moves every county's outstanding ED volume together
            var += (ED_COMMON * (1 - k) * (v_d * (sh_d - 0.5) * (self.pf == 0)).sum()) ** 2
        margin = mD - mT / 2
        if var < 1e-12:
            return 1.0 if margin > 0 else 0.0
        return float(np.clip(norm.cdf(margin / np.sqrt(var)), 1e-4, 1 - 1e-4))

    def expected_total(self):
        known_early = np.where(self.er, self.ev, np.maximum(self.Ek, self.ev))
        ed_tot = np.where(self.pf > 0, self.dv / np.maximum(self.pf, 1e-6), known_early * (1 - self.ef) / self.ef)
        return known_early.sum() + ed_tot.sum()

    # ------------------------------------------------------------ live helpers
    def apply_batch(self, b):
        """b: parse_results batch. Returns False if the county is unknown (logged by the caller)."""
        i = self.idx.get(county_key(b["county"]))
        if i is None or b["method"] not in ("early", "ed"):
            return False
        self.update(i, b["method"] == "ed", b["votes"], b["share"], b.get("reporting"))
        return True

    def raw_share(self):
        return self.Drep / self.Trep if self.Trep > 0 else np.nan

    def frac_reported(self):
        return min(self.Trep / self.expected_total(), 1.0)

    def outstanding(self):
        return bool((~self.er).any() or (self.pf < 1).any())
