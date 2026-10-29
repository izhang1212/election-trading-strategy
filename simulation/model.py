"""Stage 2-3: fit county expectations from (synthetic) history, then nowcast election night.

fit_from_history  -> our county leans, Election Day offsets and early-vote shares, estimated from 4 past elections
Nowcast           -> Kalman filter over latent swings [statewide s, 4 type factors f, ED gap g, county effects e_i];
                     each batch's surprise vs. expectation updates the state; outstanding vote is projected to a
                     win probability (Gaussian on the final two-party margin).
"""
import numpy as np
from scipy.stats import norm
from world import SIG, N_TYPES


def fit_from_history(world, hist):
    """What our Phase 3 model would estimate from downloaded county history."""
    E = np.array([h["early_share"] for h in hist]); D = np.array([h["ed_share"] for h in hist])
    Ve = np.array([h["early_votes"] for h in hist]); Vd = np.array([h["ed_votes"] for h in hist])
    state_early = (E * Ve).sum(1) / Ve.sum(1)
    lean_hat = (E - state_early[:, None]).mean(0)                    # relative lean; level set later by the prior
    gap = D - E
    ed_off_hat = np.array([np.average(gap[:, world.ctype == k], weights=Vd[:, world.ctype == k]) for k in range(N_TYPES)])
    ef_hat = (Ve / (Ve + Vd)).mean(0)
    L_hat = np.eye(N_TYPES)[world.ctype]                              # we assume pure type loadings
    return dict(lean=lean_hat, ed_off=ed_off_hat, ef=ef_hat, L=L_hat)


def smart_params(world, rng):
    """The informed traders: better county data, true structure."""
    return dict(lean=world.lean + rng.normal(0, 0.006, world.N), ed_off=world.ed_off.copy(),
                ef=world.ef_mean.copy(), L=world.L.copy())


class Nowcast:
    def __init__(self, world, params, prior_p):
        self.w = world; self.N = world.N; self.ct = world.ctype
        self.L, self.ed_off, self.ef = params["L"], params["ed_off"], params["ef"]
        self.lean = params["lean"]
        n = 6 + self.N
        self.He = np.zeros((self.N, n)); self.He[:, 0] = 1; self.He[:, 1:5] = self.L; self.He[np.arange(self.N), 6 + np.arange(self.N)] = 1
        self.Hd = self.He.copy(); self.Hd[:, 5] = 1
        self.P0 = np.diag(np.r_[SIG["s"], np.full(4, SIG["f"]), SIG["g"], np.full(self.N, SIG["e"])] ** 2)
        self.prior_p = prior_p
        self.shift = 0.0
        # v2 options (absent from params -> original behaviour): common relative uncertainty of every county's
        # outstanding Election Day volume (a statewide early-share shift), and learning that shift from counties
        # whose Election Day vote has started reporting
        self.ed_common = params.get("ed_common", 0.0); self.ed_learn = params.get("ed_learn", False)

    def reset(self, early_known):
        self.mu = np.zeros(6 + self.N); self.P = self.P0.copy()
        self.Ek = early_known
        self.ed_exp = early_known * (1 - self.ef) / self.ef
        self.er = np.zeros(self.N, bool); self.ev = np.zeros(self.N)
        self.dv = np.zeros(self.N); self.pf = np.zeros(self.N)
        self.Drep = 0.0; self.Trep = 0.0

    def calibrate(self, early_known):
        """Choose the statewide level so the pre-results win probability equals the market prior."""
        lo, hi = -1.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2; self.shift = mid; self.reset(early_known)
            if self.prob() < self.prior_p: lo = mid
            else: hi = mid
        self.shift = (lo + hi) / 2

    def base(self):
        return self.lean + self.shift

    def update(self, i, is_ed, votes, share, pfrac):
        h = self.Hd[i] if is_ed else self.He[i]
        expct = self.base()[i] + (self.ed_off[self.ct[i]] if is_ed else 0.0)
        R = SIG["b_ed"] ** 2 if is_ed else SIG["b_early"] ** 2
        Ph = self.P @ h; S = h @ Ph + R; K = Ph / S
        self.mu = self.mu + K * (share - expct - h @ self.mu); self.P = self.P - np.outer(K, Ph)
        self.Drep += votes * share; self.Trep += votes
        if is_ed:
            self.dv[i] += votes; self.pf[i] = pfrac
        else:
            self.er[i] = True; self.ev[i] = votes

    def prob(self):
        b = self.base()
        sh_e = b + self.He @ self.mu
        sh_d = b + self.ed_off[self.ct] + self.Hd @ self.mu
        v_e = np.where(self.er, 0.0, self.Ek)
        known_early = np.where(self.er, self.ev, self.Ek)
        ed_prior = known_early * (1 - self.ef) / self.ef
        rep = self.pf > 0; k = 0.0
        if self.ed_learn and rep.any():   # statewide ED-volume ratio from reporting counties, shrunk toward 1
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
        if self.ed_common:
            var += (self.ed_common * (1 - k) * (v_d * (sh_d - 0.5) * (self.pf == 0)).sum()) ** 2
        margin = mD - mT / 2
        if var < 1e-12:
            return 1.0 if margin > 0 else 0.0
        return float(np.clip(norm.cdf(margin / np.sqrt(var)), 1e-4, 1 - 1e-4))

    def expected_total(self):
        known_early = np.where(self.er, self.ev, self.Ek)
        ed_tot = np.where(self.pf > 0, self.dv / np.maximum(self.pf, 1e-6), known_early * (1 - self.ef) / self.ef)
        return known_early.sum() + ed_tot.sum()


def run_night(world, night, smart, ours, T):
    """Minute-by-minute: smart fair value, our fair value (info through t, and acted on at t+1),
    raw running D share, fraction of expected vote reported, batch flags."""
    smart.reset(night["early_known"]); ours.reset(night["early_known"])
    p_s = np.empty(T + 1); p_i = np.empty(T + 1)
    raw = np.full(T + 1, np.nan); frac = np.zeros(T + 1); flag = np.zeros(T + 1, bool)
    cs, co = smart.prob(), ours.prob(); prior_s, prior_o = cs, co
    b = night["batches"]; k = 0
    exp_total = ours.expected_total()
    for t in range(T + 1):
        got = False
        while k < len(b) and b[k][0] <= t:
            _, i, is_ed, v, sh, pf = b[k]
            smart.update(i, is_ed, v, sh, pf); ours.update(i, is_ed, v, sh, pf)
            got = True; k += 1
        if got:
            cs, co = smart.prob(), ours.prob(); exp_total = ours.expected_total()
        flag[t] = got; p_s[t] = cs; p_i[t] = co
        if ours.Trep > 0:
            raw[t] = ours.Drep / ours.Trep
        frac[t] = min(ours.Trep / exp_total, 1.0)
    p_o = np.r_[prior_o, p_i[:-1]]          # 1-minute data latency for trading
    return dict(P_S=p_s, P_O=p_o, P_I=p_i, RAW=raw, FRAC=frac, FLAG=flag, PRIOR=prior_s)
