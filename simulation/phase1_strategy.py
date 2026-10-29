"""Market model (smart + naive traders) and the two trading strategies (taker, maker)."""
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------- market
def build_market(P_S, RAW, FRAC, FLAG, PRIOR, w, rng, lag=5, kappa=0.5, noise=0.004):
    """Mid = smoothed blend of smart fair value (instant) and naive raw-count extrapolation (lagged `lag` min)."""
    n, T1 = P_S.shape
    raw_l = np.full_like(RAW, np.nan); raw_l[:, lag:] = RAW[:, :-lag]
    frac_l = np.zeros_like(FRAC); frac_l[:, lag:] = FRAC[:, :-lag]
    sig_n = 0.06 * np.sqrt(np.clip(1 - frac_l, 0, 1)) + 0.003
    p_naive = np.where(np.isnan(raw_l), PRIOR[:, None],
                       norm.cdf((np.nan_to_num(raw_l, nan=0.5) - 0.5) / sig_n))
    # media call: once the informed fair value has sat beyond 99.5/0.5 for 15 minutes, the race is "called"
    # and naive traders follow the call instead of the raw count
    sure = (P_S > 0.995) | (P_S < 0.005)
    run = np.zeros_like(P_S); acc = np.zeros(n)
    for t in range(T1):
        acc = np.where(sure[:, t], acc + 1, 0); run[:, t] = acc
    called = np.maximum.accumulate(run >= 15, axis=1)
    p_naive = np.where(called, (P_S > 0.5).astype(float), p_naive)
    target = (1 - w) * P_S + w * p_naive
    mid = np.empty((n, T1)); mid[:, 0] = PRIOR
    for t in range(1, T1):
        mid[:, t] = np.clip(mid[:, t - 1] + kappa * (target[:, t] - mid[:, t - 1])
                            + rng.normal(0, noise, n), 0.015, 0.985)
    recent = FLAG | np.c_[np.zeros((n, 1), bool), FLAG[:, :-1]]
    hs = 0.01 + 0.02 * recent                      # spreads widen for 2 min after a batch
    bid = np.clip(np.floor((mid - hs) * 100 + 1e-9) / 100, 0.01, 0.98)
    ask = np.clip(np.ceil((mid + hs) * 100 - 1e-9) / 100, 0.02, 0.99)
    ask = np.maximum(ask, bid + 0.01)
    return mid, bid, ask, p_naive


# ---------------------------------------------------------------- strategy params
TF, MF = 0.07, 0.0175      # Kalshi-style taker fee rate; maker fee rate is an ASSUMPTION
KELLY, CAP, DEPTH = 0.25, 0.5, 500   # quarter Kelly, max 50% of equity in the market, 500 contracts at touch
BUF_T, BUF_M = 0.03, 0.02  # safety buffer beyond fees (taker / maker)
START = 100.0


def fee(C, P, rate):
    return np.where(C > 0, np.ceil(rate * C * P * (1 - P) * 100 - 1e-9) / 100, 0.0)


def _equity(cash, pos, m):
    return cash + np.where(pos > 0, pos * m, 0) + np.where(pos < 0, -pos * (1 - m), 0)


def _result(cash, pos, Y, fees, fills, mdd, tim, edge_sum, edge_n, trace):
    held = pos != 0
    cash = cash + np.where(pos > 0, pos * Y, 0) + np.where(pos < 0, -pos * (1 - Y), 0)
    return dict(pnl=cash - START, fees=fees, fills=fills, mdd=mdd, tim=tim, held_to_settle=held,
                entry_edge=np.where(edge_n > 0, edge_sum / np.maximum(edge_n, 1), np.nan), trace=trace)


def run_taker(P_O, mid, bid, ask, Y, trace_idx=0, move_filter=None):
    """move_filter: skip entries when the market moved more than this in the latest minute -- that move may be
    informed by a batch we haven't processed yet (we see data 1 min late). Older moves are 'explained': by then
    we know whether a batch arrived, so a gap that persists after the move settles is safe(r) to fade."""
    n, T1 = P_O.shape
    cash = np.full(n, START); pos = np.zeros(n); fees = np.zeros(n); fills = np.zeros(n)
    peak = np.full(n, START); mdd = np.zeros(n); tim = np.zeros(n)
    edge_sum = np.zeros(n); edge_n = np.zeros(n)
    trace = dict(pos=np.zeros(T1), eq=np.full(T1, START), trades=[])
    for t in range(1, T1):
        p, b, a, m = P_O[:, t], bid[:, t], ask[:, t], mid[:, t]
        blocked = np.zeros(n, bool)
        if move_filter is not None:
            blocked = np.abs(m - mid[:, t - 1]) > move_filter
        # exits: sell when the bid (net of fee) beats our fair value (unless the market just jumped on data we
        # may not have seen yet), or when price >= 97c (settlement lockup)
        ex = (pos > 0) & (((b - TF * b * (1 - b) > p) & ~blocked) | (b >= 0.97))
        C = np.where(ex, pos, 0); fz = fee(C, b, TF)
        if ex[trace_idx]: trace["trades"].append((t, "sell YES", C[trace_idx], b[trace_idx]))
        cash += C * b - fz; fees += fz; fills += ex; pos = np.where(ex, 0, pos)
        c = 1 - a
        ex = (pos < 0) & (((c - TF * c * (1 - c) > 1 - p) & ~blocked) | (c >= 0.97))
        C = np.where(ex, -pos, 0); fz = fee(C, c, TF)
        if ex[trace_idx]: trace["trades"].append((t, "sell NO", C[trace_idx], c[trace_idx]))
        cash += C * c - fz; fees += fz; fills += ex; pos = np.where(ex, 0, pos)
        eq = _equity(cash, pos, m)
        # entries: buy YES (side +1) at ask, or NO (side -1) at 1-bid
        for side in (1, -1):
            px = a if side == 1 else 1 - b
            q = p if side == 1 else 1 - p
            edge = q - px - TF * px * (1 - px) - BUF_T
            f = (q - px) / (1 - px)
            tgt = np.minimum(np.floor(np.maximum(np.minimum(KELLY * f * eq, CAP * eq), 0) / px), DEPTH)
            have = np.where(side * pos > 0, np.abs(pos), 0)
            ok = (edge > 0) & (side * pos >= 0) & ~blocked
            C = np.where(ok, np.maximum(tgt - have, 0), 0)
            C = np.minimum(C, np.floor(np.maximum(cash - 0.05, 0) / (px + TF * px * (1 - px))))
            fz = fee(C, px, TF)
            if C[trace_idx] > 0: trace["trades"].append((t, "buy YES" if side == 1 else "buy NO", C[trace_idx], px[trace_idx]))
            cash -= C * px + fz; fees += fz; pos += side * C; fills += C > 0
            edge_sum += np.where(C > 0, q - px, 0); edge_n += C > 0
        eq = _equity(cash, pos, m)
        peak = np.maximum(peak, eq); mdd = np.maximum(mdd, (peak - eq) / peak); tim += pos != 0
        trace["pos"][t] = pos[trace_idx]; trace["eq"][t] = eq[trace_idx]
    return _result(cash, pos, Y, fees, fills, mdd, tim, edge_sum, edge_n, trace)


def run_maker(P_O, mid, bid, ask, Y, trace_idx=0, depth=None):
    """depth=None: join/improve the touch (capped at p - buffer).
    depth=d: rest only at p - d (never improving the touch) to catch large naive overshoots."""
    n, T1 = P_O.shape
    cash = np.full(n, START); pos = np.zeros(n); fees = np.zeros(n); fills = np.zeros(n)
    peak = np.full(n, START); mdd = np.zeros(n); tim = np.zeros(n)
    edge_sum = np.zeros(n); edge_n = np.zeros(n)
    side = np.zeros(n, int); opx = np.zeros(n); osz = np.zeros(n); oq = np.zeros(n)
    trace = dict(pos=np.zeros(T1), eq=np.full(T1, START), trades=[])
    names = {1: "buy YES", 2: "buy NO", 3: "sell YES", 4: "sell NO"}
    for t in range(1, T1):
        p, b, a, m = P_O[:, t], bid[:, t], ask[:, t], mid[:, t]
        # fills: last minute's resting order fills only if the market traded THROUGH our price
        for sd, cond, sgn, is_buy in ((1, m < opx, +1, True), (2, (1 - m) < opx, -1, True),
                                      (3, m > opx, -1, False), (4, (1 - m) > opx, +1, False)):
            hit = (side == sd) & cond
            C = np.where(hit, osz, 0); fz = fee(C, opx, MF); fees += fz; fills += hit
            if hit[trace_idx]: trace["trades"].append((t, names[sd], C[trace_idx], opx[trace_idx]))
            cash += (-(C * opx) - fz) if is_buy else (C * opx - fz); pos += sgn * C
            if is_buy:
                edge_sum += np.where(hit, oq - opx, 0); edge_n += hit
        side[:] = 0
        # settlement-lockup exit as taker
        ex = (pos > 0) & (b >= 0.97); C = np.where(ex, pos, 0); fz = fee(C, b, TF)
        if ex[trace_idx]: trace["trades"].append((t, "sell YES", C[trace_idx], b[trace_idx]))
        cash += C * b - fz; fees += fz; fills += ex; pos = np.where(ex, 0, pos)
        c = 1 - a
        ex = (pos < 0) & (c >= 0.97); C = np.where(ex, -pos, 0); fz = fee(C, c, TF)
        if ex[trace_idx]: trace["trades"].append((t, "sell NO", C[trace_idx], c[trace_idx]))
        cash += C * c - fz; fees += fz; fills += ex; pos = np.where(ex, 0, pos)
        eq = _equity(cash, pos, m)
        # cancel/replace every minute: entries only when flat
        flat = pos == 0
        mfee = MF * p * (1 - p)
        if depth is None:
            yb = np.minimum(np.minimum(b + 0.01, np.floor((p - BUF_M - mfee) * 100 + 1e-9) / 100), a - 0.01)
            nb = np.minimum(np.minimum((1 - a) + 0.01, np.floor(((1 - p) - BUF_M - mfee) * 100 + 1e-9) / 100), (1 - b) - 0.01)
            post_y = flat & (p - m > BUF_M) & (yb >= 0.01)
            post_n = flat & (m - p > BUF_M) & (nb >= 0.01) & ~post_y
        else:
            yb = np.minimum(np.floor((p - depth - mfee) * 100 + 1e-9) / 100, b)
            nb = np.minimum(np.floor(((1 - p) - depth - mfee) * 100 + 1e-9) / 100, 1 - a)
            post_y = flat & (yb >= 0.01) & (m <= p)
            post_n = flat & (nb >= 0.01) & (m > p)
        for mask, px, q, sd in ((post_y, yb, p, 1), (post_n, nb, 1 - p, 2)):
            pxs = np.clip(px, 0.01, 0.99)
            f = np.clip((q - pxs) / (1 - pxs), 0, 1)
            C = np.minimum(np.floor(np.minimum(KELLY * f * eq, CAP * eq) / pxs), DEPTH)
            C = np.minimum(C, np.floor(np.maximum(cash - 0.05, 0) / (pxs * 1.01)))
            ok = mask & (C > 0)
            side = np.where(ok, sd, side); opx = np.where(ok, pxs, opx)
            osz = np.where(ok, C, osz); oq = np.where(ok, q, oq)
        # exits: rest a sell at >= fair value + fee (improving the touch when possible)
        ya = np.maximum(np.maximum(a - 0.01, b + 0.01), np.ceil((p + mfee) * 100 - 1e-9) / 100)
        na = np.maximum(np.maximum((1 - b) - 0.01, (1 - a) + 0.01), np.ceil(((1 - p) + mfee) * 100 - 1e-9) / 100)
        lg, sh = (pos > 0) & (ya <= 0.99), (pos < 0) & (na <= 0.99)
        side = np.where(lg, 3, side); opx = np.where(lg, ya, opx); osz = np.where(lg, pos, osz)
        side = np.where(sh, 4, side); opx = np.where(sh, na, opx); osz = np.where(sh, -pos, osz)
        peak = np.maximum(peak, eq); mdd = np.maximum(mdd, (peak - eq) / peak); tim += pos != 0
        trace["pos"][t] = pos[trace_idx]; trace["eq"][t] = eq[trace_idx]
    return _result(cash, pos, Y, fees, fills, mdd, tim, edge_sum, edge_n, trace)


def summarize(r):
    pnl = r["pnl"]; traded = r["fills"] > 0
    sd = pnl.std(ddof=1)
    return {
        "mean_pnl": pnl.mean(), "se": sd / np.sqrt(len(pnl)), "median": np.median(pnl), "sd": sd,
        "p5": np.percentile(pnl, 5), "p95": np.percentile(pnl, 95),
        "win_rate": (pnl[traded] > 0).mean() if traded.any() else np.nan,
        "p_loss_gt20": (pnl < -20).mean(), "sharpe_night": pnl.mean() / sd if sd > 0 else np.nan,
        "traded": traded.mean(), "fills": r["fills"][traded].mean() if traded.any() else 0.0,
        "fees": r["fees"].mean(), "mdd": r["mdd"].mean(),
        "mins_in_mkt": r["tim"][traded].mean() if traded.any() else 0.0,
        "held_to_settle": r["held_to_settle"].mean(), "entry_edge": np.nanmean(r["entry_edge"]),
    }
