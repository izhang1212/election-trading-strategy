"""Stage 5: the trading strategy (Phase 1 winner): take liquidity when our fair value beats the price by more than
fees + a buffer; skip any minute in which the price just jumped (it may reflect data we haven't processed);
quarter-Kelly sizing; sell when the bid beats fair value or reaches 97c. Depth caps fills per minute.

v2 options (defaults reproduce the original exactly):
  filter_rule   None = the original float comparison |dmid| > 0.01 (an exact 1c move is blocked at some price
                levels and not others); "gt" = exact "more than 1c"; "ge" = exact "1c or more"
  filter_window block entries/model exits for this many minutes after a qualifying move (original: 1)"""
import numpy as np

TF = 0.07                   # Kalshi-style taker fee: ceil(0.07 * C * P * (1-P)) cents
KELLY, CAP, BUF, FILTER = 0.25, 0.5, 0.03, 0.01


def fee(C, P):
    return np.where(C > 0, np.ceil(TF * C * P * (1 - P) * 100 - 1e-9) / 100, 0.0)


def run_taker(P_O, mkt, Y, bankroll, depth, filter_rule=None, filter_window=1):
    mid, bid, ask = mkt["mid"], mkt["bid"], mkt["ask"]
    n, T1 = P_O.shape
    cash = np.full(n, float(bankroll)); pos = np.zeros(n); fees = np.zeros(n); fills = np.zeros(n)
    peak = cash.copy(); mdd = np.zeros(n); tim = np.zeros(n); first_trade = np.full(n, -1)
    edge_sum = np.zeros(n); edge_n = np.zeros(n); vol = np.zeros(n)
    last_jump = np.full(n, -10 ** 6)
    for t in range(1, T1):
        p, b, a, m = P_O[:, t], bid[:, t], ask[:, t], mid[:, t]
        move = np.abs(m - mid[:, t - 1])
        jump = move > FILTER if filter_rule is None else \
            (move > FILTER + 1e-9 if filter_rule == "gt" else move >= FILTER - 1e-9)
        last_jump = np.where(jump, t, last_jump)
        blocked = (t - last_jump) < filter_window
        # exits
        ex = (pos > 0) & ((((b - TF * b * (1 - b)) > p) & ~blocked) | (b >= 0.97))
        C = np.minimum(np.where(ex, pos, 0), depth); fz = fee(C, b)
        cash += C * b - fz; fees += fz; fills += C > 0; pos -= C; vol += C * b
        c = 1 - a
        ex = (pos < 0) & ((((c - TF * c * (1 - c)) > 1 - p) & ~blocked) | (c >= 0.97))
        C = np.minimum(np.where(ex, -pos, 0), depth); fz = fee(C, c)
        cash += C * c - fz; fees += fz; fills += C > 0; pos += C; vol += C * c
        eq = cash + np.where(pos > 0, pos * m, 0) + np.where(pos < 0, -pos * (1 - m), 0)
        # entries
        for side in (1, -1):
            px = a if side == 1 else 1 - b
            q = p if side == 1 else 1 - p
            edge = q - px - TF * px * (1 - px) - BUF
            f = (q - px) / (1 - px)
            tgt = np.floor(np.maximum(np.minimum(KELLY * f * eq, CAP * eq), 0) / px)
            have = np.where(side * pos > 0, np.abs(pos), 0)
            ok = (edge > 0) & (side * pos >= 0) & ~blocked
            C = np.where(ok, np.maximum(tgt - have, 0), 0)
            C = np.minimum(C, depth)
            C = np.minimum(C, np.floor(np.maximum(cash - 0.05, 0) / (px + TF * px * (1 - px))))
            fz = fee(C, px)
            cash -= C * px + fz; fees += fz; pos += side * C; fills += C > 0; vol += C * px
            edge_sum += np.where(C > 0, q - px, 0); edge_n += C > 0
            first_trade = np.where((C > 0) & (first_trade < 0), t, first_trade)
        eq = cash + np.where(pos > 0, pos * m, 0) + np.where(pos < 0, -pos * (1 - m), 0)
        peak = np.maximum(peak, eq); mdd = np.maximum(mdd, (peak - eq) / peak); tim += pos != 0
    held = pos != 0
    cash += np.where(pos > 0, pos * Y, 0) + np.where(pos < 0, -pos * (1 - Y), 0)
    return dict(pnl=cash - bankroll, ret=(cash - bankroll) / bankroll, fees=fees, fills=fills, mdd=mdd, tim=tim,
                held=held, volume=vol, first_trade=first_trade,
                entry_edge=np.where(edge_n > 0, edge_sum / np.maximum(edge_n, 1), np.nan))


def summarize(r, bankroll, cluster=None):
    """cluster: optional array of history-draw ids; then se is the cluster-robust standard error of the mean
    (nights from the same fitted model are not independent)."""
    pnl = r["pnl"]; traded = r["fills"] > 0; sd = pnl.std(ddof=1)
    se = sd / np.sqrt(len(pnl))
    if cluster is not None:
        g = np.unique(cluster); G = len(g)
        resid = np.array([(pnl[cluster == k] - pnl.mean()).sum() for k in g])
        se = np.sqrt(G / (G - 1) * (resid ** 2).sum()) / len(pnl)
    return dict(mean_pnl=pnl.mean(), se=se, median=np.median(pnl), sd=sd,
                p5=np.percentile(pnl, 5), p95=np.percentile(pnl, 95),
                mean_ret_pct=100 * r["ret"].mean(),
                win_rate=(pnl[traded] > 0).mean() if traded.any() else np.nan,
                p_loss_20pct=(r["ret"] < -0.20).mean(), sharpe_night=pnl.mean() / sd if sd > 0 else np.nan,
                traded=traded.mean(), fills=r["fills"][traded].mean() if traded.any() else 0.0,
                fees=r["fees"].mean(), mdd_pct=100 * r["mdd"].mean(), volume=r["volume"].mean(),
                held_to_settle=r["held"].mean(), entry_edge=np.nanmean(r["entry_edge"]))
