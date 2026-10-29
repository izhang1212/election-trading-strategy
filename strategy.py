"""The frozen trading rules, for one market, one decision tick at a time.
A line-by-line port of simulation/sim_strategy.py:run_taker (which is vectorised over simulated nights):

  move filter : no entries and no model-driven exits in a tick where the mid moved more than 1c
  exits       : sell YES when bid - fee/contract > fair, or bid >= 97c; same for NO with (1 - ask) vs (1 - fair)
  entries     : buy YES when fair - ask - fee/contract - 3c > 0 (NO symmetrically, vs 1 - fair and 1 - bid)
  sizing      : quarter-Kelly f = (q - px)/(1 - px) of equity, capped at 50% of equity, never flip sides
  fees        : Kalshi taker fee ceil(0.07 * C * P * (1 - P)) to the cent, per order
Position convention: pos > 0 = long YES contracts, pos < 0 = long NO contracts.
"""
import math
from dataclasses import dataclass, field
import yaml

S = yaml.safe_load(open("config.yaml"))["strategy"]
FEE_RATE = yaml.safe_load(open("config.yaml"))["kalshi"]["fee_rate"]
EPS = 1e-9   # prices are on a 0.1c grid; compare with a tolerance so 0.01 + float noise is not "> 0.01"


def fee(C, P):
    return math.ceil(FEE_RATE * C * P * (1 - P) * 100 - EPS) / 100 if C > 0 else 0.0


@dataclass
class Order:
    action: str     # "buy" | "sell"
    side: str       # "yes" | "no"
    count: int
    price: float    # price of the contract being traded (YES price for side=yes, NO price for side=no)
    reason: str


@dataclass
class Book:
    cash: float
    pos: int = 0
    fees: float = 0.0
    fills: list = field(default_factory=list)

    def equity(self, mid):
        return self.cash + (self.pos * mid if self.pos > 0 else -self.pos * (1 - mid))

    def apply(self, o, t):
        """Record a fill (paper mode, or a confirmed live fill)."""
        f = fee(o.count, o.price)
        sign = 1 if o.side == "yes" else -1
        if o.action == "buy":
            self.cash -= o.count * o.price + f; self.pos += sign * o.count
        else:
            self.cash += o.count * o.price - f; self.pos -= sign * o.count
        self.fees += f; self.fills.append(dict(t=t, fee=f, **o.__dict__))


def jumped(mid, prev_mid):
    """Did the mid move by more than the filter threshold since the last tick? Exact comparison. NB
    simulation/sim_strategy.py used a raw float `> 0.01`, under which an exact 1c move is blocked at some price levels
    (0.655-0.645 = 0.0100000000000000009) and not at others (0.30-0.29 = 0.0099999999999999998)."""
    move = abs(mid - prev_mid) if prev_mid is not None and mid is not None else 0.0
    return move >= S["move_filter"] - EPS if S["move_filter_rule"] == "ge" else move > S["move_filter"] + EPS


class MoveFilter:
    """Blocks entries and model-driven exits for `window` ticks after a qualifying move (window 1 = only the tick of
    the move, the simulated rule). The window is set to our measured data latency (simulation/summary_table_v2.md:
    at 5 min latency a 1-tick filter costs ~$2/night; a latency-length filter removes most of that)."""
    def __init__(self, window=None):
        self.window = window or S["filter_window_minutes"]; self.prev = None; self.since = 10 ** 6

    def update(self, mid):
        self.since = 0 if jumped(mid, self.prev) else self.since + 1
        if mid is not None:
            self.prev = mid
        return self.since < self.window


def decide(p, bid, ask, mid, prev_mid, book, depth=None, blocked=None):
    """Orders for this tick. p = our fair value (P[YES]); bid/ask = YES top of book; depth = max contracts per order.
    blocked: from a MoveFilter (multi-tick window); if None, the one-tick rule is applied to prev_mid -> mid."""
    depth = depth or S["max_contracts_per_order"]
    if None in (p, bid, ask, mid):
        return []
    if blocked is None:
        blocked = jumped(mid, prev_mid)
    orders, pos, cash = [], book.pos, book.cash

    # exits (run_taker order: YES exit, then NO exit)
    if pos > 0:
        model_exit = (bid - FEE_RATE * bid * (1 - bid)) > p and not blocked
        if model_exit or bid >= S["exit_price"] - EPS:
            C = min(pos, depth)
            orders.append(Order("sell", "yes", C, bid, "exit_fair" if model_exit else "exit_97"))
            cash += C * bid - fee(C, bid); pos -= C
    elif pos < 0:
        c = 1 - ask   # NO bid
        model_exit = (c - FEE_RATE * c * (1 - c)) > 1 - p and not blocked
        if model_exit or c >= S["exit_price"] - EPS:
            C = min(-pos, depth)
            orders.append(Order("sell", "no", C, round(c, 4), "exit_fair" if model_exit else "exit_97"))
            cash += C * c - fee(C, c); pos += C

    eq = cash + (pos * mid if pos > 0 else -pos * (1 - mid))
    # entries
    for side in (1, -1):
        px = ask if side == 1 else round(1 - bid, 4)
        q = p if side == 1 else 1 - p
        if px <= 0 or px >= 1:
            continue
        edge = q - px - FEE_RATE * px * (1 - px) - S["buffer"]
        if edge <= 0 or side * pos < 0 or blocked:
            continue
        f = (q - px) / (1 - px)
        tgt = math.floor(max(min(S["kelly"] * f * eq, S["max_exposure"] * eq), 0) / px)
        have = abs(pos) if side * pos > 0 else 0
        C = max(tgt - have, 0)
        C = min(C, depth, math.floor(max(cash - 0.05, 0) / (px + FEE_RATE * px * (1 - px))))
        if C > 0:
            orders.append(Order("buy", "yes" if side == 1 else "no", int(C), px, f"entry edge={q - px:.3f}"))
            cash -= C * px + fee(C, px); pos += side * C
    return orders


class KillSwitch:
    """Trips once and stays tripped: after that, no orders are sent (positions are handled by hand, RUNBOOK.md)."""
    def __init__(self, cfg):
        self.cfg, self.reason, self.parse_errors = cfg, None, 0

    def check(self, minutes_since_new_file, outstanding, fair, mid, anomalies):
        if self.reason:
            return self.reason
        if anomalies:
            self.reason = f"parse anomaly: {anomalies[0]}"
        elif self.parse_errors >= self.cfg["max_parse_errors"]:
            self.reason = f"{self.parse_errors} consecutive parse errors"
        elif outstanding and minutes_since_new_file is not None and minutes_since_new_file > self.cfg["stale_feed_minutes"]:
            self.reason = f"results feed stale for {minutes_since_new_file:.0f} min"
        elif fair is not None and mid is not None and abs(fair - mid) > self.cfg["max_model_market_gap"]:
            self.reason = f"model-market gap {fair - mid:+.2f}"
        return self.reason
