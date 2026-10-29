"""Tie results capture -> parse -> nowcast -> strategy -> (paper or live) orders, one decision tick per minute.

Reads what the recorders write (it never fetches results itself):
  data/live/raw/<target>/index.jsonl + files   (scrape_results.py)
  data/live/kalshi/top.json                    (record_markets.py)
and writes one line per tick per state to data/live/decisions/<state>.jsonl with fair value, book, raw tally,
fraction reported, batches applied, orders, position, cash and kill-switch state.

Safety rules built in:
  * no county-level feed for a state -> fair value is None -> no orders (TX until county feeds are added);
  * the statewide level is calibrated to the last mid before that state's first batch (pre-registered);
  * the kill switch trips once and stays tripped; then no orders are sent at all;
  * --mode live also needs --i-understand-live.

Usage:
  python run_live.py [--mode record|paper|live] [--i-understand-live]
  python run_live.py --replay <raw root> [--market <rest_*.jsonl>] [--mode paper]   # same code, simulated clock
"""
import argparse, json, os, time, uuid
from datetime import datetime, timezone
import requests
import yaml
from nowcast import Nowcast
from parse_results import diff, parse_file
from strategy import Book, KillSwitch, MoveFilter, decide

CFG = yaml.safe_load(open("config.yaml"))
LIVE = CFG["paths"]["live"]
COUNTY_PARSERS = {"tx_county_json", "ga_export_json"}   # parsers that return county rows
RACE = {"TX": "U. S. SENATOR", "GA": "GOVERNOR"}


def parse_utc(s):
    s = s.replace("Z", "")
    for fmt in ("%Y%m%dT%H%M%S.%f", "%Y-%m-%dT%H:%M:%S.%f", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    raise ValueError(s)


# ---------------------------------------------------------------- sources (live or replay share one interface)
class Results:
    """New raw files of one target with receive time <= clock, in receive order."""
    def __init__(self, target, root):
        self.t, self.dir, self.seen = target, os.path.join(root, target["name"]), 0

    def _all(self):
        idx = os.path.join(self.dir, "index.jsonl")
        if os.path.exists(idx):
            out = []
            for line in open(idx):
                r = json.loads(line)
                if r.get("file") and r.get("status") == 200:
                    out.append((parse_utc(r["recv_utc"]), r["recv_utc"], os.path.join(self.dir, r["file"])))
            return out
        if not os.path.isdir(self.dir):
            return []
        fs = sorted(f for f in os.listdir(self.dir) if not f.endswith((".jsonl", ".part")))
        return [(parse_utc(f.split("_")[0]), f.split("_")[0], os.path.join(self.dir, f)) for f in fs]

    def new(self, clock):
        files = [f for f in self._all() if f[0] <= clock]
        out = files[self.seen:]; self.seen = len(files)
        return out


class Market:
    def __init__(self, replay_file=None):
        self.rows = None
        if replay_file:   # REST log from record_markets.py: rebuild top of book over time
            from record_markets import _levels
            self.rows = []
            for line in open(replay_file):
                r = json.loads(line)
                book = (r.get("orderbook") or {}).get("orderbook_fp") or (r.get("orderbook") or {}).get("orderbook")
                if not book:
                    continue
                yes, no = _levels(book, "yes"), _levels(book, "no")
                self.rows.append((parse_utc(r["recv_utc"]), r["ticker"],
                                  dict(bid=max(yes) if yes else None, ask=round(1 - max(no), 4) if no else None)))

    def top(self, ticker, clock):
        if self.rows is None:
            try:
                return json.load(open(os.path.join(LIVE, "kalshi", "top.json")))["books"].get(ticker)
            except (OSError, ValueError):
                return None
        last = None
        for t, tk, b in self.rows:
            if t > clock:
                break
            if tk == ticker:
                last = b
        return last


# ---------------------------------------------------------------- live orders (verify with a 1-contract test, RUNBOOK)
def send_order(ticker, o):
    from record_markets import sign_headers
    path = "/trade-api/v2/portfolio/orders"
    body = {"ticker": ticker, "action": o.action, "side": o.side, "count": o.count, "type": "limit",
            f"{o.side}_price_dollars": f"{o.price:.4f}", "time_in_force": "immediate_or_cancel",
            "client_order_id": str(uuid.uuid4())}
    r = requests.post(CFG["kalshi"]["rest"] + "/portfolio/orders", json=body,
                      headers=sign_headers("POST", path), timeout=10)
    return r.status_code, r.json() if r.content else {}


def live_position(ticker):
    from record_markets import sign_headers
    path = "/trade-api/v2/portfolio/positions"
    r = requests.get(CFG["kalshi"]["rest"] + "/portfolio/positions", params={"ticker": ticker},
                     headers=sign_headers("GET", path), timeout=10).json()
    mp = (r.get("market_positions") or [{}])[0]
    return int(float(mp.get("position_fp", mp.get("position", 0))))


# ---------------------------------------------------------------- one state's pipeline
class StateRun:
    def __init__(self, state, root, mode, fitted=None, out_dir=None):
        self.state, self.mode = state, mode
        self.targets = [t for t in CFG["scrape"]["targets"] if t["state"] == state]
        self.sources = [Results(t, root) for t in self.targets]
        self.has_county_feed = any(t["parser"] in COUNTY_PARSERS for t in self.targets)
        self.nc = Nowcast(state, fitted) if self.has_county_feed else None
        self.ticker = CFG["kalshi"]["primary"][state]
        self.book = Book(cash=float(CFG["strategy"]["bankroll"]))
        self.kill = KillSwitch(CFG["kill_switch"])
        self.mfilter = MoveFilter()
        self.prev = {}; self.anchored = False; self.last_mid = None; self.prev_mid = None
        self.last_file_t = None; self.statewide = {}
        out_dir = out_dir or os.path.join(LIVE, "decisions")
        os.makedirs(out_dir, exist_ok=True)
        self.log = open(os.path.join(out_dir, f"{state}.jsonl"), "a")

    def tick(self, clock, market):
        applied, unknown, anomalies = 0, [], []
        top = market.top(self.ticker, clock) or {}
        bid, ask = top.get("bid"), top.get("ask")
        mid = round((bid + ask) / 2, 4) if bid is not None and ask is not None else None
        for src in self.sources:
            if src.t["parser"] == "raw":          # capture-only target: parsed after the election, not live
                continue
            for t_recv, recv_utc, path in src.new(clock):
                try:
                    snap = parse_file(path, src.t["parser"], RACE[self.state], recv_utc)
                    self.kill.parse_errors = 0
                except Exception as e:
                    self.kill.parse_errors += 1
                    self.log.write(json.dumps(dict(utc=recv_utc, event="parse_error", file=path, error=repr(e))) + "\n")
                    continue
                self.last_file_t = t_recv
                self.statewide[src.t["name"]] = snap["statewide"]
                batches, an = diff(self.prev.get(src.t["name"]), snap); self.prev[src.t["name"]] = snap
                anomalies += an
                if self.nc is None or not batches:
                    continue
                if not self.anchored:                     # pre-registered anchor: last mid before first batch
                    self.nc.calibrate(self.last_mid if self.last_mid is not None else 0.5)
                    self.anchored = True
                    self.log.write(json.dumps(dict(utc=recv_utc, event="anchor", prior=self.last_mid)) + "\n")
                for b in batches:
                    if self.nc.apply_batch(b):
                        applied += 1
                    elif b["method"] in ("early", "ed"):
                        unknown.append(b["county"])
        if not self.anchored and mid is not None:
            self.last_mid = mid
        fair = self.nc.prob() if self.nc is not None and self.anchored else None
        stale = (clock - self.last_file_t) / 60 if self.last_file_t else None
        outstanding = self.nc.outstanding() if self.nc is not None and self.anchored else False
        reason = self.kill.check(stale, outstanding, fair, mid, anomalies)
        orders = []
        blocked = self.mfilter.update(mid)
        if self.mode in ("paper", "live") and not reason:
            orders = decide(fair, bid, ask, mid, self.prev_mid, self.book, blocked=blocked)
            for o in orders:
                if self.mode == "paper":
                    self.book.apply(o, clock)
                else:
                    status, resp = send_order(self.ticker, o)
                    self.log.write(json.dumps(dict(utc=clock, event="order", order=o.__dict__, status=status, resp=resp)) + "\n")
            if self.mode == "live" and orders:
                self.book.pos = live_position(self.ticker)    # reconcile with the exchange, not our arithmetic
        self.prev_mid = mid
        rec = dict(utc=datetime.fromtimestamp(clock, timezone.utc).isoformat(), state=self.state, mode=self.mode,
                   fair=fair, bid=bid, ask=ask, mid=mid, book_source=top.get("source"),
                   raw=self.nc.raw_share() if self.nc is not None and self.nc.Trep else None,
                   frac=self.nc.frac_reported() if self.nc is not None and self.anchored else None,
                   statewide=self.statewide, applied=applied, unknown=sorted(set(unknown)), anomalies=anomalies,
                   orders=[o.__dict__ for o in orders], pos=self.book.pos, cash=round(self.book.cash, 4),
                   fees=round(self.book.fees, 4), kill=reason, county_feed=self.has_county_feed)
        self.log.write(json.dumps(rec, default=float) + "\n"); self.log.flush()
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=CFG["mode"], choices=["record", "paper", "live"])
    ap.add_argument("--i-understand-live", action="store_true")
    ap.add_argument("--replay", help="raw root with <target>/ subfolders (simulated clock)")
    ap.add_argument("--market", help="REST log to replay prices from (with --replay)")
    ap.add_argument("--states", default="TX,GA")
    ap.add_argument("--fitted", help="fitted-inputs dir (default config paths.fitted)")
    a = ap.parse_args()
    if a.mode == "live" and not a.i_understand_live:
        raise SystemExit("--mode live sends real orders; add --i-understand-live")
    if a.replay and a.mode == "live":
        raise SystemExit("replay cannot run in live mode")
    root = a.replay or os.path.join(LIVE, "raw")
    runs = [StateRun(s, root, "paper" if a.replay and a.mode == "paper" else a.mode, a.fitted,
                     os.path.join(a.replay, "decisions") if a.replay else None) for s in a.states.split(",")]
    market = Market(a.market)
    step = CFG["strategy"]["decision_seconds"]
    if a.replay:
        times = [f[0] for r in runs for s in r.sources for f in s._all()]
        if not times:
            raise SystemExit(f"no raw files under {root}")
        clock, end = min(times) - step, max(times) + 30 * step
        while clock <= end:
            for r in runs:
                r.tick(clock, market)
            clock += step
        print(f"replayed {len(times)} files; decisions in {a.replay}/decisions/")
        return
    print(f"run_live: mode={a.mode} states={a.states}", flush=True)
    while True:
        t0 = time.time()
        for r in runs:
            rec = r.tick(t0, market)
            print(f"{rec['utc'][11:19]} {rec['state']} fair={rec['fair']} mid={rec['mid']} applied={rec['applied']} "
                  f"pos={rec['pos']} kill={rec['kill']}", flush=True)
        time.sleep(max(0.0, step - (time.time() - t0)))


if __name__ == "__main__":
    main()
