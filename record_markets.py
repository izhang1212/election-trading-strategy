"""Record Kalshi order books and trades to data/live/kalshi/.

Two independent recorders, so one failing never stops the other:
  WebSocket  (needs an API key, even for public data): orderbook_delta + trade + ticker for all tickers in
             config.yaml. Every message is written raw with the local receive time. seq is checked per
             subscription; on a gap the gap is logged and the connection is rebuilt, which re-sends snapshots.
  REST       (no key needed): order book + market summary for every ticker every rest_snapshot_seconds.

Files (one per UTC day):
  ws_YYYYMMDD.jsonl    {"recv_ns", "recv_utc", "msg": <raw message>}  plus {"event": "gap"|"connect"|...} lines
  rest_YYYYMMDD.jsonl  {"recv_ns", "recv_utc", "ticker", "orderbook": ..., "market": ...}
  top.json             latest top of book per ticker (atomic rewrite), read by run_live.py

Usage: python record_markets.py           (credentials from KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH)
"""
import asyncio, base64, json, os, threading, time
from datetime import datetime, timezone
import requests
import yaml

CFG = yaml.safe_load(open("config.yaml"))
K = CFG["kalshi"]
OUT = os.path.join(CFG["paths"]["live"], "kalshi")
TICKERS = list(K["primary"].values()) + K["related"]
WS_PATH = "/trade-api/ws/v2"


def load_dotenv(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"'))


def sign_headers(method, path):
    """Kalshi API-key auth: RSA-PSS(SHA256) over timestamp_ms + METHOD + path (no query string)."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key_id, key_path = os.environ.get(K["key_id_env"]), os.environ.get(K["private_key_env"])
    if not key_id or not key_path:
        return None
    key = serialization.load_pem_private_key(open(key_path, "rb").read(), password=None)
    ts = str(int(time.time() * 1000))
    sig = key.sign((ts + method + path).encode(),
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                   hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": key_id, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts}


def now():
    ns = time.time_ns()
    return ns, datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Log:
    """Append-only JSONL, one file per UTC day, flushed every line and fsynced at most once a second."""
    def __init__(self, prefix):
        self.prefix, self.f, self.day, self.last_sync = prefix, None, None, 0.0
        self.lock = threading.Lock()

    def write(self, obj):
        with self.lock:
            day = datetime.now(timezone.utc).strftime("%Y%m%d")
            if day != self.day:
                if self.f:
                    self.f.close()
                os.makedirs(OUT, exist_ok=True)
                self.f, self.day = open(os.path.join(OUT, f"{self.prefix}_{day}.jsonl"), "a"), day
            self.f.write(json.dumps(obj) + "\n"); self.f.flush()
            if time.time() - self.last_sync > 1:
                os.fsync(self.f.fileno()); self.last_sync = time.time()


# ---------------------------------------------------------------- order book state (prices in dollars)
def _levels(msg, side):
    """Snapshot levels in either the legacy cents format ('yes': [[64, 100]]) or fixed-point dollars
    ('yes_dollars': [['0.6400', '100.00']])."""
    if f"{side}_dollars" in msg:
        return {round(float(p), 4): float(q) for p, q in msg[f"{side}_dollars"] or []}
    return {round(p / 100, 4): float(q) for p, q in msg.get(side) or []}


class Books:
    def __init__(self):
        self.b, self.src, self.lock = {}, {}, threading.Lock()

    def snapshot(self, ticker, yes, no, source):
        with self.lock:
            self.b[ticker] = {"yes": yes, "no": no}; self.src[ticker] = source

    def delta(self, msg):
        t = msg.get("market_ticker"); side = msg.get("side")
        if t not in self.b or side not in ("yes", "no"):
            return
        p = round(float(msg["price_dollars"]) if "price_dollars" in msg else msg["price"] / 100, 4)
        d = float(msg["delta_fp"]) if "delta_fp" in msg else float(msg["delta"])
        with self.lock:
            lv = self.b[t][side]; lv[p] = lv.get(p, 0.0) + d
            if lv[p] <= 1e-9:
                del lv[p]
            self.src[t] = "ws"

    def top(self):
        with self.lock:
            out = {}
            for t, bk in self.b.items():
                yb = max(bk["yes"]) if bk["yes"] else None          # best YES bid
                nb = max(bk["no"]) if bk["no"] else None            # best NO bid -> YES ask = 1 - nb
                out[t] = dict(bid=yb, ask=round(1 - nb, 4) if nb is not None else None,
                              bid_size=bk["yes"].get(yb) if yb is not None else None,
                              ask_size=bk["no"].get(nb) if nb is not None else None, source=self.src.get(t))
            return out


BOOKS = Books()


def write_top(updated_utc):
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "top.json.part")
    json.dump(dict(updated_utc=updated_utc, books=BOOKS.top()), open(tmp, "w"))
    os.replace(tmp, os.path.join(OUT, "top.json"))


# ---------------------------------------------------------------- REST backup
def rest_loop(stop):
    log = Log("rest"); s = requests.Session()
    while not stop.is_set():
        start = time.time()
        for t in TICKERS:
            try:
                ob = s.get(f"{K['rest']}/markets/{t}/orderbook", timeout=10).json()
                mk = s.get(f"{K['rest']}/markets/{t}", timeout=10).json()
                ns, utc = now()
                log.write(dict(recv_ns=ns, recv_utc=utc, ticker=t, orderbook=ob, market=mk.get("market")))
                book = ob.get("orderbook_fp") or ob.get("orderbook") or {}
                if BOOKS.src.get(t) != "ws":          # WS is primary; REST only fills in when WS is absent
                    BOOKS.snapshot(t, _levels(book, "yes"), _levels(book, "no"), "rest")
            except Exception as e:
                ns, utc = now(); log.write(dict(recv_ns=ns, recv_utc=utc, ticker=t, error=repr(e)))
        write_top(now()[1])
        stop.wait(max(0.0, K["rest_snapshot_seconds"] - (time.time() - start)))


# ---------------------------------------------------------------- WebSocket
async def ws_session(log):
    import websockets
    headers = sign_headers("GET", WS_PATH)
    async with websockets.connect(K["ws"], additional_headers=headers, ping_interval=10, ping_timeout=10,
                                  max_size=2 ** 24) as ws:
        ns, utc = now(); log.write(dict(recv_ns=ns, recv_utc=utc, event="connect"))
        for i, ch in enumerate(("orderbook_delta", "trade", "ticker"), start=1):
            await ws.send(json.dumps({"id": i, "cmd": "subscribe",
                                      "params": {"channels": [ch], "market_tickers": TICKERS}}))
        last_seq, last_top = {}, 0.0
        async for raw in ws:
            ns, utc = now()
            msg = json.loads(raw)
            log.write(dict(recv_ns=ns, recv_utc=utc, msg=msg))
            sid, seq, typ = msg.get("sid"), msg.get("seq"), msg.get("type")
            if seq is not None and sid is not None:
                if sid in last_seq and seq != last_seq[sid] + 1:
                    log.write(dict(recv_ns=ns, recv_utc=utc, event="gap", sid=sid, expected=last_seq[sid] + 1, got=seq))
                    raise ConnectionError(f"seq gap on sid {sid}")   # reconnect -> fresh snapshots
                last_seq[sid] = seq
            body = msg.get("msg") or {}
            if typ == "orderbook_snapshot":
                BOOKS.snapshot(body["market_ticker"], _levels(body, "yes"), _levels(body, "no"), "ws")
            elif typ == "orderbook_delta":
                BOOKS.delta(body)
            elif typ == "error":
                print("ws error:", msg, flush=True)
            if time.time() - last_top > 1:
                write_top(utc); last_top = time.time()


async def ws_loop():
    log, backoff = Log("ws"), 1
    while True:
        try:
            await ws_session(log); backoff = 1
        except Exception as e:
            ns, utc = now(); log.write(dict(recv_ns=ns, recv_utc=utc, event="disconnect", error=repr(e)))
            print(f"{utc} ws disconnected: {e!r}; retry in {backoff}s", flush=True)
            for t in list(BOOKS.src):        # stale WS books: let REST refill until the WS snapshot arrives
                BOOKS.src[t] = "stale"
            await asyncio.sleep(backoff); backoff = min(backoff * 2, 30)


def main():
    load_dotenv()
    stop = threading.Event()
    threading.Thread(target=rest_loop, args=(stop,), daemon=True).start()
    print(f"recording {TICKERS} -> {OUT}", flush=True)
    if sign_headers("GET", WS_PATH) is None:
        print("no Kalshi API key in the environment: REST-only recording (set KALSHI_API_KEY_ID and "
              "KALSHI_PRIVATE_KEY_PATH for the WebSocket feed)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop.set(); return
    try:
        asyncio.run(ws_loop())
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
