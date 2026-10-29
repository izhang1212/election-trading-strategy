"""Raw capture of results feeds. This is the one script that must not fail, so it does no parsing.

Each target in config.yaml (scrape.targets) has a kind:
  url         poll one URL (optionally with a cache-busting query parameter)
  clarity     Clarity ENR county site: discover the election id if not given (elections.json entry dated
              election day, or a link on the county's own elections page), poll <eid>/current_ver.txt, and
              when the version changes fetch each listed file under <eid>/<version>/
  page_links  poll a page and also fetch every linked file whose URL matches link_regex, once per URL
              (counties that post cumulative CSV/PDF reports during the night)

Every changed body is written as data/live/raw/<target>/<UTC receive time>_<sha8>[_<part>].<ext> and logged
in data/live/raw/<target>/index.jsonl:
  {recv_utc, recv_ns, latency_ms, url, status, last_modified, etag, content_type, sha256, bytes, changed, file, part}
Unchanged bodies (same SHA-256 as that URL's last good body) are logged with changed=false, not written.
Errors are logged and retried; nothing here exits on its own.

Usage:
  python scrape_results.py                 # all targets
  python scrape_results.py tx_harris ...   # only the named targets
"""
import hashlib, json, os, re, sys, threading, time
from datetime import datetime, timezone
from urllib.parse import urljoin
import requests
import yaml

CFG = yaml.safe_load(open("config.yaml"))
ROOT = os.path.join(CFG["paths"]["live"], "raw")
EXT = {"application/json": "json", "text/html": "html", "text/xml": "xml", "application/xml": "xml",
       "text/csv": "csv", "application/zip": "zip", "application/pdf": "pdf", "text/plain": "txt"}


def utc_stamp(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


class Target:
    """Disk state of one target: the index file and the last good SHA per URL (restart-safe)."""
    def __init__(self, t):
        self.t, self.dir = t, os.path.join(ROOT, t["name"])
        os.makedirs(self.dir, exist_ok=True)
        self.idx = os.path.join(self.dir, "index.jsonl")
        self.last_sha, self.lock = {}, threading.Lock()
        if os.path.exists(self.idx):
            for line in open(self.idx):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("file"):
                    self.last_sha[r["url"]] = r["sha256"]
        self.s = requests.Session(); self.s.headers["User-Agent"] = CFG["scrape"]["user_agent"]

    def log(self, rec):
        with self.lock, open(self.idx, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

    def fetch(self, url, part=None, bust=False, save=True):
        """GET url; write the body if it changed; return (record, body)."""
        t0 = time.time_ns()
        r = self.s.get(url, timeout=30, params={"nocache": int(time.time())} if bust else None,
                       headers={"Cache-Control": "no-cache"})
        recv = time.time_ns(); body = r.content
        key = body   # change detection ignores volatile fragments (e.g. a Cloudflare script with a per-request id)
        if self.t.get("hash_ignore"):
            key = re.sub(self.t["hash_ignore"].encode(), b"", body, flags=re.S)
        sha = hashlib.sha256(key).hexdigest()
        rec = dict(recv_utc=utc_stamp(recv), recv_ns=recv, latency_ms=round((recv - t0) / 1e6, 1), url=url,
                   status=r.status_code, last_modified=r.headers.get("Last-Modified"), etag=r.headers.get("ETag"),
                   content_type=r.headers.get("Content-Type"), sha256=sha, bytes=len(body),
                   changed=r.status_code == 200 and sha != self.last_sha.get(url), file=None, part=part)
        if rec["changed"] and save:
            ext = EXT.get((rec["content_type"] or "").split(";")[0].strip(), "bin")
            tag = "_" + re.sub(r"[^A-Za-z0-9-]", "-", os.path.splitext(part)[0]) if part else ""
            fn = f"{rec['recv_utc']}_{sha[:8]}{tag}.{ext}"
            tmp = os.path.join(self.dir, fn + ".part")
            with open(tmp, "wb") as f:
                f.write(body); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, os.path.join(self.dir, fn))       # atomic: readers never see a partial file
            rec["file"] = fn; self.last_sha[url] = sha
        self.log(rec)
        if rec["file"] or r.status_code != 200:
            print(f"{rec['recv_utc']} {self.t['name']:14s} {r.status_code} {len(body):>9,}B "
                  f"{'NEW ' + rec['file'] if rec['file'] else url}", flush=True)
        return rec, body


# ---------------------------------------------------------------- target kinds
def poll_url(tg, state):
    tg.fetch(tg.t["url"], bust=tg.t.get("cache_bust", False))
    for extra in tg.t.get("also", []):               # companion files, e.g. Harris pr.json
        tg.fetch(extra, part=os.path.basename(extra), bust=tg.t.get("cache_bust", False))


def clarity_discover(tg):
    base, day = tg.t["base"].rstrip("/"), CFG["election_day_clarity"]
    try:
        for e in tg.s.get(f"{base}/elections.json", timeout=20).json():
            if e.get("Date", "").startswith(day):
                return e["EID"]
    except Exception:
        pass
    county = base.rsplit("/", 1)[1]
    for page in tg.t.get("discover_pages", []):
        try:
            ids = re.findall(rf"clarityelections\.com/TX/{county}/(\d{{5,7}})", tg.s.get(page, timeout=20).text)
            if ids:
                return max(ids, key=int)
        except Exception:
            pass
    return None


def poll_clarity(tg, state):
    if not state.get("eid"):
        state["eid"] = tg.t.get("eid") or clarity_discover(tg)
        if not state["eid"]:
            return
        tg.log(dict(recv_utc=utc_stamp(time.time_ns()), event="clarity_eid", eid=state["eid"]))
    base = f"{tg.t['base'].rstrip('/')}/{state['eid']}"
    rec, body = tg.fetch(f"{base}/current_ver.txt", part="current_ver")
    ver = body.decode(errors="replace").strip() if rec["status"] == 200 else None
    if ver and ver.isdigit() and ver != state.get("ver"):
        for f in tg.t.get("files", ["json/en/summary.json", "reports/detailxml.zip"]):
            tg.fetch(f"{base}/{ver}/{f}", part=f"v{ver}_{os.path.basename(f)}")
        state["ver"] = ver


def poll_page_links(tg, state):
    rec, body = tg.fetch(tg.t["url"], part="page")
    if rec["status"] != 200:
        return
    seen = state.setdefault("seen", set())
    for h in re.findall(r'href="([^"]+)"', body.decode(errors="replace")):
        u = urljoin(tg.t["url"], h.replace("&amp;", "&"))
        if u not in seen and re.search(tg.t["link_regex"], u, re.I):
            seen.add(u)
            tg.fetch(u, part=os.path.basename(u.split("?")[0])[-80:])


KINDS = {"url": poll_url, "clarity": poll_clarity, "page_links": poll_page_links}


def run_target(t, stop):
    tg, state, poll = Target(t), {}, KINDS[t.get("kind", "url")]
    tg.log(dict(recv_utc=utc_stamp(time.time_ns()), event="start", kind=t.get("kind", "url")))
    while not stop.is_set():
        start = time.time()
        try:
            poll(tg, state)
        except Exception as e:   # network errors, timeouts: log and keep going
            tg.log(dict(recv_utc=utc_stamp(time.time_ns()), url=t.get("url") or t.get("base"), error=repr(e)))
            print(f"{utc_stamp(time.time_ns())} {t['name']:14s} ERROR {e!r}", flush=True)
        stop.wait(max(0.0, t.get("interval", 30) - (time.time() - start)))


def main(names):
    targets = [t for t in CFG["scrape"]["targets"] if not names or t["name"] in names]
    stop = threading.Event()
    for t in targets:
        threading.Thread(target=run_target, args=(t, stop), daemon=True).start()
    print(f"capturing {[t['name'] for t in targets]} -> {ROOT}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main(sys.argv[1:])
