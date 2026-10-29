"""Write a synthetic election night in a state's CURRENT results format, for run_live.py --replay.

Takes a real final export (e.g. the GA SOS 2022 export) and emits a sequence of partial snapshots: each
county's early vote appears in one drop, then its Election Day vote in 1-4 chunks with precinct statuses,
on a reporting clock shaped like simulation/world.py (early drops in the first hour, ED after a delay).
Counts are scaled copies of the real final counts, so the parser, the differencing and the nowcast see a
realistic night in the real file format. Output: <out>/<target>/<UTC stamp>_<sha8>.json (+ index.jsonl).

Usage: python tests/make_replay.py data/raw/ga_sos/export-2022NovGen.json /tmp/replay --race Governor --seed 1
"""
import argparse, copy, hashlib, json, os
from datetime import datetime, timedelta, timezone
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("export"); ap.add_argument("out")
ap.add_argument("--race", default="Governor"); ap.add_argument("--target", default="ga_export")
ap.add_argument("--start", default="2026-11-04T00:00:00"); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
src = json.load(open(a.export))
src["localResults"] = [dict(c, ballotItems=[b for b in c["ballotItems"] if b["name"] == a.race]) for c in src["localResults"]]
src["results"]["ballotItems"] = [b for b in src["results"]["ballotItems"] if b["name"] == a.race]
size = {c["name"]: sum(o["voteCount"] or 0 for b in c["ballotItems"] for o in b["ballotOptions"]) for c in src["localResults"]}

def ed_group(g):
    return g.lower().startswith("election day")

# reporting schedule (minutes after polls close): early in [5, 60], ED chunks after a 30-90 min delay
sched = {}
for c in src["localResults"]:
    te = rng.uniform(5, 60); k = int(np.clip(round(size[c["name"]] / 60000), 1, 4))
    cum = np.cumsum(rng.dirichlet(np.full(k, 4.0))); t = te + rng.uniform(30, 90) * (1.5 if size[c["name"]] > 150000 else 1)
    chunks = []
    for j in range(k):
        chunks.append((t, 1.0 if j == k - 1 else float(cum[j]))); t += rng.uniform(15, 40)
    sched[c["name"]] = (te, chunks)

def snapshot(minute):
    d = copy.deepcopy(src)
    for c in d["localResults"]:
        te, chunks = sched[c["name"]]
        fe = 1.0 if minute >= te else 0.0
        fd = max([f for t, f in chunks if minute >= t], default=0.0)
        for b in c["ballotItems"]:
            for o in b["ballotOptions"]:
                for g in o["groupResults"]:
                    g["voteCount"] = int(round((g["voteCount"] or 0) * (fd if ed_group(g["groupName"]) else fe)))
                prs = o.get("precinctResults") or []
                for i, pr in enumerate(prs):   # first fd of precincts fully reported, the rest zeroed on ED
                    done = i < round(fd * len(prs))
                    pr["reportingStatus"] = "Fully Reported" if done else "Not Reported"
                    for g in pr.get("groupResults", []):
                        if ed_group(g["groupName"]) and not done:
                            g["voteCount"] = 0
                o["voteCount"] = sum(g["voteCount"] for g in o["groupResults"])
    return d

d0 = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
out = os.path.join(a.out, a.target); os.makedirs(out, exist_ok=True)
last, n = None, 0
with open(os.path.join(out, "index.jsonl"), "w") as idx:
    for minute in np.arange(0, 360, 0.5):
        body = json.dumps(snapshot(minute)).encode(); sha = hashlib.sha256(body).hexdigest()
        if sha == last:
            continue
        stamp = (d0 + timedelta(minutes=float(minute))).strftime("%Y%m%dT%H%M%S.%fZ")
        fn = f"{stamp}_{sha[:8]}.json"; open(os.path.join(out, fn), "wb").write(body)
        idx.write(json.dumps(dict(recv_utc=stamp, status=200, sha256=sha, file=fn, changed=True)) + "\n")
        last = sha; n += 1
print(f"{n} snapshots -> {out}")
