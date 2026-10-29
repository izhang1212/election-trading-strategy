"""Parse raw captured results files into snapshots, and difference snapshots into batches.

One code path for live trading (run_live.py calls parse_file on each new file) and for the post-election
re-parse (python parse_results.py ...). Parsers never touch the network.

Snapshot = dict(source, recv_utc, file, statewide=dict(dem, rep, other, reporting), rows=[...])
  row    = dict(county, method, dem, rep, other, reporting)        cumulative counts
  method : "early" (TX early in-person + mail / GA advance + mail), "ed" (Election Day), "prov", "total"
Batch    = dict(recv_utc, county, method, votes, share, dem, rep, cum_votes, reporting)
  votes/share are the two-party increment since the previous snapshot of the same (county, method).

Usage:
  python parse_results.py <raw target dir> <parser> [race]   -> data/live/parsed/<target>_{snapshots,batches}.jsonl
"""
import html, json, os, re, sys

PARSERS = {}


def parser(name):
    def deco(f):
        PARSERS[name] = f; return f
    return deco


def _norm(s):
    return re.sub(r"\s+", " ", html.unescape(s)).strip().upper()


def county_key(name):
    """Join key across sources: 'McLennan' / 'MCLENNAN' / 'Mclennan County' / 'La Salle' / 'LASALLE' -> 'MCLENNAN', 'LASALLE'."""
    return re.sub(r"[^A-Z]", "", re.sub(r"\s+COUNTY$", "", name.strip().upper()))


def _party(text):
    t = text.upper()
    if re.search(r"\b(DEM|DEMOCRAT|DEMOCRATIC)\b", t):
        return "dem"
    if re.search(r"\b(REP|REPUBLICAN)\b", t):
        return "rep"
    return "other"


def _int(s):
    s = re.sub(r"[^\d]", "", s or "")
    return int(s) if s else 0


# ---------------------------------------------------------------- Texas
@parser("tx_statewide_html")
def tx_statewide_html(body, race="U. S. SENATOR"):
    """2026 SOS page (electionresults.sos.state.tx.us/results.html): statewide totals only, per race:
    <div class="election">RACE</div><div class="polling">Polling Locations Reporting: N%</div><table>...</table>"""
    s = body.decode("utf-8", errors="replace")
    blocks = re.split(r'<div class="election">', s)[1:]
    for b in blocks:
        name = _norm(re.sub(r"<[^>]+>", "", b.split("</div>", 1)[0]))
        if name != _norm(race):
            continue
        m = re.search(r"Reporting:\s*([\d.]+)\s*%", b)
        tot = dict(dem=0, rep=0, other=0, reporting=float(m.group(1)) / 100 if m else None)
        table = b.split("</table>", 1)[0]
        for tr in re.findall(r"<tr>(.*?)</tr>", table, flags=re.S):
            party = re.search(r'class="party">([^<]*)<', tr)
            cand = re.search(r'class="candidate">([^<]*)<', tr)
            votes = re.search(r'class="ballot"\s*>([^<]*)<', tr)
            if not cand:
                continue
            tot[_party(party.group(1) if party else "")] += _int(votes.group(1) if votes else "")
        return dict(statewide=tot, rows=[])
    raise ValueError(f"race {race!r} not found on TX statewide page")


@parser("tx_county_json")
def tx_county_json(body, race="U. S. SENATOR"):
    """2020-2024 SOS election-night JSON (results.texas-election.com .../County.json): per county, per candidate
    total V and early EV (early in-person + mail); Election Day = V - EV. PR/TPR = precincts reporting/total."""
    c = json.loads(body)
    rows, sw = [], dict(dem=0, rep=0, other=0)
    for cty in c.values():
        r = next((r for r in cty["Races"].values() if _norm(r["ON"]) == _norm(race)), None)
        if r is None:
            continue
        e, t = dict(dem=0, rep=0, other=0), dict(dem=0, rep=0, other=0)
        for cand in r["C"].values():
            p = _party(cand["P"]); t[p] += cand["V"]; e[p] += cand["EV"]
        rep = r["PR"] / r["TPR"] if r.get("TPR") else None
        rows.append(dict(county=cty["N"].title(), method="early", reporting=None, **e))
        rows.append(dict(county=cty["N"].title(), method="ed", reporting=rep, **{k: t[k] - e[k] for k in t}))
        for k in sw:
            sw[k] += t[k]
    return dict(statewide=dict(sw, reporting=None), rows=rows)


# ---------------------------------------------------------------- Georgia
GA_METHOD = {"absentee by mail": "early", "advance in person": "early", "advanced voting": "early",
             "advance voting": "early", "election day": "ed", "provisional": "prov"}


def _ga_method(group):
    return GA_METHOD[re.sub(r"\s+votes$", "", group.strip().lower())]


@parser("ga_export_json")
def ga_export_json(body, race="GOVERNOR"):
    """GA SOS Enhanced Voting export (results.sos.ga.gov/cdn/results/Georgia/export-<slug>.json).
    Per county, per option: groupResults by vote method. ED reporting fraction = share of the county's
    non-virtual precincts marked "Fully Reported" (2024+); older exports without that field fall back to
    the share of precincts with any Election Day votes in this race."""
    d = json.loads(body)
    rows, sw = [], dict(dem=0, rep=0, other=0)
    for cty in d["localResults"]:
        b = next((b for b in cty["ballotItems"] if _norm(b["name"]) == _norm(race)), None)
        if b is None:
            continue
        acc = {m: dict(dem=0, rep=0, other=0) for m in ("early", "ed", "prov")}
        prec = {}   # precinct id -> [ED votes, fully reported?]
        for o in b["ballotOptions"]:
            p = _party((o.get("politicalParty") or "") + " " + o["name"])
            for g in o["groupResults"]:
                acc[_ga_method(g["groupName"])][p] += g["voteCount"] or 0
            for pr in o.get("precinctResults") or []:
                if pr.get("isVirtual"):
                    continue
                v = sum(g["voteCount"] or 0 for g in pr.get("groupResults", []) if _ga_method(g["groupName"]) == "ed")
                st = pr.get("reportingStatus")
                e = prec.setdefault(pr.get("id") or pr.get("name"), [0, None])
                e[0] += v
                e[1] = None if st is None else st.strip().lower() == "fully reported"
        n = len(prec)
        if n and all(e[1] is not None for e in prec.values()):
            ed_rep = sum(e[1] for e in prec.values()) / n
        else:
            ed_rep = sum(e[0] > 0 for e in prec.values()) / n if n else None
        name = cty["name"].replace(" County", "")
        for m, v in acc.items():
            rows.append(dict(county=name, method=m, reporting=ed_rep if m == "ed" else None, **v))
            for k in sw:
                sw[k] += v[k]
    return dict(statewide=dict(sw, reporting=None), rows=rows)


# ---------------------------------------------------------------- files, snapshots, batches
def parse_file(path, parser_name, race=None, recv_utc=None):
    body = open(path, "rb").read()
    if body[:2] == b"\x1f\x8b":            # gzip (e.g. Internet Archive id_ captures)
        import gzip
        body = gzip.decompress(body)
    snap = PARSERS[parser_name](body, race) if race else PARSERS[parser_name](body)
    snap.update(source=parser_name, file=os.path.basename(path),
                recv_utc=recv_utc or os.path.basename(path).split("_")[0])
    return snap


def diff(prev, cur):
    """Batches = two-party increments per (county, method) between consecutive snapshots.
    Returns (batches, anomalies). A decrease in any count is an anomaly (kill-switch input), not a batch."""
    old = {(r["county"], r["method"]): r for r in (prev or {}).get("rows", [])}
    batches, anomalies = [], []
    for r in cur["rows"]:
        o = old.get((r["county"], r["method"]), dict(dem=0, rep=0, other=0))
        dd, dr = r["dem"] - o["dem"], r["rep"] - o["rep"]
        if dd < 0 or dr < 0:
            anomalies.append(dict(county=r["county"], method=r["method"], d_dem=dd, d_rep=dr, recv_utc=cur["recv_utc"]))
            continue
        if dd + dr == 0:
            continue
        batches.append(dict(recv_utc=cur["recv_utc"], county=r["county"], method=r["method"], votes=dd + dr,
                            share=dd / (dd + dr), dem=dd, rep=dr, cum_votes=r["dem"] + r["rep"],
                            reporting=r.get("reporting")))
    ps, cs = (prev or {}).get("statewide"), cur["statewide"]
    if ps and (cs["dem"] < ps["dem"] or cs["rep"] < ps["rep"]):
        anomalies.append(dict(county="STATEWIDE", method="total", d_dem=cs["dem"] - ps["dem"],
                              d_rep=cs["rep"] - ps["rep"], recv_utc=cur["recv_utc"]))
    return batches, anomalies


def raw_files(target_dir):
    """Changed files of a target in receive order, using index.jsonl when present."""
    idx = os.path.join(target_dir, "index.jsonl")
    if os.path.exists(idx):
        out = []
        for line in open(idx):
            rec = json.loads(line)
            if rec.get("file") and rec.get("status") == 200:
                out.append((rec["recv_utc"], os.path.join(target_dir, rec["file"])))
        return out
    return sorted((f.split("_")[0], os.path.join(target_dir, f)) for f in os.listdir(target_dir)
                  if not f.endswith((".jsonl", ".part")))


def reparse(target_dir, parser_name, race=None, out_dir="data/live/parsed"):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(target_dir))
    prev, n_b, n_err = None, 0, 0
    with open(f"{out_dir}/{name}_snapshots.jsonl", "w") as fs, open(f"{out_dir}/{name}_batches.jsonl", "w") as fb:
        for recv, path in raw_files(target_dir):
            try:
                snap = parse_file(path, parser_name, race, recv)
            except Exception as e:
                n_err += 1; print("parse error", path, repr(e)); continue
            batches, anomalies = diff(prev, snap)
            fs.write(json.dumps(dict(snap, anomalies=anomalies)) + "\n")
            for b in batches:
                fb.write(json.dumps(b) + "\n")
            n_b += len(batches); prev = snap
    print(f"{name}: {n_b} batches, {n_err} parse errors -> {out_dir}")


if __name__ == "__main__":
    reparse(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
