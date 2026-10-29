"""Stage 1: fit county expectations from real 2018-2024 results -> data/fitted/{tx,ga}_counties.csv.

Same quantities as simulation/model.py:fit_from_history, estimated from official county results:
  lean     : county two-party D share of the EARLY vote minus the statewide early share (weighted mean over years)
  ed_off   : Election Day minus early D share, by county type (ED-vote weighted, mean over years)
  ef       : early share of the county's two-party vote (mean over years)
plus the 2026 early-vote turnout known before election night (early_known) and the implied ED vote.

"Early" = everything counties report in the first drop: TX early in-person + mail (the SOS "EV" column);
GA advance in-person + absentee by mail. Provisional ballots are ignored.

Sources (see data/SOURCES.md):
  TX 2020/2022/2024: Texas SOS election-night JSON (County.json), per candidate total V and early EV
  GA 2018-2024:      Georgia SOS Enhanced Voting exports, per county by vote method
  TX 2026 early vote: Texas SOS early-voting turnout API (cumulative in-person + mail by county)
  Types:             USDA ERS Rural-Urban Continuum Codes 2023, Census county population estimates (2024)

Usage: python fit_counties.py            (writes data/fitted/*.csv and prints diagnostics)
"""
import base64, glob, json, os, re
import numpy as np
import pandas as pd
import yaml
from parse_results import county_key

CFG = yaml.safe_load(open("config.yaml"))
RAW, OUT = CFG["paths"]["raw"], CFG["paths"]["fitted"]
TYPE_NAMES = {"TX": ["urban core", "suburban", "border/Hispanic", "rural"],
              "GA": ["urban core", "metro suburbs", "Black Belt rural", "white rural"]}


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().upper()


# ---------------------------------------------------------------- loaders: one row per county-year
def load_tx(year, race):
    c = json.load(open(f"{RAW}/tx_sos/{year}_County.json"))
    rows = []
    for fips, cty in c.items():
        r = next((r for r in cty["Races"].values() if _norm(r["ON"]) == race), None)
        if r is None:
            continue
        v = {p: [0, 0] for p in ("DEM", "REP")}
        for cand in r["C"].values():
            if cand["P"] in v:
                v[cand["P"]][0] += cand["V"]; v[cand["P"]][1] += cand["EV"]
        (dt, de), (rt, re_) = v["DEM"], v["REP"]
        rows.append(dict(fips=fips, county=cty["N"].title(), year=year, d_early=de, r_early=re_,
                         d_ed=dt - de, r_ed=rt - re_))
    return pd.DataFrame(rows)


GA_GROUP = {"absentee by mail": "early", "advance in person": "early", "advanced voting": "early",
            "advance voting": "early", "election day": "ed", "provisional": None}


def load_ga(year, race):
    d = json.load(open(f"{RAW}/ga_sos/export-{year}NovGen.json"))
    rows = []
    for cty in d["localResults"]:
        b = next((b for b in cty["ballotItems"] if _norm(b["name"]) == race), None)
        if b is None:
            continue
        acc = {"d_early": 0, "r_early": 0, "d_ed": 0, "r_ed": 0}
        for o in b["ballotOptions"]:
            party = (o.get("politicalParty") or "") + " " + o["name"]
            p = "d" if re.search(r"\b(dem|democrat)", party, re.I) else "r" if re.search(r"\b(rep|republican)", party, re.I) else None
            if p is None:
                continue
            for g in o["groupResults"]:
                kind = GA_GROUP[re.sub(r"\s+votes$", "", g["groupName"].strip().lower())]
                if kind:
                    acc[f"{p}_{kind}"] += g["voteCount"]
        rows.append(dict(county=cty["name"].replace(" County", ""), year=year, **acc))
    return pd.DataFrame(rows)


def tx_early_2026():
    """Latest cumulative early vote (in-person + mail) per county from the SOS turnout API files."""
    files = sorted(glob.glob(f"{RAW}/tx_sos_ev2026/ev_{CFG['tx']['ev_election_id']}_*.json"))
    best = None
    for f in files:   # take the file with the largest statewide cumulative total (the latest complete day)
        j = json.loads(base64.b64decode(json.load(open(f))["upload"]))
        t = pd.DataFrame(j["turnout_by_county"])
        t["early_ballots"] = t.total_in_person_votes_for_election + t.total_mail_votes_for_election
        if best is None or t.early_ballots.sum() > best[1].early_ballots.sum():
            best = (os.path.basename(f), t)
    print(f"TX 2026 early vote from {best[0]}: {best[1].early_ballots.sum():,} ballots")
    return best[1].assign(county=best[1].name.str.title())[["county", "early_ballots", "registered_voters"]], best[0]


def ga_early_2026():
    """GA SOS absentee/early voter file must be downloaded by hand (the site blocks scripts); see data/SOURCES.md.
    Expected: data/raw/ga_sos_ev2026/early_by_county.csv with columns county,early_ballots."""
    f = f"{RAW}/ga_sos_ev2026/early_by_county.csv"
    if not os.path.exists(f):
        print("GA 2026 early vote: MISSING (", f, ") -> early_known left blank")
        return None, None
    return pd.read_csv(f), os.path.basename(f)


# ---------------------------------------------------------------- county types
def demographics(state_fips):
    p = pd.read_csv(f"{RAW}/census/cc-est2024-alldata-{state_fips}.csv", encoding="latin-1")
    p = p[(p.YEAR == p.YEAR.max()) & (p.AGEGRP == 0)]
    out = pd.DataFrame(dict(fips=[f"{state_fips}{c:03d}" for c in p.COUNTY],
                            county=p.CTYNAME.str.replace(" County", "", regex=False).values,
                            pop=p.TOT_POP.values,
                            hisp_share=((p.H_MALE + p.H_FEMALE) / p.TOT_POP).values,
                            black_share=((p.BA_MALE + p.BA_FEMALE) / p.TOT_POP).values))
    r = pd.read_csv(f"{RAW}/census/Ruralurbancontinuumcodes2023.csv", dtype={"FIPS": str}, encoding="latin-1")
    r = r[r.Attribute == "RUCC_2023"][["FIPS", "Value"]].rename(columns={"FIPS": "fips", "Value": "rucc"})
    out = out.merge(r, on="fips", how="left")
    out["rucc"] = pd.to_numeric(out.rucc)
    return out


def assign_types(state, demo):
    t = CFG[state.lower()]["types"]
    core = set(t["urban_core"])
    def f(row):
        if row.county in core:
            return 0
        if state == "TX":
            if row.hisp_share >= t["hispanic_min"]:
                return 2
            return 1 if row.rucc <= t["suburban_max_rucc"] else 3
        if row.rucc <= t["suburban_max_rucc"]:
            return 1
        return 2 if row.black_share >= t["black_belt_min"] else 3
    return demo.apply(f, axis=1)


# ---------------------------------------------------------------- fit
def fit(state):
    sc = CFG[state.lower()]
    hist = pd.concat([(load_tx if state == "TX" else load_ga)(y, _norm(race)) for y, race in sc["history"].items()])
    demo = demographics(sc["state_fips"])
    demo["ctype"] = assign_types(state, demo)
    key = "fips" if state == "TX" else "county"
    hist = hist.drop(columns=["county"] if state == "TX" else []).merge(demo, on=key, how="left")
    assert hist.ctype.notna().all(), hist[hist.ctype.isna()].county.unique()
    hist["early_v"] = hist.d_early + hist.r_early; hist["ed_v"] = hist.d_ed + hist.r_ed
    hist["early_sh"] = hist.d_early / hist.early_v; hist["ed_sh"] = hist.d_ed / hist.ed_v.where(hist.ed_v > 0)
    w_year = pd.Series(sc["year_weights"]).rename(lambda y: int(y))
    yrs = sorted(hist.year.unique())

    # per-year statewide early share and county lean (relative to it)
    st = hist.groupby("year").apply(lambda g: g.d_early.sum() / g.early_v.sum())
    hist["lean_y"] = hist.early_sh - hist.year.map(st)
    hist["gap_y"] = hist.ed_sh - hist.early_sh
    hist["ef_y"] = hist.early_v / (hist.early_v + hist.ed_v)
    hist["w"] = hist.year.map(w_year)

    def wmean(g, col):
        ok = g[col].notna()
        return np.average(g.loc[ok, col], weights=g.loc[ok, "w"]) if ok.any() else np.nan
    cty = hist.groupby(key).apply(lambda g: pd.Series(dict(lean=wmean(g, "lean_y"), ef=wmean(g, "ef_y"),
                                                             n_years=g.lean_y.notna().sum()))).reset_index()
    # type-level ED gap: ED-vote weighted within year, then year-weighted across years (as model.py)
    gap_ty = hist[hist.ed_v > 0].groupby(["year", "ctype"]).apply(lambda g: np.average(g.gap_y, weights=g.ed_v)).unstack()
    ed_off = (gap_ty.mul(w_year.reindex(gap_ty.index), axis=0).sum() / w_year.reindex(gap_ty.index).sum())

    out = demo.merge(cty, on=key, how="inner")
    out["state"] = state; out["ctype_name"] = out.ctype.map(dict(enumerate(TYPE_NAMES[state])))
    out["ed_off_type"] = out.ctype.map(ed_off)
    for y in yrs:   # keep the raw per-year inputs next to the fitted values, for audit
        h = hist[hist.year == y].set_index(key)
        out[f"early_sh_{y}"] = out[key].map(h.early_sh).round(4); out[f"ed_sh_{y}"] = out[key].map(h.ed_sh).round(4)
        out[f"twoparty_{y}"] = out[key].map(h.early_v + h.ed_v)

    ev, ev_file = tx_early_2026() if state == "TX" else ga_early_2026()
    if ev is not None:
        out = out.merge(ev.assign(ck=ev.county.map(county_key))[["ck", "early_ballots"]],
                        left_on=out.county.map(county_key), right_on="ck", how="left").drop(columns=["ck", "key_0"], errors="ignore")
        assert out.early_ballots.notna().all(), out[out.early_ballots.isna()].county.tolist()
        out["early_known"] = (out.early_ballots * sc["twoparty_per_ballot"]).round()
        out["ed_exp"] = (out.early_known * (1 - out.ef) / out.ef).round()
    else:
        out["early_ballots"] = np.nan; out["early_known"] = np.nan; out["ed_exp"] = np.nan

    cols = ["state", "fips", "county", "ctype", "ctype_name", "lean", "ed_off_type", "ef", "n_years",
            "early_ballots", "early_known", "ed_exp", "pop", "hisp_share", "black_share", "rucc"] + \
           [c for c in out.columns if re.match(r"(early_sh|ed_sh|twoparty)_\d{4}", c)]
    out = out[cols].sort_values("pop", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    out.round(5).to_csv(f"{OUT}/{state.lower()}_counties.csv", index=False)
    types = pd.DataFrame(dict(ctype=range(4), name=TYPE_NAMES[state], ed_off=ed_off.values,
                              **{f"gap_{y}": gap_ty.loc[y].values for y in gap_ty.index},
                              n_counties=out.groupby("ctype").size().reindex(range(4)).values))
    types.round(5).to_csv(f"{OUT}/{state.lower()}_types.csv", index=False)

    print(f"\n== {state}: {len(out)} counties, years {yrs}, early-vote file: {ev_file}")
    print("statewide early D share by year:", st.round(4).to_dict())
    print("ED - early gap by type and year (two-party D share):\n", types.round(4).to_string(index=False))
    stg = hist.groupby("year").apply(lambda g: g.d_ed.sum() / g.ed_v.sum() - g.d_early.sum() / g.early_v.sum())
    print("statewide ED - early gap by year:", stg.round(4).to_dict())
    return out, types


if __name__ == "__main__":
    for s in ("TX", "GA"):
        fit(s)
