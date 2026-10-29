"""Parsers vs. saved real pages (tests/samples/). Run before and on election night: pytest -q tests/
If a state changes its page format, these fail fast instead of the nowcast silently reading zeros."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from parse_results import diff, parse_file

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def rows(snap, county):
    return {r["method"]: r for r in snap["rows"] if r["county"] == county}


def test_tx_2026_general_page_has_senate_race_with_zero_counts():
    s = parse_file(f"{S}/tx2026_general_20261029_prepoll.html", "tx_statewide_html")
    assert s["statewide"] == dict(dem=0, rep=0, other=0, reporting=0.0)


def test_tx_2026_statewide_page_with_counts():
    # 2026 GOP primary night snapshot (Internet Archive): same page template, no party column
    s = parse_file(f"{S}/tx2026_primary_20260304_wayback.html", "tx_statewide_html")
    assert s["statewide"]["reporting"] == pytest.approx(0.96)
    assert s["statewide"]["other"] == 864794 + 849038 + 278020 + 24766 + 23256 + 15091 + 8874 + 8522


def test_tx_statewide_missing_race_raises():
    with pytest.raises(ValueError):
        parse_file(f"{S}/tx2026_general_20261029_prepoll.html", "tx_statewide_html", "NO SUCH RACE")


def test_tx_county_json_early_and_election_day():
    s = parse_file(f"{S}/tx2024_County_sample.json", "tx_county_json")
    h = rows(s, "Harris")
    assert (h["early"]["dem"], h["early"]["rep"]) == (665254, 542530)
    assert (h["ed"]["dem"], h["ed"]["rep"]) == (841784 - 665254, 666027 - 542530)


def test_ga_export_2024_methods_and_reporting():
    s = parse_file(f"{S}/ga2024_export_sample.json", "ga_export_json", "President of the US")
    f = rows(s, "Fulton")
    assert set(f) == {"early", "ed", "prov"}
    assert (f["early"]["dem"], f["ed"]["dem"]) == (319790, 64475)
    assert f["ed"]["reporting"] == 1.0


def test_ga_export_2022_group_names():
    s = parse_file(f"{S}/ga2022_export_sample.json", "ga_export_json", "Governor")
    f = rows(s, "Fulton")
    assert (f["early"]["dem"], f["early"]["rep"], f["ed"]["dem"]) == (217724, 86686, 70795)


def test_diff_batches_and_anomalies():
    a = dict(recv_utc="t0", statewide=dict(dem=10, rep=10, other=0), rows=[
        dict(county="X", method="early", dem=10, rep=10, other=0, reporting=None)])
    b = dict(recv_utc="t1", statewide=dict(dem=40, rep=20, other=0), rows=[
        dict(county="X", method="early", dem=10, rep=10, other=0, reporting=None),
        dict(county="X", method="ed", dem=30, rep=10, other=0, reporting=0.5)])
    batches, anomalies = diff(a, b)
    assert anomalies == [] and len(batches) == 1
    assert batches[0]["votes"] == 40 and batches[0]["share"] == pytest.approx(0.75)
    c = dict(recv_utc="t2", statewide=dict(dem=35, rep=20, other=0), rows=[
        dict(county="X", method="ed", dem=25, rep=10, other=0, reporting=0.5)])
    _, anomalies = diff(b, c)
    assert {x["county"] for x in anomalies} == {"X", "STATEWIDE"}
