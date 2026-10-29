# Election-night runbook

All times ET. Three processes run in separate terminals and share nothing but files in `data/live/`, so any one can be restarted without touching the others.

## Before the night

| When | Do | Check |
|---|---|---|
| Oct 31 | Re-pull TX early vote (`data/SOURCES.md`); download the GA early-vote file by hand; `python fit_counties.py` | `early_known` filled for every county in both CSVs |
| Nov 1–2 | **Recorder dry run**, 3+ hours: `python record_markets.py`, `python scrape_results.py` | WebSocket connects with the API key; `seq` has no gaps (`grep '"gap"' data/live/kalshi/ws_*.jsonl`); reconnect works after pulling the network; disk files grow; `top.json` updates every second |
| Nov 1–2 | **Replay rehearsal**: `python tests/make_replay.py data/raw/ga_sos/export-2022NovGen.json /tmp/replay` then `python run_live.py --replay /tmp/replay --states GA --mode paper` | Decisions file has fair values; anchor event logged; kill switch trips on a frozen market |
| Nov 2 | `pytest -q tests/` | All pass |
| Nov 2 and Nov 3 at 5 p.m. | Texas county capture: `grep clarity_eid data/live/raw/tx_*/index.jsonl` | Every Clarity county has a 2026 id (found already 10/29: Tarrant 127297, Bexar 127291, Denton 127398, El Paso 127348). For any still missing (Dallas, Travis, Fort Bend, Williamson, Hidalgo), find the id on the county's elections page and set `eid:` in `config.yaml` (a logged deviation after the freeze). |
| Nov 2 | Live only: one 1-contract IOC buy and sell on `SENATETX-26-D` through `send_order` | Fill and position reconcile; record fee charged vs. `strategy.fee` |
| Nov 2, 11:59 p.m. | **Go / no-go for live.** No-go if any check above failed or there is no Texas county feed | Set `mode:` in `config.yaml` |
| Nov 3, before 7 a.m. | Freeze: hashes into PREREGISTRATION.md §8, tag `v1.0-prereg`, release, Zenodo | DOI resolves |

## Election night (Nov 3)

| Time | Event |
|---|---|
| 5:00 p.m. | Start `record_markets.py` and `scrape_results.py`. Confirm the GA 2026 export slug and edit `config.yaml` if needed (a config change after the freeze is a logged deviation). |
| 6:30 p.m. | Start `python run_live.py --mode <record/paper/live>`. |
| 7:00 p.m. | Georgia polls close; absentee ballots are due to be counted within the hour. |
| 8:00 p.m. | Texas polls close (Central). Early votes post first. |
| 9:00 p.m. | El Paso polls close (Mountain). |
| Ongoing | Every 10 min: recorder heartbeats in each terminal; `tail -1 data/live/decisions/TX.jsonl`. |

## Kill switch

`run_live.py` trips it automatically (`kill` field in the decision log). Once tripped:

1. Do not restart `run_live.py` to clear it. The trip is part of the record.
2. Leave the recorders running.
3. Open position: hold unless the reason is clearly our bug and the position is on the wrong side of a called race. Any manual trade goes in `data/live/manual_trades.md` with the time and reason.
4. Write down what happened in `data/live/incident.md` for the post-mortem.

Manual trip (e.g. a feed looks wrong): stop `run_live.py` with Ctrl-C and record the reason in `incident.md`.

## Failure cases

| Symptom | Action |
|---|---|
| `scrape_results.py` prints repeated errors or non-200 for one target | Check the URL in a browser. If the format changed, the parse tests fail; capture continues regardless, so fix the parser and re-parse afterward. |
| Results site blocks the recorder | Do not try to get around it. Lower the poll rate, record what is reachable and note it in `incident.md`. |
| Kalshi WebSocket keeps disconnecting | REST snapshots continue every 30 s and `top.json` falls back to them (`book_source: rest`). |
| Exchange halt or outage | No orders go through. Log the times. |
| Laptop sleep or power loss | Disable sleep and keep it plugged in. All three scripts resume from their files when restarted. |
