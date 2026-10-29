# Data sources

Every file in `data/raw/` (git-ignored) with its URL, download date and SHA-256 prefix, so each can be re-downloaded and checked. All downloads were made 2026-10-29 unless noted. `fit_counties.py` reads only the files marked **used**.

## What the model needs, and where each piece comes from

| Need | Texas | Georgia |
|---|---|---|
| County two-party results split early vs. Election Day, several cycles | TX SOS election-night JSON, 2020/2022/2024 (**used**) | GA SOS Enhanced Voting exports, 2018/2020/2022/2024 (**used**) |
| 2026 early-vote turnout by county (known before election night) | TX SOS early-voting turnout API (**used**; final pull due Oct 31) | GA SOS absentee/early voter file: **manual download needed** (see below) |
| County types (urban / suburban / Hispanic / Black Belt / rural) | USDA RUCC 2023 + Census 2024 county estimates (**used**) | same (**used**) |
| Live 2026 results feed | `electionresults.sos.state.tx.us/results.html`: **statewide totals only** | `results.sos.ga.gov/cdn/results/Georgia/export-<slug>.json`: county × vote method |
| Market data | Kalshi REST + WebSocket (recorded live) | same |

## Election results (history)

**Texas Secretary of State, election-night results JSON** (official, all 254 counties). Per county and candidate: total votes `V` and early votes `EV` (early in-person + mail); Election Day = `V − EV`.
Base URL `https://results.texas-election.com/static/data/election/<id>/<version>/County.json` (version from `.../<id>/Version.json`).

| File | Election (id / version) | Race used | SHA-256 |
|---|---|---|---|
| `tx_sos/2020_County.json`, `2020_Lookups.json` | 2020 general (44144 / 262) | President | `e8a0c0d0e40c3383`, `bf170815fc840a55` |
| `tx_sos/2022_County.json`, `2022_Lookups.json` | 2022 general (47009 / 242) | Governor | `406c52a5fe506962`, `be88a06e649bba7b` |
| `tx_sos/2024_County.json`, `2024_Lookups.json` | 2024 general (49664 / 1012) | U.S. Senate | `fa818d6a89dd0b6e`, `fabac9d23a76247b` |

Texas 2018 is not on this system. The OpenElections precinct files below have an early/ED split for only part of the state, and some columns appear mislabeled (e.g. Hidalgo 2018 shows 38k early vs. 106k Election Day; Fort Bend 2022 puts early votes under `absentee`). The 2020 file lacks Dallas, Bexar and Travis; the 2022 file lacks Harris, Tarrant and Bexar. So **Texas 2018 is excluded** from the fit. The files are kept for reference only.

| File | URL | SHA-256 |
|---|---|---|
| `openelections_tx/20181106__tx__general__county.csv` | https://github.com/openelections/openelections-data-tx/blob/master/2018/20181106__tx__general__county.csv | `1d407575b7dbf875` |
| `openelections_tx/20181106__tx__general__precinct.csv` | .../2018/20181106__tx__general__precinct.csv | `31c21c864a85fe0c` |
| `openelections_tx/20201103__tx__general__precinct.csv` | .../2020/20201103__tx__general__precinct.csv | `84256f643919e895` |
| `openelections_tx/20221108__tx__general__precinct.csv` | .../2022/20221108__tx__general__precinct.csv | `6038336971f4a8d6` |

**Georgia Secretary of State, Enhanced Voting exports** (official, all 159 counties). Per county and candidate: votes by method (Election Day, Advance/Advanced Voting, Absentee by Mail, Provisional; the labels change by year, normalized in `parse_results.py`).
URL `https://results.sos.ga.gov/cdn/results/Georgia/export-<YYYY>NovGen.json`.

| File | Race used | SHA-256 |
|---|---|---|
| `ga_sos/export-2018NovGen.json` | Governor | `cef9e93fea9fdf61` |
| `ga_sos/export-2020NovGen.json` | President | `e53c47289f0267df` |
| `ga_sos/export-2022NovGen.json` | Governor | `055d1d5302a0a258` |
| `ga_sos/export-2024NovGen.json` | President | `a4c9968087916837` |

## 2026 early-vote turnout

**Texas SOS early-voting turnout API.** This is the app linked as "Early Voting Turnout (current)" on https://www.sos.state.tx.us/elections/historical/index.shtml, at https://goelect.txelections.civixapps.com/ivis-evr-ui/evr.
Election list: `https://goelect.txelections.civixapps.com/api-ivis-system/api/v1/getFile?type=EVR_ELECTION`; 2026 general = election id **53815**.
By county, cumulative: `...getFile?type=EVR_EARLYVOTING&electionId=53815&electionDate=MM/DD/2026`. The response is base64 JSON with `total_in_person_votes_for_election` and `total_mail_votes_for_election` per county.

| Files | Notes |
|---|---|
| `tx_sos_ev2026/EVR_ELECTION.json` | SHA `f854aab6c9cd672d` |
| `tx_sos_ev2026/ev_53815_2026-10-19.json` … `2026-10-29.json` | 11 daily files. Through 10/28: 5,920,327 in-person + 238,373 mail. The 10/29 file has almost no 10/29 votes yet. **Early voting ends Oct 30: re-pull on Oct 31 and re-run `fit_counties.py` before the freeze.** |

**Georgia: manual step.** The SOS absentee/early voter file (https://elections.sos.ga.gov/Elections/voterabsenteefile.do) returns HTTP 403 to scripts, so it was not fetched automatically. Download the 2026 general file in a browser and save county totals of accepted ballots (advance in-person + accepted mail) as `data/raw/ga_sos_ev2026/early_by_county.csv` with columns `county,early_ballots`. The SOS turnout hub (https://sos.ga.gov/page/election-data-hub-unofficial-turnout) is the cross-check. Until this file exists, the Georgia nowcast refuses to run.

## County types

| File | URL | SHA-256 |
|---|---|---|
| `census/Ruralurbancontinuumcodes2023.csv` | https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/53251/Ruralurbancontinuumcodes2023.csv | `ec455ee2a8bc5fc8` |
| `census/cc-est2024-alldata-48.csv` | https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata-48.csv | `9883ee5f5564233e` |
| `census/cc-est2024-alldata-13.csv` | .../cc-est2024-alldata-13.csv | `85a9f2987a97fcfc` |

The ACS API now requires a key, so the population-estimates files (Hispanic and Black shares, July 2024) are used instead. Type rules are in `config.yaml` (`tx.types`, `ga.types`).

## Live feeds (captured on election night by `scrape_results.py`)

| Target | URL | Checked 2026-10-29 |
|---|---|---|
| `tx_statewide` | https://electionresults.sos.state.tx.us/results.html | Static HTML, U.S. Senate statewide totals and "Polling Locations Reporting %". **No county breakdown, no early/ED split.** Same template as the March 2026 primary (Internet Archive capture 20260304165047, saved as `tests/samples/tx2026_primary_20260304_wayback.html`). |
| `tx_harris` | https://appfiles.harrisvotes.com/harrisvotes/prd/Data/1126/resultsbyballot.json (+ `pr.json`) | Harris County results app data (election code 1126 = Nov 3, 2026). Per candidate: `ab` mail, `ev` early, `ed` Election Day, `tv` total; `tp`/`pr` precincts. Already lists the Senate race with zero counts. Found via https://app.harrisvotes.com/electionresults/electionday |
| `tx_tarrant`, `tx_elpaso`, `tx_bexar_clarity`, `tx_denton` | `https://results.enr.clarityelections.com/TX/<County>/<eid>/<ver>/json/en/summary.json` and `.../reports/detailxml.zip` (vote types) | Clarity ENR 2026 general ids: Tarrant 127297 (linked from tarrantcountytx.gov), El Paso 127348 (linked from epcountyvotestx.gov), Bexar 127291 and Denton 127398 (found by the scraper) |
| `tx_dallas`, `tx_travis`, `tx_fortbend`, `tx_williamson`, `tx_hidalgo` | Clarity, id discovered automatically from `elections.json` or the county page | Not yet listed on 10/29 (Dallas's latest is the June 2026 runoff, 126485) |
| `tx_collin`, `tx_bexar` | https://www.collincountytx.gov/elections/election-results, https://elections.bexar.gov/ | Pages captured, plus any linked Nov 3 report files (CSV/PDF) |
| `ga_export` | https://results.sos.ga.gov/cdn/results/Georgia/export-2026NovGen.json | Slug unconfirmed (404 until it goes live; past slugs: `2024NovGen`, `2024MayGenPri`). Confirm on the GA results site on Nov 3. |

## Markets

Kalshi API (`https://api.elections.kalshi.com/trade-api/v2`). Checked 2026-10-29:

| Ticker | Contract | Price 2026-10-29 | Volume (contracts) |
|---|---|---|---|
| `SENATETX-26-D` | Democrat (Talarico) wins TX Senate | 68¢ bid / 69¢ ask | 16.3M |
| `GOVPARTYGA-26-D` | Democrat (Bottoms) wins GA Governor | 61¢ / 62¢ | 1.2M (R side 1.8M) |
| `CONTROLS-2026-D`, `CONTROLH-2026-D` | Senate / House control | 67¢ / 68¢, 91.7¢ / 91.8¢ | |

Series fee: `fee_type=quadratic`, `fee_multiplier=1` (taker fee 0.07·C·P·(1−P)). Price grid `tapered_deci_cent`: 1¢ ticks from 10¢ to 90¢, 0.1¢ below 10¢ and above 90¢. Sizes are fractional (`*_fp`). The TX contract resolves on the party sworn in. It is "eligible for accelerated determination after a consensus of media organizations project the winner."

## Not yet downloaded

- **Historical reporting timestamps** (2020 NYT county time series; 2022/2024 Internet Archive snapshots). These are for the replay and the simulation's reporting clock, not the fit.
- **MIT Election Lab county returns** (Harvard Dataverse doi:10.7910/DVN/VOQCHQ) need a guestbook form (name and email), so they were not fetched. They aren't needed now that the GA SOS exports cover 2018–2024.
