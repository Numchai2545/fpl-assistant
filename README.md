# FPL Assistant — Gnum United (6014213)

A decision-support report for the Fantasy Premier League deadline: the weakest
slot in your squad, three or four affordable replacements, captain choices,
starting XI and bench order. It never logs in or changes the FPL team.

Everything runs on free, public data from the official FPL API. No key, no
scraping, no cost.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements-lock.txt  # exact versions verified by the test suite
set PYTHONPATH=src               # Windows;  export PYTHONPATH=src  elsewhere
set PYTHONIOENCODING=utf-8       # Windows only: a cp1252 console cannot
                                 # print Thai without this

python -m fplbot build
```

That writes `docs/index.html` (the dashboard, in Thai) and `docs/deadlines.ics`
(a calendar reminder 24 hours before each deadline). On Windows, double-click
`เปิดเว็บ FPL.bat` to open the web dashboard, then use its update button.

Run the tests with `python -m pytest tests/ -q` — the suite takes about five
seconds, no network needed.

**No network to the FPL API?** Seed a frozen dataset first and work offline:

```bash
python tests/seed_offline_fixture.py
python -m fplbot build --offline
```

The offline seed pulls a season mirror from GitHub. It is good enough to exercise
the whole pipeline, but its player stats lag the live API by a few gameweeks —
use it for development, never for an actual transfer decision.

---

## Commands

| Command | What it does |
|---|---|
| `python -m fplbot build` | Fetch, model, optimise, write the dashboard and calendar |
| `python -m fplbot build --offline` | Rebuild from the cached snapshot, no API calls |
| `python -m fplbot check` | Print the next deadline and how far away it is |
| `python -m fplbot serve` | Open the local web dashboard with a refresh button |
| `python -m fplbot backtest` | Compare frozen pre-deadline projections with finished matches |
| `python -m pytest tests/ -q` | Run the test suite |

Add `-v` before the subcommand for debug logging: `python -m fplbot -v build`.

---

## How it works

```
fetch.py         official FPL API  ->  data/snapshots/<date>/*.json
features.py      raw JSON          ->  player rates, team strength, fixture schedule
model.py         rates + fixtures  ->  expected points per player per gameweek
optimize.py      EP + prices       ->  cumulative transfer priorities + current XI
report.py        the analysis      ->  docs/index.html + docs/summary.json
calendar_feed.py the fixture list  ->  docs/deadlines.ics
```

### The expected points model

```
EP = P(plays) x [ minutes + goals + assists + clean sheet
                  + defensive contribution + bonus + saves - cards - goals conceded ]
```

Each term comes from the player's own per-90 rates, **shrunk toward a positional
prior** so a two-game purple patch does not outrank a season of work:

```
rate = w * observed + (1 - w) * prior        w = minutes / (minutes + 360)
```

Then adjusted for the actual opponent and venue of that gameweek's fixture.
FPL's recent `form` value contributes 15% after availability adjustment, so it
can break close calls and change the best formation without dominating the
underlying statistics or five-fixture schedule.

| Term | Built from |
|---|---|
| Playing time | Recent six-match starts/minutes for a bounded shortlist, blended with the season prior and availability |
| Goals | 70% `expected_goals_per_90` + 30% actual, x opponent defence, x venue |
| Assists | 75% `expected_assists_per_90` + 25% actual, +6% for the corner taker |
| Clean sheet | Poisson on goals conceded, from team defence vs opponent attack |
| Defensive contribution | Poisson on `defensive_contribution_per_90` vs the 10 / 12 threshold |
| Bonus | `bps` per 90, mapped to an expected bonus (crude in v0.1 — see roadmap) |
| Saves | `saves_per_90` scaled by how much shooting the opponent generates |

The `#1` penalty taker gets an 8% uplift on goals: penalties keep arriving whether
or not any fell inside the sample so far.

### The transfer analysis

Every legal out → in pair is compared over each player's next five actual
fixtures. The report selects the squad slot with the largest upgrade and shows
the best three or four same-position replacements that fit the bank and
three-per-club rule. Ownership is displayed but never changes the ranking.

The dashboard shows one cumulative priority order. Each row chooses the largest
remaining legal same-position upgrade, using the bank and club slots left after
previous rows. Earlier pairs never change when the manager chooses more moves.
Every prefix is affordable; bought players cannot be sold again in the chain.
This greedy order is not claimed to be a global multi-transfer optimum.

A move needs complete five-fixture coverage and at least +3.0 net EP to be
endorsed. Extra transfers display their four-point cost; the emergency policy
still gates paid recommendations. The manager decides how many rows to use.
Skipping rows or choosing other replacements requires a fresh budget check.
Ownership remains display-only. CBC separately selects the XI, bench, captain
and vice-captain from the current owned squad, without assuming transfers.

---

## Configuration

Everything personal lives in `config.yaml`. The fields worth knowing:

| Key | Meaning |
|---|---|
| `entry.team_id` | Your FPL entry id. Currently `6014213`. |
| `planning.outlook_matches` | Actual upcoming fixtures used to judge a purchase. Currently 5. |
| `planning.min_transfer_gain` | Five-match EP improvement required for a recommendation. Currently 3.0. |
| `planning.candidate_count` | Replacement alternatives shown for the selected weak link. Currently 4. |
| `planning.comparison_count` | Affordable peers shown when a squad player is clicked. Currently 8. |
| `planning.min_candidate_start` | Hides replacement candidates with a very low projected chance of starting. Currently 0.50. |
| `strategy.mode` | Legacy compatibility only; ownership never changes EP in any mode. |
| `planning.priority_count` | Maximum cumulative priority rows displayed. Currently 5. |
| `planning.build_time_budget` | Entire local web build budget including subprocesses, in seconds. Currently 300. |
| `planning.stale_after_hours` | Age of the oldest decision input that triggers a warning. Currently 24. |
| `model.cameo_minutes` | Conditional cameo duration used in the defensive-contribution mixture. Currently 14 minutes. |
| `model.recent_form_weight` | Weight given to current FPL form when projecting points. Currently 0.15. |
| `model.auto_sub_slot_probability` | Estimated rescue value of GK and outfield bench slots 1–3. |
| `planning.minutes_shortlist` | Maximum plausible targets whose match histories are fetched, in addition to the current squad. |
| `chips.*` | Mark chips as used; the dashboard only flags structural Blank/Double opportunities. |
| `notify.remind_hours_before` | Calendar alarm timing. Currently 24 hours. |

---

## Deadline reminder

`docs/deadlines.ics` carries every remaining deadline of the season with alarms
24 hours beforehand. Import it once and your phone fires the reminder locally.
The reminder tells you to double-click `เปิดเว็บ FPL.bat`, then press the update
button to fetch current data and rebuild the analysis.

* **iPhone** — open the dashboard, tap *เพิ่มลงปฏิทิน*, then *Add All*.
* **Android** — import `docs/deadlines.ics` into Google Calendar.

GitHub Pages also supports a free, queued mobile refresh request. It opens a
GitHub issue that only the repository owner can use to trigger a fresh build;
no token is stored in the PWA and no FPL account action is automated.


A build keeps the previous report when the public transfer history, permanent
squad/bank, or holding baseline cannot be verified. After a Free Hit deadline,
the permanent squad and bank are recovered from the preceding permanent picks;
temporary Free Hit purchases are excluded from selling-price reconstruction.
If public history identifies an active Free Hit for the target GW, permanent
transfer advice is withheld. Availability of pre-deadline chip state still
depends on what the public API exposes.

---

## Calibration and remaining roadmap

Version 0.4 uses conditional start/cameo probabilities for defensive contribution,
preserves blank GWs in the planning timeline, and resets a player's selling-cost
basis after a repurchase under the fixed-price assumption. It distinguishes a
verified feasible CBC incumbent from a proven optimum; neither proves the model
predicts real points accurately. Autosubs and vice-captain fallback remain
approximations, and the model is still uncalibrated.

Pre-deadline projections include the model version, configuration hash, season,
and source timestamps. The Pages workflow validates and commits these small
JSON records alongside the report; raw API snapshots stay uncommitted.
Backtests separate results by model/configuration, position and availability,
and filter the outcome season. Samples are a selected shortlist, not an
unbiased evaluation of the whole league.

Run `python -m fplbot backtest` after a projected gameweek finishes. It compares
model MAE and bias with FPL's captured `ep_next`; results are written to
`data/backtests/latest.json`. The next priority is accumulating enough finished
gameweeks to calibrate the bonus mapping, xG/xA blend, shrinkage and recent-form
weight. Chip estimates and price alerts remain decision support—not automatic
chip or early-transfer instructions—until that evidence exists.

---

## Project layout

```
config.yaml                 everything personal
requirements.txt
requirements-lock.txt        reproducible versions used by CI
เปิดเว็บ FPL.bat            user-facing shortcut to open the website
open_fpl_web.bat            ASCII launcher alias
run_weekly.bat              silent local web launcher
start_fpl_server.py         background server entry point used at Windows sign-in
.github/workflows/          Pages builds, owner issue requests, twice-daily fallback, manual verification
src/fplbot/
    fetch.py                API client + snapshot cache
    features.py             player rates, team strength, fixture schedule
    model.py                expected points
    optimize.py             ranked transfer advice + legal lineup solver
    report.py               dashboard rendering
    copy_th.py              shared Thai decision and notification copy
    cli.py                  command line
web/                        HTML template, PWA manifest, service worker, icons
tests/seed_offline_fixture.py   build a frozen dataset for offline work
data/snapshots/             one folder per run — this is your growing history
data/projections/           timestamped, pre-deadline EP records for backtesting
data/backtests/latest.json  latest model-vs-FPL evaluation
docs/                       generated local dashboard and calendar
```

`data/snapshots/` remains gitignored because raw responses are large. Compact
files in `data/projections/` are the durable backtest input and contain only the
current squad plus the transfer shortlist, each stamped with model version and
deadline so later information cannot leak into the evaluation.

---

## A caveat worth repeating

Expected points are a model's estimate, not a forecast of what will happen.
A 6.2 EP captain scoring two is normal, not a bug. The value here is in making the
*average* decision better across a season and in never missing a deadline again —
not in getting any single week right.
