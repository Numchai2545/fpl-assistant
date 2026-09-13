# FPL Assistant — Gnum United (6014213)

A weekly report that tells you what to do before the Fantasy Premier League
deadline: who to transfer, who to captain, who to start — and a plan for the
next six gameweeks so this week's move does not paint you into a corner.

Everything runs on free, public data from the official FPL API. No key, no
scraping, no cost.

---

## Quick start

```bash
git clone <your-repo-url> fpl-assistant
cd fpl-assistant

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
set PYTHONPATH=src               # Windows;  export PYTHONPATH=src  elsewhere

python -m fplbot build
```

That writes `docs/index.html`. Open it in a browser.

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
| `python -m fplbot build` | Fetch, model, optimise, write the dashboard |
| `python -m fplbot build --offline` | Rebuild from today's cached snapshot, no API calls |
| `python -m fplbot check` | Print the next deadline and how far away it is |
| `python -m fplbot notify` | Send the Telegram alert, if it is due |
| `python -m fplbot notify --force` | Send it regardless of timing (for testing) |

Add `-v` before the subcommand for debug logging: `python -m fplbot -v build`.

---

## How it works

```
fetch.py      official FPL API  ->  data/snapshots/<date>/*.json
features.py   raw JSON          ->  player rates, team strength, fixture schedule
model.py      rates + fixtures  ->  expected points per player per gameweek
optimize.py   EP matrix         ->  a six-gameweek transfer plan (integer program)
report.py     the plan          ->  docs/index.html + docs/summary.json
notify.py     summary.json      ->  Telegram message
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

| Term | Built from |
|---|---|
| Playing time | `starts` / club matches played, `status`, `chance_of_playing_next_round` |
| Goals | 70% `expected_goals_per_90` + 30% actual, x opponent defence, x venue |
| Assists | 75% `expected_assists_per_90` + 25% actual, +6% for the corner taker |
| Clean sheet | Poisson on goals conceded, from team defence vs opponent attack |
| Defensive contribution | Poisson on `defensive_contribution_per_90` vs the 10 / 12 threshold |
| Bonus | `bps` per 90, mapped to an expected bonus (crude in v0.1 — see roadmap) |
| Saves | `saves_per_90` scaled by how much shooting the opponent generates |

The `#1` penalty taker gets an 8% uplift on goals: penalties keep arriving whether
or not any fell inside the sample so far.

### The transfer planner

All six gameweeks are solved together as one integer program (`PuLP` + the CBC
solver that ships with it). Choosing the best player for *this* week and worrying
about next week later is exactly how you end up making a transfer you regret.

Constraints, all enforced simultaneously:

- 15 players — 2 GKP, 5 DEF, 5 MID, 3 FWD; max 3 per club
- an XI of 11 with at least 1 GKP, 3 DEF, 2 MID, 1 FWD
- budget: your squad's market value plus the bank
- one free transfer per gameweek, banked up to five
- −4 per extra transfer, taken only when the expected gain beats it
- squad continuity: each gameweek's fifteen is the previous fifteen, plus buys, minus sells

The objective discounts future gameweeks by `planning.decay` (0.86 per week), so
near-term points count for more — a plan six weeks out is a sketch, not a promise.

---

## Configuration

Everything personal lives in `config.yaml`. The fields worth knowing:

| Key | Meaning |
|---|---|
| `entry.team_id` | Your FPL entry id. Currently `6014213`. |
| `planning.horizon` | Gameweeks to plan ahead. 6 is a good balance; beyond 8 the solver slows and the forecast is noise. |
| `planning.max_hit_per_gw` | You said you accept hits — this is capped at 8 (two hits) per week. |
| `strategy.mode` | `balanced` ignores ownership. Switch to `template` to protect overall rank, `differential` to chase in a mini-league. |
| `strategy.bench_weight` | How much a bench slot is worth. Non-zero so the optimiser does not fill the bench with 3.9m ghosts. |
| `chips.*` | Mark a chip `true` once you play it. All eight are currently unused; the first four expire after GW19. |
| `notify.hours_before_deadline` | 24, as requested. |

---

## Getting it on your phone

The dashboard is a **progressive web app**: a manifest, an icon and a service
worker ship alongside it, so once it is on a URL you can use your browser's
*Add to Home Screen* and it opens like an app, full screen, and still works with
no signal.

That needs a URL, which means hosting. The recommended route also solves a bigger
problem — your PC being asleep on a Friday night:

### GitHub Pages + GitHub Actions (recommended)

`.github/workflows/weekly.yml` is ready to go. GitHub runs the build in its own
cloud on a schedule, commits the refreshed `docs/`, and Pages serves it.

1. Push this repo to GitHub (private is fine — Pages works on private repos for
   personal accounts on any paid plan; otherwise make it public, there is nothing
   secret in here).
2. **Settings → Pages** → Source: *Deploy from a branch* → branch `main`, folder `/docs`.
3. **Settings → Secrets and variables → Actions**:
   - Variables → `FPL_TEAM_ID` = `6014213`
   - Secrets → `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional, see below)
4. Open `https://<username>.github.io/<repo>/` on your phone → *Add to Home Screen*.

The workflow runs three times a day, including late Thursday and Friday Bangkok
time, which is the 24-hour window before a normal Saturday deadline. It costs
nothing on a public repo.

### Windows Task Scheduler (alternative, or as well)

`run_weekly.bat` does a build then a notify. Point a daily task at it. This keeps
everything on your machine, but only runs when the machine is on.

### Telegram alerts

LINE Notify was discontinued in 2025, so Telegram is the simplest push that still
works and costs nothing.

1. Message **@BotFather** on Telegram → `/newbot` → copy the token.
2. Send your new bot any message, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `chat.id`.
3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as environment variables (or as
   GitHub Actions secrets) and flip `notify.telegram.enabled` to `true`.

The notifier will not spam you: it fires once per gameweek, inside the configured
window, and records that it did.

---

## Roadmap

The model is deliberately v0.1 — honest, readable, and not yet tuned. In rough
order of how much each would improve the output:

1. **Backtest and calibrate.** Run the model over 2024/25 and 2025/26 (the season
   mirror on GitHub has complete per-gameweek history) and fit the pieces that are
   currently hand-set: the bonus-point mapping, the xG/actual blend, the shrinkage
   constant, the decay. Measure against FPL's own `ep_next` as the baseline to beat.
2. **A real minutes model.** Playing time drives everything and is currently a
   simple start rate. Pulling `element-summary` histories for a shortlist would give
   a proper recent-minutes trend and catch rotation before it costs you a week.
3. **Chip planning.** The optimiser knows the chip rules but does not yet decide
   when to play them. Bench Boost and Triple Captain are worth solving for
   explicitly once double gameweeks are on the calendar.
4. **Correct selling prices.** The budget constraint uses market value; FPL gives
   back only half of a price rise. The gap is usually small but it can make a plan
   infeasible in practice.
5. **Price-change forecasting.** `transfer_pressure` is already computed but only
   displayed. Turning it into an expected value — buy tonight or lose 0.1m — is a
   small, high-value addition.

---

## Project layout

```
config.yaml                 everything personal
requirements.txt
run_weekly.bat              Windows Task Scheduler entry point
.github/workflows/          the cloud build
src/fplbot/
    fetch.py                API client + snapshot cache
    features.py             player rates, team strength, fixture schedule
    model.py                expected points
    optimize.py             the integer program
    report.py               dashboard rendering
    notify.py               Telegram
    cli.py                  command line
web/                        HTML template, PWA manifest, service worker, icons
tests/seed_offline_fixture.py   build a frozen dataset for offline work
data/snapshots/             one folder per run — this is your growing history
docs/                       the generated site (GitHub Pages serves this)
```

`data/snapshots/` is gitignored by default. If you want to accumulate history for
backtesting, remove that line from `.gitignore` — the JSON compresses well and a
season is a few hundred megabytes.

---

## A caveat worth repeating

Expected points are a model's estimate, not a forecast of what will happen.
A 6.2 EP captain scoring two is normal, not a bug. The value here is in making the
*average* decision better across a season and in never missing a deadline again —
not in getting any single week right.
