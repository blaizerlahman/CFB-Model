# CFB Predictive Model

A college football model that predicts each game's score differential against the Las Vegas
spread, and reports the historical success rate of every pick. Originally a set of Jupyter
notebooks driven by browser automation; now a Python package that runs from one command and
pulls all data from the official [CollegeFootballData.com](https://collegefootballdata.com) API.

```bash
python -m cfb_model predict
```

That single command detects the current week, ingests any finished week it is missing, fetches
the upcoming slate's betting lines and SP+ ratings, predicts every game, prints the picks
grouped by historical success rate, and saves them. It is non-interactive and cron-safe.

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Provide your CFBD API key by any one of these (checked in order):

1. `CFBD_API_KEY` environment variable
2. a `.env` file at the project root (`CFBD_API_KEY=...`)
3. `Repository/Personal Access/cfb_api.txt`

Then build the local database and models:

```bash
python -m cfb_model migrate
python -m cfb_model setup-season --year 2026
```

## Commands

| Command | What it does |
|---|---|
| `predict [--year Y --week W --day tue]` | The daily driver. Auto-detects week, catches up on missing ingests, predicts the upcoming slate, prints the tiered report, writes predictions to the DB and a CSV. |
| `update [--year Y --week W]` | Ingests finished weeks only: box scores, betting lines, rolling features, talent/SP+. Idempotent. |
| `analyze [--year Y] [--week W]` | Grades a week (`--week`) or rolls up a season: overall/good/great/best win rates, day-of-week splits, since-week-N trends, big-spread slice. |
| `setup-season --year Y` | Backfills team histories via the API and retrains + saves every qualifying team model for season Y. |
| `backtest --year Y [--seed 50]` | Replays a past season week by week with as-of models and that week's real lines and SP+ ratings. |
| `backfill-sp --year Y` | Reconstructs true weekly SP+ snapshots for a past season (see below). |
| `matchup TEAM1 TEAM2 [--spread X]` | On-demand prediction for any two teams. |
| `migrate [--validate-only]` | One-time load of the legacy CSVs into SQLite, with a validation gate. |

`--analyze` also works as a bare flag (`python -m cfb_model --analyze`).

### Running on a schedule

Every command is non-interactive, reads its key from the environment, exits 0 on success
(including "no slate today"), and takes a lockfile so overlapping runs no-op. A daily crontab
entry is just:

```bash
cd "/path/to/CFB Model" && .venv/bin/python -m cfb_model predict
```

Running daily during the season is intentional: lines move, each day's run is stored under its
own day tag, and grading keeps the most confident version of each game while the day-of-week
report shows which day's lines actually paid.

## API usage

The free CFBD tier allows 1,000 calls/month, which is comfortable:

| Activity | Calls |
|---|---|
| Ingesting a finished week | 3 |
| Each daily prediction run | 2 |
| Typical in-season week (1 ingest + 7 daily runs) | ~19 |
| **Typical month** | **~85** |
| One-time backfill of a full past season | ~48 |

The weekly SP+ backfill uses the Wayback Machine, not CFBD, so it costs no quota.

## How the model works

Each FBS team gets its own model. Features are that team's trailing 20-game and 8-game rolling
sums across 25 box-score statistics, the same 50 figures for its opponent, plus both teams'
247Sports talent composite and Bill Connelly's SP+ rating — about 104 features. Rolling windows
are lagged, so a game never sees its own result.

The estimator is a scikit-learn pipeline: constant imputation, standard scaling, then a
`BaggingRegressor` of 50 Lasso regressions whose alpha is chosen by a 5-fold grid search over
50 candidates. The target is score differential, and predictions round to the nearest half
point. Models train on every season before the one being predicted and are refit once per
season, not per week.

Missing statistics get special handling: CFBD did not track some categories in older seasons, so
zeros in those columns are treated as missing, regressed on score differential, and filled with
draws from the fitted distribution.

A prediction becomes a pick by comparing it to the Vegas spread. If the model expects a team to
beat the spread it calls a cover, otherwise it takes the opponent. The gap between prediction and
spread — the spread differential — is looked up in a table of historical success rates built by
replaying the 2018–2023 seasons ten times over and binning outcomes by that gap. Picks are then
reported in tiers: above 70%, 65–70%, 60–65%, and below 60%.

<img width="758" alt="Historical success rate by spread differential bin" src="https://github.com/user-attachments/assets/7a84a575-0d5f-4d5c-bbc4-a239c739f369">

The red, yellow, and green dashed lines represent 50, 60, and 70 percent success rates.

## Measured performance — read this before betting anything

Earlier versions of this project advertised a 57–58% success rate against the spread. **That
number does not survive a clean test, and neither do the tier success rates above.**

| Season | Setup | Result |
|---|---|---|
| 2024 (actually played, weekly ratings) | live weekly runs | 325–348, **48.3%** |
| 2024 | replay | 324–327, **49.8%** |
| 2025 | replay, true pre-kickoff weekly SP+ | 318–304, **51.1%** |
| 2025 | replay, SP+ leaked from the same week | 371–295, 55.7% |

At standard −110 odds a bettor needs **52.4%** just to break even. Measured honestly, this model
is at or below that line.

Two compounding sources of leakage explain the gap between the old claim and reality:

1. **Season-final SP+ in historical training data.** The API serves one SP+ rating per season, so
   every historical game carries its season's *final* rating — information that did not exist when
   that game was played. The 2018–2023 backtests behind the 57–58% figure were built this way, as
   was the bin table that assigns each pick its advertised success rate. The original README's own
   disclaimer flagged this and noted only 2024 was trustworthy; the clean results above confirm it.
2. **Same-week ratings in the 2025 reconstruction.** The first version of the SP+ backfill picked
   an archived snapshot from late in each week, after that week's games. Fixing it to use only
   snapshots published before the week's first kickoff dropped the 2025 result from 55.7% to 51.1%
   — and the "great bet" tier from 63.9% to 50.5%.

That second row is the useful lesson: the confidence tiers showed a large edge only while future
information was leaking in. On clean data they are indistinguishable from coin flips, because the
bins were themselves calibrated on leaked backtests.

The pipeline is verified faithful to the original model — a replay of week 15 of 2024 reproduces
the saved predictions exactly, and regrading 2024 reproduces all 687 stored results. So these
numbers reflect the model, not the refactor. Rebuilding the bins on leak-free data, and training
on contemporaneous rather than season-final SP+, are the obvious next steps for anyone who wants
this to have a real edge.

### Which teams get predicted

A team needs at least 56 games of history to earn a model. Teams that recently moved up from FCS
are therefore skipped automatically until they accumulate enough data, and games against them are
skipped too. This replaced a hardcoded list, so teams graduate on their own as they play — no
maintenance required.

## Weekly SP+ ratings

SP+ is published weekly during the season, but the CFBD API only serves one rating per season —
the final one. Backtesting a season with final ratings leaks end-of-season information into
week-1 predictions and inflates results. (The original notebooks hit exactly this problem; only
2024 was trustworthy because its weekly ratings had been captured by hand at the time.)

`backfill-sp` solves it by reading archived snapshots of ESPN's living SP+ article from the
Wayback Machine and parsing the full 136-team table. For each week it takes the freshest capture
published **before that week's first kickoff**, so a rating can never encode the result of a game
it is used to predict. A week with no qualifying capture gets no snapshot at all and is skipped by
the backtest rather than filled with later ratings — 2025 week 1 is such a week, because the
article was first archived after that week began.

This boundary matters more than it sounds: choosing snapshots from later in each week instead
inflated the 2025 backtest from 51.1% to 55.7%.

## Storage

Everything lives in one SQLite file, `data/cfb.db`: per-team game frames, betting lines from all
providers, weekly SP+ snapshots, talent ratings, the success-rate bins, predictions, graded
results, the model registry, and a separate log of ad-hoc matchup queries. Matchup queries are
deliberately kept out of the analysis path so that success reporting only ever reflects real
scheduled games.

Predictions and results are also exported as CSVs under `output/` in the original layout and
column order, so anything that read the old files still works.

The original per-team CSV directories are left untouched and are used as read-only correctness
oracles by the test suite.

## Serving predictions

The prediction core has no CLI coupling, so a web service can import it directly. See
[`examples/fastapi_app.py`](examples/fastapi_app.py):

```python
from cfb_model.model.predict import predict_matchup
```

## Tests

```bash
.venv/bin/pytest
```

The suite verifies ported logic against the original stored data rather than against itself:
recomputed rolling features must equal the values in the legacy frames, mapped API responses must
match the old CSV exports field for field, a full replay of week 15 of 2024 must reproduce that
week's saved predictions row for row, and regrading 2024 must reproduce all 687 stored results
with no outcome differences.

## Project history

This began as six notebooks (`p1`–`p6`, plus `MidSeason.ipynb` for in-season use) that required
manually editing week numbers across cells and driving a browser to download CSVs. Those
notebooks are still in the repository for reference. The refactor preserves their modeling
decisions exactly — same features, same estimator, same thresholds, same rounding — while
replacing the data plumbing and collapsing the workflow into one command.

## License

GNU Affero General Public License v3.0

## Source

https://github.com/blaizerlahman/CFB-Model

Contributions and forks are welcome.
