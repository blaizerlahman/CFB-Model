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

Everything runs as `python -m cfb_model <command>`. All commands are
non-interactive, take a lockfile so overlapping runs no-op, and exit 0 on success
(including "nothing to do today").

### In-season, week to week

| Command | What it does |
|---|---|
| `predict [--year Y --week W --day tue]` | The daily driver. Works out the current week, ingests any finished week it is missing, fetches the slate's lines and SP+, predicts every game, prints the picks by tier, and writes them to the database and a CSV. Defaults are almost always right — run it bare. |
| `update [--year Y --week W]` | Ingest only: a finished week's box scores, betting lines, rolling features, talent and SP+. `predict` calls this for you; run it directly to backfill a gap. Idempotent. |
| `snapshot-sp [--year Y --week W]` | Record this week's SP+ ratings. Worth its own weekly cron entry — CFBD serves SP+ live and freezes it afterwards, so a week not captured while current is gone for good. First capture of a week wins. |
| `analyze [--year Y] [--week W]` | With `--week`, grades that week and writes a results CSV. Without it, rolls up the season: overall and per-tier records, day-of-week splits, since-week-N trends, and a big-spread slice. |

A daily crontab line is just:

```bash
cd "/path/to/CFB Model" && .venv/bin/python -m cfb_model predict
```

A full weekly schedule lives in `scripts/crontab.txt` — predictions daily at
4 PM Central, an SP+ snapshot Tuesdays at 9 PM Central, and grading Sundays at
2 PM Central. cron fires on the machine's local clock rather than a timezone
of its own, so those lines are written in local time; the file records the
conversion it used. Install it with:

```bash
crontab scripts/crontab.txt
```

Run that from a terminal you are sitting at: macOS asks for permission the
first time anything edits your crontab, and the prompt cannot appear for a
background process. If cron proves unreliable under macOS's privacy controls,
the same three commands work just as well from launchd or any scheduler.

The Sunday job is two commands: `update --current` pulls in the results of the
week that just finished, then `analyze --last-week` scores that week's
predictions against them. Both work out the week from the schedule, and both
exit quietly out of season, so the jobs can stay installed year round.

### Once per season

| Command | What it does |
|---|---|
| `setup-season --year Y` | Backfills missing team histories via the API and retrains every qualifying team model for season Y. Run before the season, and again after any change to past training data. Takes ~6 minutes. |
| `migrate [--validate-only]` | One-time load of the legacy CSVs into SQLite, with a validation gate. Already done; re-run to rebuild the store from the original files. |

### Evaluating the model

| Command | What it does |
|---|---|
| `backtest --year Y [--seed 50] [--bins NAME] [--no-sp]` | Replays a past season week by week using models trained only on earlier seasons and the ratings that were public before each week's games. `--bins legacy` grades with the original calibration for comparison; `--no-sp` trains without SP+ features. |
| `rebuild-bins [--seasons 2024,2025] [--reps 5] [--name leakfree]` | Recalibrates the spread-differential success table from leak-free replays. Only seasons with genuine weekly SP+ qualify. Slow (hours): each rep retrains every team model. `--from-replays FILE` rebuilds from an existing pooled run instead. |

### Getting weekly SP+ for past seasons

Ratings for a finished season can only be recovered if something recorded them
at the time. In rough order of preference:

| Command | Source |
|---|---|
| `import-sp-sheet --year Y` | Connelly's public season spreadsheet. First-party and reaches week 0/1. Only 2025's sheet carries weekly ratings; earlier ones hold picks alone. |
| `backfill-sp --year Y` | Wayback captures of ESPN's living article, one per week, each verified to predate that week's first kickoff. |
| `import-sp-json [--file F]` | Output of `scripts/espn_sp_scrape.js`, which you run in a signed-in browser console because ESPN blocks server-side fetches. Finds the file on its own in `output/sp_manual`, the project root, or `~/Downloads`. |
| `import-sp-html --year Y --dir D` | ESPN pages saved by hand, named `week5.html` and so on. |
| **`apply-weekly-sp --year Y`** | **Run this after any of the above.** Importing only fills the ratings table; the model reads SP+ off each game row, and this is what moves it across. Skipping it means the import silently changes nothing. |

### On demand

| Command | What it does |
|---|---|
| `matchup TEAM1 TEAM2 [--spread X] [--season Y]` | Predicts any two teams from current data. `--spread` is from TEAM1's perspective (negative means TEAM1 favoured) and adds the cover call and tier. These are logged separately and never enter success analysis. |

### A worked example

Recovering a past season's weekly ratings, end to end:

```bash
python -m cfb_model import-sp-json                  # or backfill-sp / import-sp-sheet
python -m cfb_model apply-weekly-sp --year 2023     # the step that makes it count
python -m cfb_model setup-season --year 2026        # retrain on the corrected data
python -m cfb_model backtest --year 2025            # check whether it changed anything
```

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
| 2025 | replay, true pre-kickoff weekly SP+ | 315–308, **50.6%** |
| 2025 | replay, SP+ leaked from the same week | 371–295, 55.7% |

Where the ratings come from matters, so it is worth stating plainly. The 2024 replay uses the
weekly SP+ files captured during that season, each downloaded mid-week before the games it was
used to predict. The 2025 replay reconstructs the equivalent from archived snapshots, each one
verified to predate its week's first kickoff; 2025 week 1 has no such snapshot and is excluded
rather than filled in. Both seasons therefore use contemporaneous weekly ratings.

At standard −110 odds a bettor needs **52.4%** just to break even. Measured honestly, this model
is at or below that line.

Two compounding sources of leakage explain the gap between the old claim and reality:

1. **Season-final SP+ in historical training data.** The API serves one SP+ rating per season, so
   every historical game carries its season's *final* rating — information that did not exist when
   that game was played. Measured directly: every season from 2013 to 2023 carries exactly one
   distinct SP+ value per team, while 2024 and 2025 carry about ten. The 2018–2023 backtests behind
   the 57–58% figure were built this way, as was the bin table that assigns each pick its
   advertised success rate. The original README's own disclaimer flagged this and noted only 2024
   was trustworthy; the clean results above confirm it.
2. **Same-week ratings in the 2025 reconstruction.** The first version of the SP+ backfill picked
   an archived snapshot from late in each week, after that week's games. Fixing it to use only
   snapshots published before the week's first kickoff dropped the 2025 result from 55.7% to 51.1%
   — and the "great bet" tier from 63.9% to 50.5%.

That second row is the useful lesson: the confidence tiers showed a large edge only while future
information was leaking in.

### The tiers were recalibrated, and the result is worth stating plainly

The success rates advertised for each tier came from `Bin Data/Bin_Data.csv`, built by replaying
2018–2023 with season-final SP+. Those replays were rebuilt using only pre-kickoff information —
6,673 graded games pooled from ten replays of 2024 and 2025 — keeping the original binning
arithmetic and changing only which games feed it.

Calibration improved a great deal. The legacy table contained bins claiming success rates as high
as **100%**; the rebuilt table tops out at **62.7%**. Graded out-of-sample (each season scored with
bins built from the other), the old table promised 57–58% overall and delivered 49–51%, while the
rebuilt table promises 48–51% and delivers 48.8–51.1%. It now tells the truth.

What recalibration could not do is make the tiers *work*. Sorting by how far the model departs from
the spread does not find better bets:

| \|prediction − spread\| | Games | Win rate |
|---|---|---|
| 0–1 | 392 | 49.0% |
| 2–3 | 675 | 51.1% |
| 4–6 | 1,051 | 53.3% |
| 8–10 | 716 | 52.1% |
| 10–14 | 877 | 47.1% |
| 14+ | 701 | 46.6% |

The correlation between disagreement size and winning is **−0.031** (n=6,600, z=−2.52) — flat, and
if anything slightly inverted. The largest disagreements are the *worst* bets, which is intuitive
once stated: a 20-point gap against an efficient line usually means the model has bad information
about a team, not that it has found an edge.

So the honest reading is that the tier system never had predictive power; it had leakage. The
recalibrated table is the default (`bin_set` in `config.py`, or `--bins legacy` to compare), and it
now reports what the model actually does, which is hover around break-even with no reliable way to
pick out its better calls.

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

**Capture ratings as the season runs.** CFBD serves SP+ live during a season and freezes it only
afterwards, so a season's weekly values exist later only if something recorded them at the time.
`snapshot-sp` does that and is worth its own weekly cron entry, independent of prediction runs:

```bash
cd "/path/to/CFB Model" && .venv/bin/python -m cfb_model snapshot-sp
```

The first capture of a week wins, so a later re-run cannot overwrite ratings that were public
before kickoff with ones that already reflect that week's results.

Weekly ratings for earlier seasons are simply gone. CFBD returns only the season-final value and
ignores a `week` parameter; ESPN's 2019–2023 weekly articles are paywalled in the archive; and
Football Outsiders' pages, though richly archived, were captured after each season ended (and
pre-2014 uses an older, incompatible scale). Seasons before 2024 therefore keep season-final SP+
in training, which is the main open calibration problem described above.

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
