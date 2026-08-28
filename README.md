# CFB Predictive Model

A college football model that predicts each game's score differential against the Las Vegas
spread, and reports the historical success rate of every pick. Pulls all data from the official [CollegeFootballData.com](https://collegefootballdata.com) API.

```bash
python -m cfb_model predict
```

This command detects the current week, ingests any finished week it is missing, fetches
the upcoming slate's betting lines and SP+ ratings, predicts every game, prints the picks
grouped by historical success rate, and saves them. 

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
non-interactive and exit 0 on success.

### In-season, week to week

| Command | What it does |
|---|---|
| `predict [--year Y --week W --day tue]` | The daily driver. Works out the current week, ingests any finished week it is missing, fetches the slate's lines and SP+, predicts every game, prints the picks by tier, and writes them to the database and a CSV.
| `update [--year Y --week W]` | Ingest only: a finished week's box scores, betting lines, rolling features, talent and SP+. `predict` calls this for you, run it directly to backfill if needed.
| `snapshot-sp [--year Y --week W]` | Record this week's SP+ ratings. SP+ ratings update weekly and are not stored persistently, so must be captured every week.
| `analyze [--year Y] [--week W]` | With `--week`, grades that week and writes a results CSV. Without it, rolls up the season: overall and per-tier records, day-of-week splits, since-week-N trends, and probabilities based on spreads.

### Once per season

| Command | What it does |
|---|---|
| `setup-season --year Y` | Backfills missing team histories via the API and retrains every qualifying team model for season Y. Run before the season, and again after any change to past training data. Takes ~6 minutes. |

### Evaluating the model

| Command | What it does |
|---|---|
| `backtest --year Y [--seed 50]` | Replays a past season week by week using models trained only on earlier seasons and the ratings that were public before each week's games.
| `rebuild-bins [--seasons 2024,2025] [--reps 5] [--name leakfree]` | Recalibrates the spread-differential success table from leak-free replays. Only seasons with genuine weekly SP+ qualify. This takes a long time (hours): each rep retrains every team model.

### On demand

| Command | What it does |
|---|---|
| `matchup TEAM1 TEAM2 [--spread X] [--season Y]` | Predicts any two teams from current data. `--spread` is from TEAM1's perspective (negative means TEAM1 favoured) and adds the cover call and tier. These are logged separately and never enter success analysis.

# How the model works

Each FBS team gets its own model. Features are that team's trailing 20-game and 8-game rolling
sums across 25 box-score statistics, the same 50 figures for its opponent, plus both teams'
247Sports talent composite and Bill Connelly's SP+ rating — about 104 features. Rolling windows
are lagged, so a game never sees its own result.

The estimator is a scikit-learn pipeline: constant imputation, standard scaling, then a
`BaggingRegressor` of 50 Lasso regressions whose alpha is chosen by a 5-fold grid search over
50 candidates. The target is score differential, and predictions round to the nearest half
point. Models train on every season before the one being predicted and are refit once per
season.

Missing statistics get special handling: CFBD did not track some categories in older seasons, so
zeros in those columns are treated as missing, regressed on score differential, and filled with
draws from the fitted distribution.

Each game is predicted from **both** teams' models and the two are averaged.

A prediction becomes a pick by comparing it to the Vegas spread, and it is reported as a game: the
model's pick and the line it has to cover.

```
Key:
Model pick | Model projection | Historical hit rate for model based on predicted difference

  Tulane +9.5 vs Duke  |  Tulane loses by 4.5  |  62.71%
  Clemson +10.5 vs LSU |  Clemson wins by 2.5  |  56.18%
```

Back Tulane getting 9.5: the model expects them to lose by only 4.5, comfortably inside the line.
The projection is always stated for the side being backed, so it reads the same way whether that
side is winning outright or just covering.

The hit rate is the historical record for disagreements of that size. Measured over 2024-25 the model's largest disagreements with the market line (so pretty optimistic).

Each run writes the CSV and a `.txt` copy of this report beside it. The CSVs keep the original
columns and add `pick` and `pickSpread`. The gap between prediction and
spread — the spread differential — is looked up in a table of historical success rates built by
replaying the 2018–2023 seasons ten times over and binning outcomes by that gap. Picks are then
reported in tiers: above 70%, 65–70%, 60–65%, and below 60%.

<img width="758" alt="Historical success rate by spread differential bin" src="https://github.com/user-attachments/assets/7a84a575-0d5f-4d5c-bbc4-a239c739f369">

The red, yellow, and green dashed lines represent 50, 60, and 70 percent success rates.

## Measured performance (read this before using the model)

Earlier versions of this project advertised a 57–58% success rate against the spread. **That
number was derived from bad data using incorrect SP+ ratings.**

| Season | Setup | Result |
|---|---|---|
| 2024 (actually played, weekly ratings) | live weekly runs | 325–348, **48.3%** |
| 2025 | replay | 315–308, **50.6%** |

At standard −110 odds a bettor needs **52.4%** just to break even. This model
is at or below that line.

### Which teams get predicted

A team needs at least 56 games of history to earn a model. Teams that recently moved up from FCS
are therefore skipped automatically until they accumulate enough data, and games against them are
skipped too.

## Storage

Everything lives in one SQLite file, `data/cfb.db`: per-team game frames, betting lines from all
providers, weekly SP+ snapshots, talent ratings, the success-rate bins, predictions, graded
results, the model registry, and a separate log of ad-hoc matchup queries. 

Predictions and results are also exported as CSVs under `output/` in the original layout and
column order, so anything that read the old files still works.

## Serving predictions

The prediction engine has no CLI coupling, so a web service can import it directly. See
[`examples/fastapi_app.py`](examples/fastapi_app.py):

```python
from cfb_model.model.predict import predict_matchup
```

## Project history

This began as six notebooks (`p1`–`p6`, plus `MidSeason.ipynb` for in-season use) that required
manually editing week numbers across cells and driving a browser to download CSVs. Those
notebooks are still in the repository for reference. The refactor preserves their modeling
decisions exactly while replacing the data plumbing and collapsing the workflow into one command.

## License

GNU Affero General Public License v3.0

Contributions and forks are welcome.
