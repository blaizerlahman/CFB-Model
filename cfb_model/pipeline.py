"""Weekly operations: finished-week ingestion and prediction runs.

Ports grabLastWeekData (ingest) and the runModel orchestration (predict),
API-backed and DB-backed. Sentinel rows exist only in memory during a
prediction run — the DB never stores them (matching the legacy weekly CSV
exports, which were always rebuilt from clean prior-week state).
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from cfb_model.api import mapping
from cfb_model.api.client import CfbdClient
from cfb_model.config import Settings, get_settings
from cfb_model.constants import MIN_FBS_GAMES, MIN_FCS_GAMES, TOTAL_TD_COMPONENTS
from cfb_model.data.store import Store
from cfb_model.features.assemble import feature_columns, own_rolling_columns
from cfb_model.features.rolling import update_last_row_rolling
from cfb_model.model.predict import (
    build_upcoming_frames,
    classification_report,
    predict_week,
    select_preferred_lines,
)

logger = logging.getLogger(__name__)

# The 13 betting columns written onto a completed game row (legacy: the
# frame's last 13 columns; fixed list here).
BETTING_WRITE_COLUMNS = [
    "Id", "HomeTeam", "HomeScore", "AwayTeam", "AwayScore", "LineProvider",
    "OverUnder", "Spread", "FormattedSpread", "OpeningSpread",
    "OpeningOverUnder", "HomeMoneyline", "AwayMoneyline",
]


def build_week_rows(wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-team single-game rows for one week (the tempDict rebuild in
    grabLastWeekData): totalTDs, self-merge for _opp, derived columns."""
    temp: dict[str, pd.DataFrame] = {}
    for team in wide["School"].unique():
        game_ids = wide.loc[wide["School"] == team, "Game Id"].unique()
        team_df = wide[wide["Game Id"].isin(game_ids)].copy()
        team_df = team_df.sort_values(by=["Year", "Week"]).reset_index(drop=True)
        team_df["totalTDs"] = team_df[list(TOTAL_TD_COMPONENTS)].sum(axis=1, skipna=True)

        merged = team_df.merge(team_df, on="Game Id", suffixes=("", "_opp"))
        merged = merged[merged["School"] != merged["School_opp"]]
        merged["scoreDiff"] = merged["Points"] - merged["Points_opp"]
        merged["pointTotal"] = merged["Points"] + merged["Points_opp"]
        merged["Win"] = merged["scoreDiff"] > 0

        temp[team] = merged[merged["School"] == team].reset_index(drop=True)
    return temp


def ingest_week(store: Store, client: CfbdClient, year: int, week: int,
                settings: Settings | None = None) -> dict:
    """grabLastWeekData equivalent: fetch the finished week's box scores and
    lines, append one real row per team, persist. Idempotent (PK upserts)."""
    settings = settings or get_settings()

    records = client.team_game_stats(year, week=week, classification="fbs")
    records += client.team_game_stats(year, week=week, classification="fcs")
    wide = mapping.team_stats_to_wide(records, year=year, week=week)
    if wide.empty:
        return {"year": year, "week": week, "teams_updated": 0, "note": "no games returned"}

    line_records = client.lines(year, week=week)
    lines = mapping.lines_to_frame(line_records)
    store.upsert_lines(year, week, lines)
    preferred = select_preferred_lines(lines)

    temp = build_week_rows(wide)

    # SP+ snapshot for this week: the one captured at prediction time if
    # present, else the closest earlier snapshot, else fetch current.
    sp_weeks = [w for w in store.sp_weeks(year) if w <= week]
    if sp_weeks:
        sp = store.load_sp(year, sp_weeks[-1])
    else:
        sp = mapping.sp_to_frame(client.sp_ratings(year))
        store.upsert_sp(year, week, sp)

    talent_table = store.load_talent(year)
    if week == 1 and talent_table.empty:
        talent_table = resolve_talent(store, client, year)

    fbs = store.load_all_team_frames("fbs")
    fcs = store.load_all_team_frames("fcs")
    classification = {k: "fbs" for k in fbs} | {k: "fcs" for k in fcs}
    frames = fbs | fcs

    sample = next(iter(fbs.values()))
    rolling_cols = own_rolling_columns(sample)

    # Pass 1: append the played row + own rolling sums.
    updated: list[str] = []
    for team in list(frames):
        if team not in temp:
            continue
        # True idempotency: a game already in the frame is never re-appended
        # (recomputing rolling over a duplicated row would corrupt it).
        if temp[team]["Game Id"].iloc[0] in frames[team]["Game Id"].values:
            continue
        df = pd.concat([frames[team], temp[team]], ignore_index=True)
        df = update_last_row_rolling(df, rolling_cols)
        frames[team] = df
        updated.append(team)

    # Pass 2: _opp rolling from the opponent's extended frame, talent carried
    # forward, SP from the snapshot, opponent talent/SP (FBS opponents only —
    # the legacy FCS branch was dead, see PROGRESS.md).
    for team in updated:
        df = frames[team]
        last_index = df.index[-1]
        last_row = df.iloc[-1]

        opp_name = last_row["School_opp"]
        opp_df = frames.get(opp_name)
        if opp_df is not None:
            opp_row = opp_df[opp_df["Game Id"] == last_row["Game Id"]]
            if not opp_row.empty:
                for col in rolling_cols:
                    df.loc[last_index, col + "_opp"] = opp_row.iloc[-1][col]

        if "talent" in df.columns:
            # Week 1 seeds from the season's ratings tables (matching the
            # observed legacy data); later weeks carry talent forward from
            # the previous row and take SP from the weekly snapshot.
            if week == 1:
                team_talent = talent_table.loc[talent_table["School"] == team, "Talent"]
                if len(team_talent.values):
                    df.loc[last_index, "talent"] = team_talent.values[0]
            elif last_index >= 1:
                df.loc[last_index, "talent"] = df.loc[last_index - 1, "talent"]
            team_sp = sp.loc[sp["Team"] == team, "Rating"]
            if len(team_sp.values) and "SP" in df.columns:
                df.loc[last_index, "SP"] = team_sp.values[0]

            if opp_df is not None and classification.get(opp_name) == "fbs":
                if week == 1:
                    opp_talent = talent_table.loc[talent_table["School"] == opp_name, "Talent"]
                    if len(opp_talent.values) and "talent_opp" in df.columns:
                        df.loc[last_index, "talent_opp"] = opp_talent.values[0]
                    if "SP_opp" in df.columns:
                        opp_sp = sp.loc[sp["Team"] == opp_name, "Rating"]
                        if len(opp_sp.values):
                            df.loc[last_index, "SP_opp"] = opp_sp.values[0]
                else:
                    filtered = opp_df[opp_df["Year"] == year]
                    if not filtered.empty:
                        opp_first = filtered.iloc[0]
                        if "talent_opp" in df.columns:
                            df.loc[last_index, "talent_opp"] = opp_first.get("talent")
                        if "SP_opp" in df.columns:
                            opp_sp = sp.loc[sp["Team"] == opp_name, "Rating"]
                            if len(opp_sp.values):
                                df.loc[last_index, "SP_opp"] = opp_sp.values[0]

        # Betting columns on FBS rows (home Spread flipped, away raw).
        if classification[team] == "fbs":
            game_row = preferred[preferred["Id"] == last_row["Game Id"]]
            if not game_row.empty:
                game_row = game_row.iloc[0].copy()
                if game_row["HomeTeam"] == team:
                    game_row["Spread"] = float(game_row["Spread"]) * -1
                df.loc[last_index, BETTING_WRITE_COLUMNS] = game_row[BETTING_WRITE_COLUMNS].values

        frames[team] = df

    # Persist: one appended row per updated team.
    for team in updated:
        row = frames[team].iloc[[-1]]
        store.append_team_rows(team, classification[team], row)

    return {"year": year, "week": week, "teams_updated": len(updated),
            "api_calls": client.calls_made}


def resolve_talent(store: Store, client: CfbdClient, year: int, log=print) -> pd.DataFrame:
    """Talent ratings for `year`, falling back to the most recent published
    season when CFBD has not released the current one yet.

    Without the fallback, missing talent becomes NaN on the upcoming-game row
    and the pipeline's constant imputer turns it into 0 — far outside the
    trained range (~400-1050), which would distort every prediction. Talent
    composites move slowly year to year, so carrying the prior season forward
    is far closer to the truth than zero.
    """
    talent = store.load_talent(year)
    if not talent.empty:
        return talent

    talent = mapping.talent_to_frame(client.talent(year))
    if not talent.empty:
        store.upsert_talent(year, talent)
        return talent

    row = store.conn.execute(
        "SELECT MAX(season) FROM talent WHERE season < ?", (year,)
    ).fetchone()
    fallback_year = row[0] if row else None
    if fallback_year is None:
        for candidate in range(year - 1, year - 4, -1):
            candidate_talent = mapping.talent_to_frame(client.talent(candidate))
            if not candidate_talent.empty:
                store.upsert_talent(candidate, candidate_talent)
                fallback_year = candidate
                break
    if fallback_year is None:
        log(f"WARNING: no talent ratings available for {year} or any prior season.")
        return pd.DataFrame(columns=["School", "Talent"])

    log(f"WARNING: CFBD has no {year} talent ratings yet — using {fallback_year} values.")
    return store.load_talent(int(fallback_year))


def latest_ingested_week(store: Store, year: int) -> int | None:
    row = store.conn.execute(
        'SELECT MAX("Week") FROM team_games WHERE "Year" = ?', (year,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def load_gated_frames(store: Store) -> tuple[dict, dict]:
    """Legacy load gates: >=56 games FBS, >=10 FCS."""
    fbs = {k: v for k, v in store.load_all_team_frames("fbs").items() if len(v) >= MIN_FBS_GAMES}
    fcs = {k: v for k, v in store.load_all_team_frames("fcs").items() if len(v) >= MIN_FCS_GAMES}
    return fbs, fcs


def derived_skip_teams(line_records: list[dict], model_dict: dict) -> set[str]:
    """The data-derived skip rule replacing the hardcoded list: any team CFBD
    classifies as FBS that has no qualifying model."""
    skip: set[str] = set()
    for game in line_records:
        for side in ("home", "away"):
            name = mapping.normalize_school(game.get(f"{side}Team") or "")
            if game.get(f"{side}Classification") == "fbs" and name and name not in model_dict:
                skip.add(name)
    return skip


def load_models(settings: Settings, season: int) -> dict:
    """Load {team_key: pipeline} for a season from models/, falling back to
    the legacy Team Models pickles for 2024."""
    import glob as _glob
    import warnings
    from pathlib import Path

    import joblib

    paths = _glob.glob(str(settings.models_dir / f"*_model_{season}.pkl"))
    if not paths and season == 2024:
        paths = _glob.glob(str(settings.legacy_team_models / "*2024*.pkl"))
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in paths:
            key = Path(path).name.split("_model")[0].replace("_", " ")
            models[key] = joblib.load(path)
    return models


def predict_run(store: Store, client: CfbdClient, year: int | None = None,
                week: int | None = None, day: str | None = None,
                settings: Settings | None = None, log=print) -> pd.DataFrame | None:
    """The daily one-liner: auto-detect the week, ingest any finished week
    that is missing, build the upcoming slate, predict, classify, persist,
    export the legacy CSV, print the report. Returns the preds frame (None
    when there is no slate)."""
    settings = settings or get_settings()

    if year is None or week is None:
        detected = mapping.current_week(client.calendar(year or datetime.now().year))
        if detected is None:
            log("No upcoming CFB week found (off-season). Nothing to do.")
            return None
        year = year or detected[0]
        week = week if week is not None else detected[1]
    day = day or datetime.now().strftime("%a").lower()[:3]

    # Ingest the previous week if its games are missing from the DB.
    if week > 1:
        latest = latest_ingested_week(store, year)
        if latest is None or latest < week - 1:
            target = (latest or 0) + 1
            for w in range(target, week):
                log(f"Ingesting missing week {year} w{w}...")
                summary = ingest_week(store, client, year, w, settings)
                log(f"  updated {summary['teams_updated']} teams")

    line_records = client.lines(year, week=week)
    lines = mapping.lines_to_frame(line_records)
    if lines.empty:
        log(f"No betting lines for {year} week {week} yet. Nothing to predict.")
        return None
    store.upsert_lines(year, week, lines)
    preferred = select_preferred_lines(lines)

    sp = mapping.sp_to_frame(client.sp_ratings(year))
    store.upsert_sp(year, week, sp)
    talent = resolve_talent(store, client, year, log)

    fbs, fcs = load_gated_frames(store)
    fbs, fcs = build_upcoming_frames(fbs, fcs, preferred, sp, talent, year, week)

    models = load_models(settings, year)
    if not models:
        raise RuntimeError(
            f"No models found for season {year} — run `setup-season --year {year}` first."
        )

    skip = derived_skip_teams(line_records, models)
    features = feature_columns(next(iter(fbs.values())))
    preds, statuses = predict_week(models, fbs, features, skip_teams=skip)

    bins = store.load_bins()
    log(classification_report(preds, bins, fbs, fcs, statuses))

    if not preds.empty:
        store.upsert_predictions(year, week, day, preds)
        from cfb_model.export import preds_csv_path, write_preds_csv

        out = write_preds_csv(preds, preds_csv_path(settings.output_root, year, week, day))
        log(f"\nSaved {len(preds)} predictions -> {out}")
    log(f"API calls this run: {client.calls_made}")
    return preds
