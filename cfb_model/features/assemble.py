"""Team-frame assembly — verbatim ports of createTeam / mergeRollingSum /
the week-0 fix (p1) and predictMissing (p4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from cfb_model.constants import (
    FEATURE_SUBSTRINGS,
    LEGACY_ZERO_AS_MISSING_PATTERNS,
    PREDICT_MISSING_EXTRA_COLS,
    TOTAL_TD_COMPONENTS,
)
from cfb_model.features.rolling import add_rolling_features


def feature_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """~104 model features: any column containing rolling_sum/talent/SP.

    `exclude` drops columns whose name contains any of the given substrings.
    It exists for leak-free experiments: historical SP+ rows carry the
    season-FINAL rating (CFBD serves only one per season), so training on
    them leaks end-of-season information that is not available at kickoff.
    """
    cols = [c for c in df.columns if any(s in c for s in FEATURE_SUBSTRINGS)]
    if exclude:
        cols = [c for c in cols if not any(e in c for e in exclude)]
    return cols


def own_rolling_columns(df: pd.DataFrame) -> list[str]:
    """Own-side rolling columns (excludes the _opp copies)."""
    return [c for c in df.columns if "rolling_sum" in c and "_opp" not in c]


def fix_columns(df: pd.DataFrame) -> list[str]:
    """predictMissing's fixCols: rolling cols (own AND _opp) matching the
    legacy-missing patterns, plus talent / talent_opp (p4)."""
    rolling_cols = [c for c in df.columns if "rolling_sum" in c]
    fix_cols = [
        c for c in rolling_cols
        if any(word in c for word in LEGACY_ZERO_AS_MISSING_PATTERNS)
    ]
    fix_cols.extend(PREDICT_MISSING_EXTRA_COLS)
    return fix_cols


def build_team_frame(wide_games: pd.DataFrame, team: str) -> pd.DataFrame:
    """Exact port of p1 createTeam(team).

    `wide_games` is the pivoted all-teams frame (one row per game/team) with
    columns Game Id, School, Conference, HomeAway, Points, Week, Year plus the
    stat categories. Week may be 'Week N' strings (legacy) or ints (API path).
    """
    team_games = wide_games[wide_games["School"] == team]
    game_ids = team_games["Game Id"].unique()

    team_df = wide_games[wide_games["Game Id"].isin(game_ids)].copy()

    if team_df["Week"].dtype == object:
        team_df["Week"] = team_df["Week"].str[-2:].astype(int)
    team_df = team_df.sort_values(by=["Year", "Week"])
    team_df = team_df.reset_index(drop=True)

    team_df["totalTDs"] = team_df[list(TOTAL_TD_COMPONENTS)].sum(axis=1, skipna=True)

    merged = team_df.merge(team_df, on="Game Id", suffixes=("", "_opp"))
    merged = merged[merged["School"] != merged["School_opp"]]

    merged["scoreDiff"] = merged["Points"] - merged["Points_opp"]
    merged["pointTotal"] = merged["Points"] + merged["Points_opp"]
    merged["Win"] = merged["scoreDiff"] > 0

    team_df = merged[merged["School"] == team]
    team_df = add_rolling_features(team_df.copy())
    return team_df


def attach_opp_rolling(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Exact port of p1 mergeRollingSum, applied across a dict of team frames:
    for every row, copy the opponent's own rolling_sum columns (matched by
    Game Id in the opponent's frame) into '{col}_opp'."""
    result: dict[str, pd.DataFrame] = {}
    for team, team_df in frames.items():
        team_df = team_df.copy()
        for index, row in team_df.iterrows():
            game_id = row["Game Id"]
            opp_name = row["School_opp"]
            if opp_name not in frames:
                continue
            opp_row = frames[opp_name][frames[opp_name]["Game Id"] == game_id]
            if opp_row.empty:
                continue
            rolling_cols = [c for c in opp_row.columns if c.startswith("rolling_sum")]
            for col in rolling_cols:
                team_df.loc[index, col + "_opp"] = opp_row.iloc[0][col]
        result[team] = team_df
    return result


def fix_week_zero(team_df: pd.DataFrame) -> pd.DataFrame:
    """Exact port of the week-0 fix: when a season holds two Week==1 games,
    swap that season's first two rows and relabel the first as Week 0."""
    for year in np.unique(team_df["Year"].values):
        year_df = team_df[team_df["Year"] == year]
        if np.sum(year_df["Week"].values == 1) > 1:
            week0_idx = year_df.index[0]
            week1_idx = year_df.index[1]
            team_df.loc[week0_idx], team_df.loc[week1_idx] = (
                team_df.loc[week1_idx].copy(),
                team_df.loc[week0_idx].copy(),
            )
            team_df.at[week0_idx, "Week"] = 0
    return team_df


def predict_missing(model_team: pd.DataFrame, fix_cols: list[str], random_state) -> None:
    """Exact port of p4 predictMissing(modelTeam, fixCols, randomState).

    Mutates model_team in place. For each fix column: 0 -> NaN, fit a
    univariate LinearRegression on scoreDiff over known rows, predict the
    missing rows, and add Gaussian noise scaled by the column's std. The RNG
    is re-created with the same seed for every column (original behavior).
    """
    for col in fix_cols:
        model_team[col] = model_team[col].replace(0, np.nan)

        non_nulls = ~model_team[col].isnull()
        if non_nulls.any():
            X = model_team.loc[non_nulls, ["scoreDiff"]].values.reshape(-1, 1)
            y = model_team.loc[non_nulls, col].values.reshape(-1, 1)

            lr = LinearRegression()
            lr.fit(X, y)

            missing_vals = model_team.loc[model_team[col].isnull(), ["scoreDiff"]].values.reshape(-1, 1)
            if missing_vals.shape[0] > 0:
                preds = lr.predict(missing_vals).flatten()
                std = model_team[col].std(skipna=True)
                rng = np.random.default_rng(seed=random_state)
                sample_vals = rng.normal(loc=preds, scale=std, size=len(preds))
                model_team.loc[model_team[col].isnull(), col] = sample_vals
