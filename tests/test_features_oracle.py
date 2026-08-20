"""Phase 1 oracle tests: ported feature functions must reproduce the values
stored in the legacy per-team frames (read-only).

Stored rolling values at row i depend only on visible rows when i >= window
(rows below that were computed before the historical 20-row warm-up drop and
reference games not present in the file), so exact-match assertions start at
i == window.
"""

import numpy as np
import pandas as pd
import pytest

from cfb_model.config import get_settings
from cfb_model.constants import BASE_STATS, PER_GAME_STATS, TOTAL_TD_COMPONENTS, WINDOWS
from cfb_model.features.assemble import fix_week_zero, predict_missing
from cfb_model.features.rolling import (
    encode_stat,
    lagged_rolling_sum,
    update_last_row_rolling,
)

SETTINGS = get_settings()
ORACLE_TEAMS = ("Alabama", "Florida_State", "Ohio_State", "Boise_State")


def load_week16(team: str) -> pd.DataFrame:
    path = SETTINGS.legacy_season_dir / "Week 16" / f"{team}_2024week16.csv"
    df = pd.read_csv(path)
    # Sentinel rows (upcoming-game placeholders) have NaN School.
    return df[df["School"].notna()].reset_index(drop=True)


@pytest.mark.parametrize("team", ORACLE_TEAMS)
def test_rolling_sums_match_stored(team):
    df = load_week16(team)
    for window in WINDOWS:
        for stat in BASE_STATS:
            col = f"rolling_sum_{stat}{window}"
            recomputed = lagged_rolling_sum(df[stat], window)
            if stat in PER_GAME_STATS:
                recomputed = recomputed / window
            stored = df[col]
            pd.testing.assert_series_equal(
                recomputed.iloc[window:],
                stored.iloc[window:],
                check_names=False,
                rtol=1e-9,
                obj=f"{team} {col}",
            )


@pytest.mark.parametrize("team", ORACLE_TEAMS)
def test_derived_columns_match_stored(team):
    df = load_week16(team)

    total_tds = df[list(TOTAL_TD_COMPONENTS)].sum(axis=1, skipna=True)
    pd.testing.assert_series_equal(total_tds, df["totalTDs"], check_names=False, rtol=1e-9)

    opp_components = [c + "_opp" for c in TOTAL_TD_COMPONENTS]
    total_tds_opp = df[opp_components].sum(axis=1, skipna=True)
    pd.testing.assert_series_equal(total_tds_opp, df["totalTDs_opp"], check_names=False, rtol=1e-9)

    pd.testing.assert_series_equal(
        df["Points"] - df["Points_opp"], df["scoreDiff"], check_names=False, rtol=1e-9
    )
    pd.testing.assert_series_equal(
        df["Points"] + df["Points_opp"], df["pointTotal"], check_names=False, rtol=1e-9
    )
    pd.testing.assert_series_equal(
        (df["scoreDiff"] > 0), df["Win"].astype(bool), check_names=False
    )


def test_rolling_edge_cases():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = lagged_rolling_sum(s, 3)
    assert out.iloc[0] != out.iloc[0] and out.iloc[1] != out.iloc[1]  # NaN, NaN
    assert out.iloc[2] == 0.0            # i == window-1: empty wrapped slice
    assert out.iloc[3] == 1 + 2 + 3      # lagged: excludes current row
    assert out.iloc[4] == 2 + 3 + 4

    all_nan = pd.Series([np.nan] * 5)
    assert lagged_rolling_sum(all_nan, 3).iloc[4] == 0.0  # skipna sum of NaNs


def test_update_last_row_rolling_matches_full_recompute():
    df = load_week16("Alabama").copy()
    rolling_cols = [c for c in df.columns if "rolling_sum" in c and "_opp" not in c]

    new_row = df.iloc[-1].copy()
    new_row["Points"] = 31.0
    df2 = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    df2 = update_last_row_rolling(df2, rolling_cols)

    last = df2.index[-1]
    for stat in ("Points", "totalYards", "yardsPerPass"):
        for window in WINDOWS:
            expected = df2[stat].iloc[len(df2) - 1 - window : len(df2) - 1].sum(skipna=True)
            if stat in PER_GAME_STATS:
                expected = expected / window
            assert df2.at[last, f"rolling_sum_{stat}{window}"] == pytest.approx(expected)


def test_encode_stat():
    assert encode_stat("12-95") == pytest.approx(12.95)
    assert encode_stat("31:24") == pytest.approx(31.24)
    assert encode_stat("7") == 7.0
    assert encode_stat(3.5) == 3.5
    assert np.isnan(encode_stat("DNP"))


def test_fix_week_zero():
    df = pd.DataFrame(
        {
            "Year": [2021, 2021, 2021, 2022, 2022],
            "Week": [1, 1, 2, 1, 2],
            "Points": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    out = fix_week_zero(df.copy())
    # 2021 had two week-1 games: rows swapped, first relabeled week 0.
    assert out.at[0, "Week"] == 0 and out.at[0, "Points"] == 20.0
    assert out.at[1, "Week"] == 1 and out.at[1, "Points"] == 10.0
    # 2022 untouched.
    assert out.at[3, "Week"] == 1 and out.at[3, "Points"] == 40.0


def test_predict_missing_deterministic_and_targeted():
    rng = np.random.default_rng(1)
    base = pd.DataFrame(
        {
            "scoreDiff": rng.normal(0, 14, 40),
            "rolling_sum_sacks20": np.concatenate([np.zeros(10), rng.uniform(20, 60, 30)]),
            "rolling_sum_Points20": rng.uniform(400, 700, 40),
        }
    )
    a = base.copy()
    b = base.copy()
    predict_missing(a, ["rolling_sum_sacks20"], 50)
    predict_missing(b, ["rolling_sum_sacks20"], 50)

    # Zeros imputed, deterministic under the same seed, non-fix column untouched.
    assert not a["rolling_sum_sacks20"].isna().any()
    assert (a["rolling_sum_sacks20"][:10] != 0).all()
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(a["rolling_sum_Points20"], base["rolling_sum_Points20"])
    # Known (nonzero) values untouched.
    pd.testing.assert_series_equal(
        a["rolling_sum_sacks20"][10:], base["rolling_sum_sacks20"][10:]
    )
