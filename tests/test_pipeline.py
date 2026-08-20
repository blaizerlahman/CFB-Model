"""Phase 5 offline tests: week-row construction from API fixtures reproduces
the stored played-game rows."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cfb_model.api.mapping import team_stats_to_wide
from cfb_model.config import get_settings
from cfb_model.pipeline import build_week_rows

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = get_settings()


@pytest.fixture(scope="module")
def week_rows():
    records = json.loads((FIXTURES / "games_teams_2024w10_sample.json").read_text())
    wide = team_stats_to_wide(records, year=2024, week=10)
    return build_week_rows(wide)


def test_week_row_matches_stored_game_row(week_rows):
    stored = pd.read_csv(SETTINGS.legacy_season_dir / "Week 10" / "Georgia_2024week10.csv")
    stored = stored[(stored["Year"] == 2024) & (stored["Week"] == 10)].iloc[0]

    row = week_rows["Georgia"]
    assert len(row) == 1
    row = row.iloc[0]

    assert row["School_opp"] == "Florida"
    for col in ("scoreDiff", "pointTotal", "totalTDs", "totalTDs_opp",
                "Points", "Points_opp", "sacks_opp", "totalYards_opp"):
        assert row[col] == pytest.approx(stored[col]), col
    assert bool(row["Win"]) == bool(stored["Win"])


def test_week_rows_cover_both_sides(week_rows):
    assert "Georgia" in week_rows and "Florida" in week_rows
    g = week_rows["Georgia"].iloc[0]
    f = week_rows["Florida"].iloc[0]
    assert g["Game Id"] == f["Game Id"]
    assert g["scoreDiff"] == -f["scoreDiff"]
