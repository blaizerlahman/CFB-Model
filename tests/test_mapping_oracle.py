"""Phase 2 oracle tests: API-response fixtures (captured live 2026-08-20) map
to frames matching the stored legacy exporter data field-by-field.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from cfb_model.api.mapping import (
    current_week,
    lines_to_frame,
    sp_to_frame,
    talent_to_frame,
    team_stats_to_wide,
)
from cfb_model.config import get_settings

FIXTURES = Path(__file__).parent / "fixtures"
SETTINGS = get_settings()

GEORGIA_FLORIDA_2024_W10 = 401628408


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture(scope="module")
def wide():
    return team_stats_to_wide(load_fixture("games_teams_2024w10_sample.json"), year=2024, week=10)


@pytest.fixture(scope="module")
def georgia_stored():
    df = pd.read_csv(SETTINGS.legacy_season_dir / "Week 10" / "Georgia_2024week10.csv")
    return df[(df["Year"] == 2024) & (df["Week"] == 10)].iloc[0]


def test_wide_frame_identity_columns(wide):
    row = wide[(wide["Game Id"] == GEORGIA_FLORIDA_2024_W10) & (wide["School"] == "Georgia")]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["HomeAway"] == "home"
    assert row["Points"] == 34
    assert row["Week"] == 10 and row["Year"] == 2024
    assert row["Conference"] == "SEC"


def test_wide_frame_stats_match_stored_exporter_row(wide, georgia_stored):
    row = wide[(wide["Game Id"] == GEORGIA_FLORIDA_2024_W10) & (wide["School"] == "Georgia")].iloc[0]
    # Cross-check every base stat present in both against the legacy frame,
    # including the compound-encoded totalPenaltiesYards ('5-48' -> 5.48).
    for stat in ("sacks", "qbHurries", "totalPenaltiesYards", "firstDowns",
                 "totalYards", "turnovers", "yardsPerPass", "rushingYards",
                 "netPassingYards", "passingTDs", "rushingTDs", "interceptions"):
        assert row[stat] == pytest.approx(georgia_stored[stat]), stat


def test_name_normalization_applied(wide):
    schools = set(wide["School"])
    # The API already uses 'App State'; normalization must keep it canonical
    # and never emit the long-form names.
    assert "Appalachian State" not in schools


def test_lines_match_stored_betting_columns(georgia_stored):
    frame = lines_to_frame(load_fixture("lines_2024w10_sample.json"))
    dk = frame[(frame["Id"] == GEORGIA_FLORIDA_2024_W10) & (frame["LineProvider"] == "DraftKings")]
    assert len(dk) == 1
    dk = dk.iloc[0]
    # Raw API spread is home-perspective; the stored frame flipped Spread
    # (home team) but left OpeningSpread raw.
    assert dk["Spread"] == pytest.approx(-georgia_stored["Spread"])
    assert dk["OverUnder"] == pytest.approx(georgia_stored["OverUnder"])
    assert dk["OpeningSpread"] == pytest.approx(georgia_stored["OpeningSpread"])
    assert dk["HomeMoneyline"] == pytest.approx(georgia_stored["HomeMoneyline"])
    assert dk["HomeTeam"] == "Georgia" and dk["AwayTeam"] == "Florida"
    assert dk["FormattedSpread"] == georgia_stored["FormattedSpread"]


def test_lines_provider_coverage():
    frame = lines_to_frame(load_fixture("lines_2024w10_sample.json"))
    providers = set(frame["LineProvider"])
    assert "ESPN Bet" in providers and "DraftKings" in providers


def test_talent_matches_stored_file():
    api = talent_to_frame(load_fixture("talent_2024.json"))
    stored = pd.read_csv(SETTINGS.legacy_talent_dir / "talent_2024.csv")
    merged = stored.merge(api, on="School", suffixes=("_stored", "_api"))
    # Every stored school must resolve, with identical talent values.
    assert len(merged) == len(stored)
    pd.testing.assert_series_equal(
        merged["Talent_stored"], merged["Talent_api"], check_names=False, rtol=1e-9
    )


def test_sp_frame_shape():
    frame = sp_to_frame(load_fixture("sp_2024_sample.json"))
    assert list(frame.columns) == ["Year", "Team", "Rating"]
    assert frame["Rating"].notna().all()
    assert (frame["Year"] == 2024).all()


def test_current_week_detection():
    cal = load_fixture("calendar_2026.json")
    in_week1 = datetime(2026, 8, 30, tzinfo=timezone.utc)
    assert current_week(cal, in_week1) == (2026, 1)
    preseason = datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert current_week(cal, preseason) == (2026, 1)  # next upcoming
    offseason = datetime(2027, 3, 1, tzinfo=timezone.utc)
    assert current_week(cal, offseason) is None
