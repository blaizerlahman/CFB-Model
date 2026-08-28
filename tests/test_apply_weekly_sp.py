"""Applying weekly SP+ to a season's game rows.

Importing snapshots only fills `sp_ratings`; the model reads SP/SP_opp off
each game row. This step moves them across, and must put the rating from the
row's OWN week on that row — anything later would be information the game
had not produced yet.
"""

import pandas as pd
import pytest

from cfb_model.data.apply_sp import apply_weekly_sp, sp_variety
from cfb_model.data.store import Store

COLUMNS = ["Game Id", "Year", "Week", "School", "School_opp", "SP", "SP_opp"]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "a.db")
    s.init_schema(COLUMNS)
    frame = pd.DataFrame([
        {"Game Id": 1.0, "Year": 2023, "Week": 1, "School": "Alabama",
         "School_opp": "Georgia", "SP": 30.0, "SP_opp": 30.0},
        {"Game Id": 2.0, "Year": 2023, "Week": 2, "School": "Alabama",
         "School_opp": "Georgia", "SP": 30.0, "SP_opp": 30.0},
        {"Game Id": 3.0, "Year": 2023, "Week": 9, "School": "Alabama",
         "School_opp": "Georgia", "SP": 30.0, "SP_opp": 30.0},
    ])
    s.replace_team_frame("Alabama", "fbs", frame)
    s.upsert_sp(2023, 1, pd.DataFrame({"Team": ["Alabama", "Georgia"], "Rating": [10.0, 11.0]}))
    s.upsert_sp(2023, 2, pd.DataFrame({"Team": ["Alabama", "Georgia"], "Rating": [20.0, 21.0]}))
    return s


def test_each_row_gets_its_own_week(store):
    apply_weekly_sp(store, 2023, log=lambda m: None)
    got = store.load_team_frame("Alabama").set_index("Week")
    assert got.loc[1, "SP"] == 10.0 and got.loc[1, "SP_opp"] == 11.0
    assert got.loc[2, "SP"] == 20.0 and got.loc[2, "SP_opp"] == 21.0


def test_weeks_without_a_snapshot_are_left_alone(store):
    result = apply_weekly_sp(store, 2023, log=lambda m: None)
    got = store.load_team_frame("Alabama").set_index("Week")
    assert got.loc[9, "SP"] == 30.0, "week 9 has no snapshot and must not be guessed at"
    assert result["skipped_no_week"] == 1
    assert result["updated"] == 2


def test_is_idempotent(store):
    apply_weekly_sp(store, 2023, log=lambda m: None)
    first = store.load_team_frame("Alabama")
    apply_weekly_sp(store, 2023, log=lambda m: None)
    pd.testing.assert_frame_equal(first, store.load_team_frame("Alabama"))


def test_requires_snapshots(store):
    with pytest.raises(ValueError):
        apply_weekly_sp(store, 2019, log=lambda m: None)


def test_variety_reports_the_change(store):
    assert sp_variety(store, 2023).distinct_sp.iloc[0] == 1
    apply_weekly_sp(store, 2023, log=lambda m: None)
    assert sp_variety(store, 2023).distinct_sp.iloc[0] > 1
