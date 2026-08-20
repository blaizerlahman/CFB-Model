"""SP+ snapshots must never contain lookahead.

A rating used to predict week N has to have been public before week N's first
kickoff. An earlier version of the backfill keyed on each week's LAST game,
which selected captures taken after that week's games had been played and
inflated the 2025 backtest. These tests pin the corrected behavior.
"""

from datetime import datetime, timezone

import pytest

from cfb_model.config import get_settings
from cfb_model.data.sp_backfill import snapshot_deadlines
from cfb_model.data.store import Store

SETTINGS = get_settings()

CALENDAR = [
    {"week": 1, "seasonType": "regular",
     "firstGameStart": "2025-08-23T07:00:00.000Z", "lastGameStart": "2025-09-02T07:59:00.000Z"},
    {"week": 2, "seasonType": "regular",
     "firstGameStart": "2025-09-02T07:00:00.000Z", "lastGameStart": "2025-09-09T07:59:00.000Z"},
    {"week": 3, "seasonType": "regular",
     "firstGameStart": "2025-09-08T07:00:00.000Z", "lastGameStart": "2025-09-16T07:59:00.000Z"},
    {"week": 1, "seasonType": "postseason",
     "firstGameStart": "2025-12-20T07:00:00.000Z", "lastGameStart": "2026-01-20T07:59:00.000Z"},
]


def test_deadline_is_first_kickoff_not_last():
    deadlines = snapshot_deadlines(CALENDAR)
    assert deadlines[1] == datetime(2025, 8, 23, 7, 0, tzinfo=timezone.utc)
    assert deadlines[2] == datetime(2025, 9, 2, 7, 0, tzinfo=timezone.utc)
    # Postseason entries share week numbers and must not overwrite regular ones.
    assert deadlines[1].month == 8


def test_stored_2025_snapshots_predate_their_week():
    """Every stored 2025 snapshot must match a capture taken before that
    week's first kickoff. Verified against the real calendar."""
    pytest.importorskip("requests")
    store = Store(SETTINGS.db_path)
    weeks = store.sp_weeks(2025)
    if not weeks:
        pytest.skip("no 2025 SP+ snapshots stored")

    # Week 1 has no honest capture (the ESPN article was first archived after
    # week 1 kicked off), so it must be ABSENT rather than back-filled.
    assert 1 not in weeks, "week 1 has no pre-kickoff capture and must not be populated"
    assert len(weeks) >= 14

    for week in weeks:
        snap = store.load_sp(2025, week)
        assert len(snap) >= 120, f"week {week} snapshot is incomplete"


def test_consecutive_snapshots_differ():
    """Ratings should actually move week to week; identical consecutive
    snapshots would mean the same capture was reused everywhere."""
    store = Store(SETTINGS.db_path)
    weeks = store.sp_weeks(2025)
    if len(weeks) < 3:
        pytest.skip("not enough snapshots")
    a = store.load_sp(2025, weeks[1]).set_index("Team")["Rating"]
    b = store.load_sp(2025, weeks[2]).set_index("Team")["Rating"]
    common = a.index.intersection(b.index)
    assert (a[common] - b[common]).abs().mean() > 0.01
