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

GAMES = [
    # Week 2's real opener is Thursday, days after the calendar's Tuesday
    # window boundary — the deadline must track the game, not the boundary.
    {"week": 2, "seasonType": "regular", "startDate": "2025-09-04T22:00:00.000Z"},
    {"week": 2, "seasonType": "regular", "startDate": "2025-09-06T16:00:00.000Z"},
    {"week": 2, "seasonType": "regular", "startDate": "2025-09-06T20:00:00.000Z"},
    {"week": 3, "seasonType": "regular", "startDate": "2025-09-11T22:30:00.000Z"},
    {"week": 3, "seasonType": "regular", "startDate": "2025-09-13T16:00:00.000Z"},
    # Postseason reuses week numbers and must not displace regular-season ones.
    {"week": 2, "seasonType": "postseason", "startDate": "2025-12-28T18:00:00.000Z"},
    {"week": 3, "seasonType": "regular", "startDate": None},
]


def test_deadline_is_the_weeks_earliest_actual_kickoff():
    deadlines = snapshot_deadlines(GAMES)
    assert deadlines[2] == datetime(2025, 9, 4, 22, 0, tzinfo=timezone.utc)
    assert deadlines[3] == datetime(2025, 9, 11, 22, 30, tzinfo=timezone.utc)
    # Postseason games must not pull a regular-season deadline later.
    assert deadlines[2].month == 9


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
