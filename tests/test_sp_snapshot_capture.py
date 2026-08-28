"""In-season SP+ capture must be first-write-wins.

CFBD serves SP+ live during a season, so each week has to be captured while
it is current. A later re-run in the same week (e.g. Sunday, once results are
in) must never replace ratings that were public before kickoff — otherwise
the stored history acquires exactly the lookahead the backfill exists to
avoid.
"""

import pandas as pd
import pytest

from cfb_model.data.store import Store
from cfb_model.pipeline import capture_sp_snapshot

BEFORE = pd.DataFrame({"Team": ["Alabama", "Georgia"], "Rating": [20.0, 22.0]})
AFTER = pd.DataFrame({"Team": ["Alabama", "Georgia"], "Rating": [31.0, 12.0]})


class StubClient:
    def __init__(self, frame):
        self.frame = frame
        self.calls_made = 0

    def sp_ratings(self, year, **kw):
        self.calls_made += 1
        return [{"year": year, "team": r.Team, "rating": r.Rating}
                for r in self.frame.itertuples(index=False)]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema(["Game Id", "Year", "Week", "School"])
    return s


def test_first_capture_is_kept(store):
    capture_sp_snapshot(store, StubClient(BEFORE), 2026, 5, log=lambda m: None)
    messages = []
    capture_sp_snapshot(store, StubClient(AFTER), 2026, 5, log=messages.append)

    stored = store.load_sp(2026, 5).set_index("Team")["Rating"]
    assert stored["Alabama"] == 20.0, "later capture overwrote pre-kickoff ratings"
    assert stored["Georgia"] == 22.0
    assert any("keeping the earlier capture" in m for m in messages)


def test_distinct_weeks_are_independent(store):
    capture_sp_snapshot(store, StubClient(BEFORE), 2026, 5, log=lambda m: None)
    capture_sp_snapshot(store, StubClient(AFTER), 2026, 6, log=lambda m: None)
    assert store.load_sp(2026, 5).set_index("Team")["Rating"]["Alabama"] == 20.0
    assert store.load_sp(2026, 6).set_index("Team")["Rating"]["Alabama"] == 31.0
    assert store.sp_weeks(2026) == [5, 6]


def test_empty_api_response_does_not_erase_existing(store):
    capture_sp_snapshot(store, StubClient(BEFORE), 2026, 5, log=lambda m: None)
    messages = []
    out = capture_sp_snapshot(store, StubClient(pd.DataFrame(columns=["Team", "Rating"])),
                              2026, 5, log=messages.append)
    assert len(out) == 2
    assert store.load_sp(2026, 5).shape[0] == 2
    assert any("no SP+ ratings" in m for m in messages)


def test_backfill_paths_may_still_overwrite(store):
    store.upsert_sp(2026, 5, BEFORE)
    store.upsert_sp(2026, 5, AFTER)  # explicit correction, overwrite defaults True
    assert store.load_sp(2026, 5).set_index("Team")["Rating"]["Alabama"] == 31.0
