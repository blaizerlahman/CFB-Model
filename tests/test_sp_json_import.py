"""Importing browser-collected SP+ pages.

The script in scripts/espn_sp_scrape.js runs in a signed-in browser because
ESPN refuses server-side requests. These tests pin the contract between what
it emits and what the importer stores — above all the week shift, since
ratings computed after week N are the ones in hand for week N+1.
"""

import json

import pytest

from cfb_model.data.sp_backfill import import_scraped_json
from cfb_model.data.store import Store


def make_rows(n=130, base=20.0):
    return [{"rank": i + 1, "team": f"Team {i}", "rating": base - i * 0.25} for i in range(n)]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "j.db")
    s.init_schema(["Game Id", "Year", "Week", "School"])
    return s


def write(tmp_path, payload):
    path = tmp_path / "sp.json"
    path.write_text(json.dumps(payload))
    return path


def test_ratings_land_on_the_following_week(store, tmp_path):
    path = write(tmp_path, {"year": 2023, "articles": [
        {"url": "u", "week": 4, "rows": make_rows()},
    ]})
    stored = import_scraped_json(store, path, log=lambda m: None)
    assert stored == {5: 130}, "after-week-4 ratings belong to week 5"
    assert store.sp_weeks(2023) == [5]


def test_preseason_becomes_week_one(store, tmp_path):
    path = write(tmp_path, {"year": 2023, "articles": [
        {"url": "u", "week": 0, "rows": make_rows()},
    ]})
    import_scraped_json(store, path, log=lambda m: None)
    assert store.sp_weeks(2023) == [1]


def test_partial_and_failed_articles_are_skipped(store, tmp_path):
    path = write(tmp_path, {"year": 2023, "articles": [
        {"url": "good", "week": 6, "rows": make_rows()},
        {"url": "paywalled", "week": 7, "rows": make_rows(4)},
        {"url": "broken", "error": "HTTP 403"},
        {"url": "unknown-week", "rows": make_rows()},
    ]})
    messages = []
    stored = import_scraped_json(store, path, log=messages.append)
    assert sorted(stored) == [7], "only the complete article should import"
    assert any("paywalled or partial" in m for m in messages)
    assert any("HTTP 403" in m for m in messages)
    assert any("no week identified" in m for m in messages)


def test_espn_shorthand_names_are_normalised(store, tmp_path):
    rows = [{"rank": 1, "team": "Ohio St."}, {"rank": 2, "team": "Miami-OH"},
            {"rank": 3, "team": "So. Miss"}, {"rank": 4, "team": "App. State"}]
    for i, r in enumerate(rows):
        r["rating"] = 20.0 - i
    rows += make_rows(120, base=0.0)
    path = write(tmp_path, {"year": 2023, "articles": [{"url": "u", "week": 2, "rows": rows}]})
    import_scraped_json(store, path, log=lambda m: None)
    teams = set(store.load_sp(2023, 3)["Team"])
    assert {"Ohio State", "Miami (OH)", "Southern Miss", "App State"} <= teams


def test_year_argument_overrides_payload(store, tmp_path):
    path = write(tmp_path, {"year": 2023, "articles": [{"url": "u", "week": 1, "rows": make_rows()}]})
    import_scraped_json(store, path, year=2021, log=lambda m: None)
    assert store.sp_weeks(2021) == [2]
    assert store.sp_weeks(2023) == []
