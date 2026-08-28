"""`predict` must not fan out into unbounded API calls.

A mistyped --week (or a long-idle database) previously made the catch-up loop
ingest every intervening week at 3 API calls each, which can exhaust the free
monthly quota in a single run.
"""

import pandas as pd
import pytest

from cfb_model import pipeline
from cfb_model.config import get_settings
from cfb_model.data.store import Store

SETTINGS = get_settings()


class CountingClient:
    """Records calls; returns no games/lines so ingestion is a no-op."""

    def __init__(self):
        self.calls_made = 0
        self.weeks_fetched = []

    def team_game_stats(self, year, week=None, classification=None, **kw):
        self.calls_made += 1
        self.weeks_fetched.append(week)
        return []

    def lines(self, year, week=None, **kw):
        self.calls_made += 1
        return []

    def sp_ratings(self, year, **kw):
        self.calls_made += 1
        return []

    def talent(self, year):
        self.calls_made += 1
        return []

    def calendar(self, year):
        self.calls_made += 1
        return []


def test_far_future_week_refuses_bulk_catchup(monkeypatch):
    store = Store(SETTINGS.db_path)
    client = CountingClient()
    messages: list[str] = []

    # Pretend nothing has been ingested for this season yet.
    monkeypatch.setattr(pipeline, "latest_ingested_week", lambda *a, **k: None)

    pipeline.predict_run(store, client, year=2026, week=99, day="thu",
                         settings=SETTINGS, log=messages.append)

    assert any("Refusing to auto-ingest" in m for m in messages), messages[:5]
    # Nothing should have been ingested; only the slate lookup may hit the API.
    assert client.weeks_fetched == []
    assert client.calls_made <= 3, f"burned {client.calls_made} API calls"


def test_catchup_stops_when_a_week_is_empty(monkeypatch):
    store = Store(SETTINGS.db_path)
    client = CountingClient()
    messages: list[str] = []
    ingested: list[int] = []

    monkeypatch.setattr(pipeline, "latest_ingested_week", lambda *a, **k: 1)

    def fake_ingest(store_, client_, year, week, settings=None):
        ingested.append(week)
        return {"year": year, "week": week, "teams_updated": 0}

    monkeypatch.setattr(pipeline, "ingest_week", fake_ingest)

    pipeline.predict_run(store, client, year=2026, week=5, day="thu",
                         settings=SETTINGS, log=messages.append)

    # Weeks 2,3,4 were pending; the first empty one ends the loop.
    assert ingested == [2], ingested
    assert any("stopping catch-up" in m for m in messages)


def test_within_bounds_catchup_proceeds(monkeypatch):
    store = Store(SETTINGS.db_path)
    client = CountingClient()
    ingested: list[int] = []

    monkeypatch.setattr(pipeline, "latest_ingested_week", lambda *a, **k: 1)
    monkeypatch.setattr(
        pipeline, "ingest_week",
        lambda s, c, year, week, settings=None: (
            ingested.append(week) or {"year": year, "week": week, "teams_updated": 120}
        ),
    )

    pipeline.predict_run(store, client, year=2026, week=4, day="thu",
                         settings=SETTINGS, log=lambda m: None)

    assert ingested == [2, 3]
