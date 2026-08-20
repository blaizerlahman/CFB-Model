"""Talent must never silently become zero.

CFBD publishes talent composites late (2026 was still empty in Aug 2026).
Missing talent would be imputed to 0 by the pipeline's constant imputer —
far outside the trained range (~400-1050) — so the resolver carries the most
recent published season forward instead.
"""

import pandas as pd
import pytest

from cfb_model.config import get_settings
from cfb_model.data.store import Store
from cfb_model.pipeline import resolve_talent

SETTINGS = get_settings()


class FakeClient:
    """Stands in for CfbdClient; returns nothing for any season."""

    def __init__(self):
        self.calls = []

    def talent(self, year):
        self.calls.append(year)
        return []


@pytest.fixture(scope="module")
def store():
    return Store(SETTINGS.db_path)


def test_known_season_comes_from_store(store):
    client = FakeClient()
    talent = resolve_talent(store, client, 2024, log=lambda s: None)
    assert not talent.empty
    assert client.calls == [], "should not hit the API when the store has the season"
    # Service academies legitimately carry 0.0 (247Sports does not rate them);
    # training's predictMissing treats those zeros as missing.
    assert talent["Talent"].max() > 900
    assert (talent["Talent"] >= 0).all()


def test_missing_season_falls_back_to_prior_year(store):
    client = FakeClient()
    warnings: list[str] = []
    talent = resolve_talent(store, client, 2026, log=warnings.append)

    assert not talent.empty, "fallback must supply values rather than nothing"
    assert any("2026" in w and "talent" in w.lower() for w in warnings), \
        "the fallback must announce itself"
    # Real composites carried forward, not an all-zero placeholder.
    assert talent["Talent"].max() > 900
    assert (talent["Talent"] > 0).sum() > 100

    stored_2025 = store.load_talent(2025)
    if not stored_2025.empty:
        pd.testing.assert_frame_equal(
            talent.sort_values("School").reset_index(drop=True),
            stored_2025.sort_values("School").reset_index(drop=True),
        )
