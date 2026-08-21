"""The rebuilt calibration must keep p5's binning arithmetic exactly; only the
source of games changes (leak-free replays instead of season-final-SP+ ones).
"""

import numpy as np
import pandas as pd
import pytest

from cfb_model.analysis.bins import bins_from_replays, success_by_deviation
from cfb_model.config import get_settings
from cfb_model.data.store import Store


def test_bin_edges_are_half_point_near_zero_and_wide_outside():
    diffs = list(np.arange(-20.0, 20.0, 0.5))
    results = [1] * len(diffs)
    bins = success_by_deviation(diffs, results)

    widths = (bins["upperBin"] - bins["lowerBin"]).round(3)
    near = bins[(bins["lowerBin"] >= -5.5) & (bins["upperBin"] <= 3.5)]
    assert set(widths[near.index]) == {0.5}, "bins near zero must stay 0.5 wide"
    far = bins[bins["lowerBin"] >= 3.5]
    assert set(widths[far.index]) <= {2.5}, "bins away from zero must be 2.5 wide"
    # A bin covers [lower, upper) and the table is contiguous and ordered.
    assert (bins["lowerBin"].shift(-1).dropna() == bins["upperBin"][:-1]).all()


def test_rate_is_wins_over_all_games_in_bin():
    # Four games in the same 0.5-wide bin: 2 wins, 1 loss, 1 push.
    diffs = [1.0, 1.0, 1.0, 1.0]
    results = [1, 1, -1, 0]
    bins = success_by_deviation(diffs, results)
    hit = bins[(bins["lowerBin"] <= 1.0) & (bins["upperBin"] > 1.0)].iloc[0]
    assert hit["gameCount"] == 4
    # Pushes and losses both count against, as in p5.
    assert hit["successRate"] == pytest.approx(0.5)


def test_zero_differential_games_are_excluded():
    diffs = [0.0, 0.0, 0.0, 2.0, -8.0]
    results = [1, 1, 1, 1, -1]
    bins = success_by_deviation(diffs, results)
    covering_zero = bins[(bins["lowerBin"] <= 0.0) & (bins["upperBin"] > 0.0)]
    assert covering_zero["gameCount"].sum() == 0, "toss-ups must not be binned"
    assert bins["gameCount"].sum() == 2


def test_lookup_predicate_matches_stored_table_shape():
    from cfb_model.model.classify import lookup_success_rate

    diffs = [4.0] * 10 + [-9.0] * 10
    results = [1] * 7 + [-1] * 3 + [1] * 2 + [-1] * 8
    bins = success_by_deviation(diffs, results)
    assert lookup_success_rate(4.0, bins) == pytest.approx(0.7)
    assert lookup_success_rate(-9.0, bins) == pytest.approx(0.2)
    assert lookup_success_rate(500.0, bins) is None


def test_bins_from_replays_ignores_incomplete_rows():
    replays = pd.DataFrame({
        "spreadDiff": [3.0, 3.0, np.nan, 3.0],
        "result": [1, -1, 1, np.nan],
    })
    bins = bins_from_replays(replays)
    assert bins["gameCount"].sum() == 2


def test_named_bin_sets_do_not_disturb_the_legacy_table(tmp_path):
    store = Store(tmp_path / "b.db")
    store.init_schema(["Game Id", "Year", "Week", "School"])
    legacy = pd.DataFrame({"lowerBin": [-1.0], "upperBin": [1.0],
                           "successRate": [0.5], "gameCount": [10.0]})
    store.upsert_bins(legacy)

    rebuilt = pd.DataFrame({"lowerBin": [-1.0], "upperBin": [1.0],
                            "successRate": [0.61], "gameCount": [99.0]})
    store.upsert_bin_set("leakfree", rebuilt)

    assert store.load_bin_set("legacy").iloc[0]["successRate"] == pytest.approx(0.5)
    assert store.load_bin_set("leakfree").iloc[0]["successRate"] == pytest.approx(0.61)
    assert "leakfree" in store.bin_set_names()


def test_shipped_legacy_table_is_untouched_on_disk():
    path = get_settings().legacy_bin_csv
    df = pd.read_csv(path)
    assert len(df) == 59, "the original Bin_Data.csv must not be rewritten"
