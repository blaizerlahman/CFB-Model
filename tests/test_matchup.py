"""Phase 8: matchup predictions work off current DB state and must never
leak into the success-analysis path."""

import warnings

import pytest

from cfb_model.config import get_settings
from cfb_model.data.store import Store
from cfb_model.model.predict import predict_matchup
from cfb_model.pipeline import load_models

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def models():
    loaded = load_models(SETTINGS, 2024)
    if not loaded:
        pytest.skip("no 2024 models available")
    return loaded


@pytest.fixture(scope="module")
def store():
    return Store(SETTINGS.db_path)


def test_matchup_returns_half_point_rounded_prediction(store, models):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = predict_matchup(store, models, "Alabama", "Georgia")
    pred = result["predicted_score_diff"]
    assert (pred * 2) == int(pred * 2), "prediction must round to a half point"
    assert {result["team"], result["opponent"]} == {"Alabama", "Georgia"}
    assert result["spread"] is None and result["spread_diff"] is None


def test_matchup_with_spread_classifies(store, models):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = predict_matchup(store, models, "Alabama", "Georgia", spread=-3.5)
    assert result["spread"] == pytest.approx(-3.5 if result["team"] == "Alabama" else 3.5)
    assert result["cover"] in (-1, 0, 1)
    assert result["spread_diff"] == pytest.approx(
        result["predicted_score_diff"] - result["spread"]
    )
    if result.get("success_rate") is not None:
        assert 0 < result["success_rate"] < 1
        assert result["tier"] in ("normal", "good", "great", "best")


def test_matchups_never_enter_predictions_or_results(store, models):
    """The whole point of the separate table: running matchups must not change
    what `analyze` sees."""
    def counts():
        return (
            store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            store.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0],
        )

    before = counts()
    matchups_before = store.conn.execute("SELECT COUNT(*) FROM matchup_queries").fetchone()[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predict_matchup(store, models, "Alabama", "Georgia", spread=-7.0)

    assert counts() == before, "matchup leaked into predictions/results"
    matchups_after = store.conn.execute("SELECT COUNT(*) FROM matchup_queries").fetchone()[0]
    assert matchups_after == matchups_before + 1


def test_unknown_team_raises(store, models):
    with pytest.raises(ValueError):
        predict_matchup(store, models, "Not A Real Team", "Also Fake")
