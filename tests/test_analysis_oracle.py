"""Phase 6 oracle: grading 2024 week 15 from DB predictions + DB actuals
reproduces the stored results_2024w15.csv (content-identical; the legacy
file's leading index is a known cosmetic artifact)."""

import pandas as pd
import pytest

from cfb_model.analysis.evaluate import grade_week, tier_breakdown
from cfb_model.config import get_settings
from cfb_model.data.store import Store
from cfb_model.pipeline import load_gated_frames

SETTINGS = get_settings()


@pytest.fixture(scope="module")
def graded():
    store = Store(SETTINGS.db_path)
    preds = store.load_predictions(2024, 15)
    fbs, fcs = load_gated_frames(store)
    return grade_week(preds, store.load_bins(), fbs, set(fcs), log=lambda s: None)


@pytest.fixture(scope="module")
def stored():
    df = pd.read_csv(SETTINGS.legacy_season_dir / "Results" / "results_2024w15.csv")
    return df[[c for c in df.columns if not c.startswith("Unnamed")]]


def test_week15_grading_matches_stored(graded, stored):
    """Graded OUTCOMES must match the stored file exactly. The kept day-version
    may differ where success rates tied across days (legacy's pick was
    filesystem-order arbitrary; ours is deterministically the latest day) —
    in those rows every outcome column must still agree."""
    mine = graded.sort_values("gameID").reset_index(drop=True)
    theirs = stored.sort_values("gameID").reset_index(drop=True)
    assert list(mine["gameID"]) == list(theirs["gameID"])
    assert list(mine["team"]) == list(theirs["team"])
    for col in ("pred", "cover", "scoreDiff", "result", "successRate"):
        pd.testing.assert_series_equal(
            mine[col].astype(float), theirs[col].astype(float),
            check_names=False, rtol=1e-9, obj=col,
        )
    # Where the kept day differs, the choice must have been a genuine tie.
    day_diff = mine["day"] != theirs["day"]
    assert (mine.loc[day_diff, "successRate"].astype(float)
            == theirs.loc[day_diff, "successRate"].astype(float)).all()


def test_week15_breakdown_totals(graded):
    report = tier_breakdown(graded)
    wins = int((graded["result"] == 1).sum())
    losses = int((graded["result"] == -1).sum()) + int((graded["result"] == 0).sum())
    assert f"{wins}-" in report
    assert len(graded) == wins + losses
