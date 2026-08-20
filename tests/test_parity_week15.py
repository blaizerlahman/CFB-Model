"""Phase 4 critical checkpoint: replay the 2024 week-15 prediction run purely
from DB state (through week 14) + the stored week-15 lines/SP snapshot, using
the ORIGINAL frozen pickles, and require row-level equality with the stored
preds_2024w15_tue.csv.

The old pickles are deterministic once fitted, so exact parity is achievable;
any diff is a pipeline bug (modulo the approved bug-fixes, none of which
touch this week's slate).
"""

import glob
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from cfb_model.api.mapping import BETTING_COLUMNS
from cfb_model.config import get_settings
from cfb_model.data.store import Store
from cfb_model.features.assemble import feature_columns
from cfb_model.model.predict import build_upcoming_frames, predict_week

SETTINGS = get_settings()
YEAR, WEEK, DAY = 2024, 15, "tue"

# The notebook's hardcoded list, used verbatim for parity replication only.
LEGACY_SKIP = frozenset(
    ["Jacksonville State", "James Madison", "Kennesaw State",
     "Coastal Carolina", "Liberty", "Sam Houston State"]
)


def truncate_through_week14(df: pd.DataFrame) -> pd.DataFrame:
    keep = (df["Year"] < YEAR) | ((df["Year"] == YEAR) & (df["Week"] < WEEK))
    return df[keep].reset_index(drop=True)


@pytest.fixture(scope="module")
def oracle() -> pd.DataFrame:
    path = SETTINGS.legacy_season_dir / "Predictions" / f"Week {WEEK}" / f"preds_{YEAR}w{WEEK}_{DAY}.csv"
    df = pd.read_csv(path)
    return df[[c for c in df.columns if not c.startswith("Unnamed")]]


@pytest.fixture(scope="module")
def replay(oracle):
    store = Store(SETTINGS.db_path)
    fbs_full = store.load_all_team_frames("fbs")
    fcs_full = store.load_all_team_frames("fcs")

    # Reconstruct the tue preferred-line rows from the oracle preds plus
    # home/away orientation taken from the played week-15 rows in the frames.
    line_rows = []
    for _, p in oracle.iterrows():
        owner_frame = fbs_full[p["team"]]
        played = owner_frame[owner_frame["Id"] == p["gameID"]]
        assert len(played) == 1, f"week-15 game row missing for {p['team']}"
        played = played.iloc[0]
        raw_spread = -p["spread"] if played["HomeTeam"] == p["team"] else p["spread"]
        row = {c: np.nan for c in BETTING_COLUMNS}
        row.update({
            "Id": p["gameID"],
            "HomeTeam": played["HomeTeam"],
            "AwayTeam": played["AwayTeam"],
            "LineProvider": played["LineProvider"],
            "Spread": raw_spread,
        })
        line_rows.append(row)
    preferred = pd.DataFrame(line_rows, columns=list(BETTING_COLUMNS))

    fbs = {k: truncate_through_week14(v) for k, v in fbs_full.items()}
    fcs = {k: truncate_through_week14(v) for k, v in fcs_full.items()}
    # Legacy load gates (cells 2/4): >=56 rows FBS, >=10 FCS.
    fbs = {k: v for k, v in fbs.items() if len(v) >= 56}
    fcs = {k: v for k, v in fcs.items() if len(v) >= 10}

    sp = store.load_sp(YEAR, WEEK)
    talent = store.load_talent(YEAR)

    fbs, fcs = build_upcoming_frames(fbs, fcs, preferred, sp, talent, YEAR, WEEK)

    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for path in glob.glob(str(SETTINGS.legacy_team_models / "*2024*.pkl")):
            key = Path(path).name.split("_model")[0].replace("_", " ")
            models[key] = joblib.load(path)

    features = feature_columns(next(iter(fbs.values())))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, statuses = predict_week(models, fbs, features, skip_teams=LEGACY_SKIP)
    return preds, statuses


def test_game_set_matches(replay, oracle):
    preds, _ = replay
    assert set(preds["gameID"]) == set(oracle["gameID"])
    assert len(preds) == len(oracle)


def test_rowwise_parity(replay, oracle):
    preds, _ = replay
    mine = preds.sort_values("team").reset_index(drop=True)
    theirs = oracle.sort_values("team").reset_index(drop=True)

    mismatches = []
    for i in range(len(theirs)):
        for col in ("team", "oppTeam"):
            if mine.at[i, col] != theirs.at[i, col]:
                mismatches.append((theirs.at[i, "team"], col, mine.at[i, col], theirs.at[i, col]))
        for col in ("pred", "spread", "spreadDiff", "cover", "gameID"):
            a, b = mine.at[i, col], theirs.at[i, col]
            if not (pd.isna(a) and pd.isna(b)) and a != pytest.approx(b):
                mismatches.append((theirs.at[i, "team"], col, a, b))
    assert not mismatches, f"parity mismatches (team, col, mine, oracle): {mismatches}"


def test_dedup_ownership(replay, oracle):
    _, statuses = replay
    already = {t for t, s in statuses.items() if s.startswith("Already predicted")}
    # Every oracle oppTeam that has a model must have deferred to the owner.
    owners = set(oracle["team"])
    for opp in oracle["oppTeam"]:
        if opp in already:
            assert opp not in owners
