"""Teams with two games in one week.

CFBD's week 1 spans about ten days, so a side that opens in "week 0" and
plays again the next weekend has both games filed under week 1 — 8 teams in
the 2026 opener. The pipeline used to look only at each team's last upcoming
row, which silently dropped the earlier game and left its features unfilled.
"""

import numpy as np
import pandas as pd
import pytest

from cfb_model.api.mapping import BETTING_COLUMNS
from cfb_model.model.predict import (
    build_upcoming_frames,
    games_by_id,
    predict_week,
    upcoming_rows,
)

ROLLING = ["rolling_sum_Points20", "rolling_sum_Points8"]


def played_frame(team, points):
    rows = []
    for i, p in enumerate(points):
        row = {c: np.nan for c in BETTING_COLUMNS}
        row.update({"Game Id": float(i + 1), "School": team, "School_opp": "Someone",
                    "Year": 2026.0, "Week": float(i + 1), "Points": float(p),
                    "talent": 700.0, "talent_opp": 700.0, "SP": 5.0, "SP_opp": 5.0})
        for c in ROLLING:
            row[c] = np.nan
            row[c + "_opp"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def line_row(game_id, home, away, spread):
    row = {c: np.nan for c in BETTING_COLUMNS}
    row.update({"Id": float(game_id), "HomeTeam": home, "AwayTeam": away,
                "Spread": float(spread), "LineProvider": "test"})
    return row


@pytest.fixture
def two_game_week():
    fbs = {"Alpha": played_frame("Alpha", range(10, 40)),
           "Bravo": played_frame("Bravo", range(20, 50)),
           "Charlie": played_frame("Charlie", range(15, 45))}
    # Alpha plays twice in the same week, as a week-0 opener plus a weekend game.
    preferred = pd.DataFrame([line_row(901, "Alpha", "Bravo", -3.0),
                              line_row(902, "Charlie", "Alpha", -7.0)],
                             columns=list(BETTING_COLUMNS))
    sp = pd.DataFrame({"Team": ["Alpha", "Bravo", "Charlie"], "Rating": [5.0, 6.0, 7.0]})
    talent = pd.DataFrame({"School": ["Alpha", "Bravo", "Charlie"], "Talent": [700.0] * 3})
    return build_upcoming_frames(fbs, {}, preferred, sp, talent, 2026, 5)


def test_both_upcoming_rows_are_kept(two_game_week):
    fbs, _ = two_game_week
    assert len(upcoming_rows(fbs["Alpha"])) == 2
    assert len(upcoming_rows(fbs["Bravo"])) == 1


def test_each_upcoming_row_sees_only_played_games(two_game_week):
    """Both of Alpha's rows must show the same trailing window: neither game
    has happened, so one must not count the other as a played game."""
    fbs, _ = two_game_week
    rows = upcoming_rows(fbs["Alpha"])
    values = [fbs["Alpha"].loc[i, "rolling_sum_Points20"] for i in rows]
    assert values[0] == values[1], f"windows differ: {values}"
    played = fbs["Alpha"][fbs["Alpha"]["School"].notna()]["Points"]
    assert values[0] == pytest.approx(played.iloc[-20:].sum())


def test_features_are_filled_on_every_upcoming_row(two_game_week):
    fbs, _ = two_game_week
    for idx in upcoming_rows(fbs["Alpha"]):
        row = fbs["Alpha"].loc[idx]
        assert not pd.isna(row["rolling_sum_Points20"])
        assert not pd.isna(row["SP"]), "SP must be filled on both rows"
        assert not pd.isna(row["talent"])


def test_games_by_id_pairs_the_right_rows(two_game_week):
    fbs, _ = two_game_week
    games = games_by_id(fbs)
    assert set(games) == {901.0, 902.0}
    assert set(games[901.0]) == {"Alpha", "Bravo"}
    assert set(games[902.0]) == {"Alpha", "Charlie"}


def test_both_games_are_predicted(two_game_week):
    """The bug: only Alpha's later game came back."""
    fbs, _ = two_game_week

    class Flat:
        def predict(self, X):
            return np.array([7.0])

    models = {t: Flat() for t in fbs}
    preds, _ = predict_week(models, fbs, features=ROLLING)
    assert set(preds["gameID"]) == {901.0, 902.0}
    assert len(preds) == 2, "each game predicted exactly once"


def test_spread_orientation_survives(two_game_week):
    fbs, _ = two_game_week
    # Alpha hosts 901 (spread flipped) and visits 902 (raw).
    rows = {fbs["Alpha"].loc[i, "Id"]: fbs["Alpha"].loc[i, "Spread"] for i in upcoming_rows(fbs["Alpha"])}
    assert rows[901.0] == pytest.approx(3.0)
    assert rows[902.0] == pytest.approx(-7.0)
