"""tier_records returns the same segment tallies the printed report uses."""

import pandas as pd

from cfb_model.analysis.evaluate import tier_records, tier_breakdown
from cfb_model.constants import THRESH_GOOD, THRESH_GREAT, THRESH_BEST


def _df():
    # 10 games: mix of wins/losses/push across the conviction tiers.
    rows = [
        (1, 0.50), (1, 0.50), (-1, 0.50), (0, 0.50),      # low tier
        (1, THRESH_GOOD), (-1, THRESH_GOOD),               # good+
        (1, THRESH_GREAT), (1, THRESH_GREAT),              # great+
        (1, THRESH_BEST), (-1, THRESH_BEST),               # best+
    ]
    return pd.DataFrame(rows, columns=["result", "successRate"])


def test_overall_counts():
    r = tier_records(_df())["overall"]
    assert (r["wins"], r["losses"], r["pushes"], r["n"]) == (6, 3, 1, 10)
    assert r["win_rate"] == 6 / 9  # pushes excluded from the rate


def test_cumulative_tiers_use_ge_threshold():
    r = tier_records(_df())
    # good = successRate >= .595  -> the good/great/best rows (6 of them)
    assert r["good"]["n"] == 6 and r["good"]["wins"] == 4 and r["good"]["losses"] == 2
    # great = >= .645 -> great + best rows (4)
    assert r["great"]["n"] == 4 and r["great"]["wins"] == 3 and r["great"]["losses"] == 1
    # best = >= .695 -> best rows (2)
    assert r["best"]["n"] == 2 and r["best"]["wins"] == 1 and r["best"]["losses"] == 1


def test_matches_report_overall_wins():
    df = _df()
    rec = tier_records(df)["overall"]
    # tier_breakdown prints "Overall win rate: <rate>, <wins>-<losses>" where
    # losses there include pushes; the win count must agree.
    assert f", {rec['wins']}-" in tier_breakdown(df)


def test_empty_is_safe():
    r = tier_records(pd.DataFrame(columns=["result", "successRate"]))
    assert r["overall"]["n"] == 0 and r["overall"]["win_rate"] is None
