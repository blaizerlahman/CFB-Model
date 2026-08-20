"""Rolling-sum features and stat encoding — verbatim ports from the notebooks.

`lagged_rolling_sum` is `customRollingSum` from p1.5/MidSeason (the corrected
version): positional window [i-window, i) that EXCLUDES the current game.
Two quirks are preserved deliberately because every stored frame and pickled
model was built on them:
  * i < window-1            -> NaN
  * i == window-1           -> empty slice (start index -1 wraps) -> sum = 0.0
  * all-NaN window          -> sum(skipna=True) = 0.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_model.constants import BASE_STATS, PER_GAME_STATS, WINDOWS


def encode_stat(value):
    """Compound stat strings -> decimals ('12-95' -> 12.95, '31:24' -> 31.24).

    Existing convention since p1 (str.replace on the long-format Stat column
    followed by to_numeric(errors='coerce')). Non-strings pass through
    to_numeric coercion unchanged.
    """
    if isinstance(value, str):
        value = value.replace("-", ".").replace(":", ".")
    return pd.to_numeric(value, errors="coerce")


def lagged_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Exact port of customRollingSum(series, window)."""
    result = []
    for i in range(len(series)):
        if i < window - 1:
            result.append(np.nan)
        else:
            # Positional slice; at i == window-1 the start is -1, which wraps
            # to the end of the series and yields an empty slice (sum 0.0).
            result.append(series.iloc[i - window : i].sum(skipna=True))
    return pd.Series(result, index=series.index)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling_sum_{stat}{window} for all BASE_STATS x WINDOWS (p1 createTeam).

    yardsPerPass / yardsPerRushAttempt rolling sums are divided by the window,
    turning them into per-game averages.
    """
    for games in WINDOWS:
        for column in BASE_STATS:
            df["rolling_sum_" + column + str(games)] = lagged_rolling_sum(df[column], games)

    for column in PER_GAME_STATS:
        for games in WINDOWS:
            name = f"rolling_sum_{column}{games}"
            df[name] = df[name] / games

    return df


def rolling_col_window(col: str) -> tuple[str, int]:
    """Parse a rolling column name into (base stat, window).

    Exact port of the suffix parsing in MidSeason's grab*WeekData: the segment
    after 'rolling_sum_' ends in '8' for window 8, otherwise in '20'.
    """
    name = col.split("_")[2]
    if name[-1] == "8":
        return name[:-1], 8
    return name[:-2], 20


def update_last_row_rolling(df: pd.DataFrame, rolling_cols: list[str]) -> pd.DataFrame:
    """Recompute own-side rolling values for the LAST row only (MidSeason).

    Used after appending a new game row (real or sentinel): runs the full
    lagged rolling over the extended frame and writes just the final value.
    """
    for col in rolling_cols:
        column_name, window = rolling_col_window(col)
        rolling = lagged_rolling_sum(df[column_name], window)
        if column_name in PER_GAME_STATS:
            rolling = rolling / window
        df.at[df.index[-1], col] = rolling.iloc[-1]
    return df
