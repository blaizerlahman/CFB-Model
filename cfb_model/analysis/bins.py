"""Rebuild the spread-differential success-rate table.

The shipped `Bin Data/Bin_Data.csv` was produced by p5 from replays of
2018-2023 in which every game carried its season's FINAL SP+ rating — future
information relative to the game being predicted. Its success rates are
therefore inflated, and on clean data its tiers stop ranking anything.

This module keeps p5's binning arithmetic exactly and changes only the source
of the games: replays that use nothing a bettor would not have had before
kickoff. Only seasons with genuine weekly SP+ snapshots qualify, which today
means 2024 and 2025.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfb_model.analysis.backtest import run_backtest
from cfb_model.config import Settings, get_settings
from cfb_model.data.store import Store

BIN_COLUMNS = ["lowerBin", "upperBin", "successRate", "gameCount"]


def success_by_deviation(spread_diffs, results) -> pd.DataFrame:
    """Exact port of p5's successByDeviation.

    Bins are 0.5 wide between -5.5 and 3.5 and 2.5 wide outside it, laid out
    from the observed range. A bin covers [lower, upper). Games whose
    differential is exactly 0 (toss-ups) are excluded, and the rate is wins
    divided by every game in the bin, so pushes and losses both count against.
    """
    diffs = np.asarray(spread_diffs, dtype=float)
    outcomes = np.asarray(results, dtype=float)
    pairs = np.column_stack([diffs, outcomes])

    unique_diffs = np.unique(pairs[:, 0])
    diff_probs = []
    for diff in unique_diffs:
        mask = pairs[:, 0] == diff
        count = int(np.sum(mask))
        if count > 0:
            diff_probs.append([diff, np.sum(pairs[mask, 1] == 1) / count, count])
    diff_probs = np.array(diff_probs)
    diff_probs = diff_probs[~np.isnan(diff_probs[:, 1])]

    bins = np.concatenate([
        np.arange(-5.5, 3.5, 0.5),
        np.arange(3.5, np.max(diff_probs[:, 0]) + 2.5, 2.5),
        np.arange(np.min(diff_probs[:, 0]), -5.5, 2.5),
    ])
    bins = np.unique(bins)
    bins.sort()

    bin_indices = np.digitize(pairs[:, 0], bins)
    non_zero = pairs[:, 0] != 0

    rows = []
    for i in range(1, len(bins)):
        selected = (bin_indices == i) & non_zero
        values = pairs[selected, 1]
        count = int(np.sum(selected))
        rate = (np.sum(values == 1) / count) if values.size > 0 else np.nan
        rows.append([bins[i - 1], bins[i], rate, float(count)])

    return pd.DataFrame(rows, columns=BIN_COLUMNS)


def collect_replays(store: Store, seasons: list[int], reps: int, seed_base: int = 1000,
                    settings: Settings | None = None, log=print) -> pd.DataFrame:
    """Run `reps` leak-free replays per season and pool every graded game.

    Repetition mirrors p5, which rebuilt its models ten times over: the
    imputation step is stochastic, so each pass reseeds it and yields a
    different draw of predictions over the same games.
    """
    settings = settings or get_settings()
    frames = []
    for season in seasons:
        for rep in range(reps):
            seed = seed_base + rep
            log(f"  replay {season} rep {rep + 1}/{reps} (seed {seed})...")
            results = run_backtest(store, season, random_state=seed, settings=settings,
                                   log=lambda m: None, label=f"binrep{seed}")
            if results.empty:
                continue
            keep = results[["spreadDiff", "result"]].copy()
            keep["season"] = season
            keep["rep"] = rep
            frames.append(keep)
    if not frames:
        return pd.DataFrame(columns=["spreadDiff", "result", "season", "rep"])
    return pd.concat(frames, ignore_index=True)


def bins_from_replays(replays: pd.DataFrame) -> pd.DataFrame:
    sub = replays.dropna(subset=["spreadDiff", "result"])
    return success_by_deviation(sub["spreadDiff"].values, sub["result"].values)


def summarize(bins: pd.DataFrame, min_games: int = 30) -> str:
    usable = bins[bins["gameCount"] >= min_games].dropna(subset=["successRate"])
    lines = [
        f"bins: {len(bins)} total, {len(usable)} with >= {min_games} games",
        f"games pooled: {int(bins['gameCount'].sum())}",
    ]
    if not usable.empty:
        lines.append(f"success rate range (usable bins): "
                     f"{usable['successRate'].min():.3f} - {usable['successRate'].max():.3f}")
    return "\n".join(lines)
