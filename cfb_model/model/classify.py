"""Bet classification: bin lookup + tier thresholds (p5/MidSeason)."""

from __future__ import annotations

import pandas as pd

from cfb_model.constants import THRESH_BEST, THRESH_GOOD, THRESH_GREAT


def lookup_success_rate(spread_diff: float, bins: pd.DataFrame) -> float | None:
    """Bin where lowerBin <= spreadDiff < upperBin (exact legacy predicate)."""
    if pd.isna(spread_diff):
        return None
    hit = bins[(bins["lowerBin"] <= spread_diff) & (bins["upperBin"] > spread_diff)]
    if hit.empty:
        return None
    return float(hit.iloc[0]["successRate"])


def tier(success_rate: float) -> str:
    """Exclusive print-report buckets (elif chain in printPredictions)."""
    if success_rate < THRESH_GOOD:
        return "normal"
    if success_rate <= THRESH_GREAT:
        return "good"
    if success_rate <= THRESH_BEST:
        return "great"
    return "best"
