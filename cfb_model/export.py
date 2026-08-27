"""Legacy-identical CSV exports for predictions and results.

Reproduces `predsDF.to_csv(path)` from MidSeason exactly: default integer
index, legacy column order, alphabetical team order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfb_model.constants import PREDS_COLUMNS, RESULTS_COLUMNS


def preds_csv_path(output_root: Path, year: int, week: int, day: str) -> Path:
    name = f"preds_{year}w{week}_{day}.csv" if day else f"preds_{year}w{week}.csv"
    return output_root / f"{year} Season" / "Predictions" / f"Week {week}" / name


def results_csv_path(output_root: Path, year: int, week: int) -> Path:
    return output_root / f"{year} Season" / "Results" / f"results_{year}w{week}.csv"


def preds_report_path(output_root: Path, year: int, week: int, day: str) -> Path:
    """The readable report sits beside the CSV, same name, .txt."""
    return preds_csv_path(output_root, year, week, day).with_suffix(".txt")


def write_preds_report(report: str, path: Path) -> Path:
    """Save the run's printed report verbatim, so the week's picks can be read
    back without reconstructing them from the CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.rstrip() + "\n")
    return path


def write_preds_csv(preds: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = preds.sort_values("team").reset_index(drop=True)
    out[list(PREDS_COLUMNS)].to_csv(path)
    return path


def write_results_csv(results: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = results.reset_index(drop=True)
    out[list(RESULTS_COLUMNS)].to_csv(path)
    return path
