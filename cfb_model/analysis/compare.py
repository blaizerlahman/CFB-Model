"""Tier-segmented comparison of backtest runs.

Every row reports the record and hit rate for all graded bets and for each
confidence tier, next to the -110 break-even line, so a variant can be judged
on the tiers it actually claims an edge in rather than on one pooled number.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfb_model.constants import THRESH_BEST, THRESH_GOOD, THRESH_GREAT

BREAK_EVEN = 0.5238  # -110 odds

TIERS = (
    ("all", None),
    ("good", THRESH_GOOD),
    ("great", THRESH_GREAT),
    ("best", THRESH_BEST),
)


def tier_row(results: pd.DataFrame, label: str) -> dict:
    row: dict = {"run": label}
    for name, threshold in TIERS:
        sub = results if threshold is None else results[results["successRate"] >= threshold]
        wins = int((sub["result"] == 1).sum())
        losses = int((sub["result"] == -1).sum())
        decided = wins + losses
        row[f"{name}_record"] = f"{wins}-{losses}"
        row[f"{name}_rate"] = (wins / decided) if decided else float("nan")
        row[f"{name}_n"] = decided
    return row


def format_table(rows: list[dict]) -> str:
    header = (f"{'run':<28}" + "".join(f"{n:>22}" for n, _ in TIERS))
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = ""
        for name, _ in TIERS:
            rate = row[f"{name}_rate"]
            pct = "  n/a" if pd.isna(rate) else f"{rate * 100:5.1f}%"
            flag = "" if pd.isna(rate) else ("+" if rate >= BREAK_EVEN else " ")
            cells += f"{row[f'{name}_record'] + ' ' + pct + flag:>22}"
        lines.append(f"{row['run']:<28}{cells}")
    lines.append("")
    lines.append(f"'+' marks a tier at or above the {BREAK_EVEN * 100:.2f}% break-even at -110 odds.")
    return "\n".join(lines)


def load_backtest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[[c for c in df.columns if not c.startswith("Unnamed")]]


def compare_backtests(paths: dict[str, Path]) -> tuple[list[dict], str]:
    rows = [tier_row(load_backtest(p), label) for label, p in paths.items() if Path(p).exists()]
    return rows, format_table(rows)
