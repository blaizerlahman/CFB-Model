"""Experiment: use the PRIOR season's final SP+ as a preseason rating.

Seasons before 2023 have no weekly SP+ and currently carry their own
season-FINAL rating on every game row — a number that did not exist when the
game was played. Dropping the feature outright made things worse (2024 fell
from 49.8% to 47.0%), so the question is whether a leak-free stand-in helps:
where a team FINISHED the previous season is knowable before kickoff and is a
reasonable read on how strong it enters the new one.

Seasons that already have genuine weekly ratings (2023-2025) are left alone,
as are individual rows that have a real weekly snapshot.

Runs entirely against a COPY of the database. The live store, the models, and
every stored calibration are untouched.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from cfb_model.analysis.backtest import run_backtest  # noqa: E402
from cfb_model.analysis.compare import compare_backtests, tier_row  # noqa: E402
from cfb_model.api.client import CfbdClient  # noqa: E402
from cfb_model.api.mapping import sp_to_frame  # noqa: E402
from cfb_model.config import get_settings  # noqa: E402
from cfb_model.data.store import Store  # noqa: E402

WEEKLY_SEASONS = {2023, 2024, 2025}   # already contemporaneous; do not touch


def season_finals(client: CfbdClient, years: list[int], log=print) -> dict[int, dict[str, float]]:
    finals: dict[int, dict[str, float]] = {}
    for year in years:
        frame = sp_to_frame(client.sp_ratings(year))
        if frame.empty:
            log(f"  {year}: no SP+ published")
            continue
        finals[year] = dict(zip(frame["Team"], frame["Rating"]))
        log(f"  {year} final SP+: {len(finals[year])} teams")
    return finals


def apply_lagged(store: Store, finals: dict[int, dict[str, float]], log=print) -> dict:
    seasons = sorted(
        int(y) for (y,) in store.conn.execute(
            'SELECT DISTINCT "Year" FROM team_games WHERE "Year" IS NOT NULL'))
    updated = missing_prior = kept_weekly = no_rating = 0

    for season in seasons:
        if season in WEEKLY_SEASONS:
            continue
        prior = finals.get(season - 1)
        if not prior:
            missing_prior += 1
            log(f"  {season}: no {season - 1} finals available, left as-is")
            continue
        weekly_weeks = {w for w in store.sp_weeks(season) if w > 0}

        for classification in ("fbs", "fcs"):
            for team in store.team_keys(classification):
                frame = store.load_team_frame(team)
                if frame is None:
                    continue
                rows = frame[frame["Year"] == season]
                for _, row in rows.iterrows():
                    week = row.get("Week")
                    if pd.notna(week) and int(week) in weekly_weeks:
                        kept_weekly += 1          # a genuine weekly value already
                        continue
                    own = prior.get(team)
                    opp = prior.get(row.get("School_opp"))
                    if own is None and opp is None:
                        no_rating += 1
                        continue
                    sets, params = [], []
                    if own is not None:
                        sets.append('"SP" = ?'); params.append(float(own))
                    if opp is not None:
                        sets.append('"SP_opp" = ?'); params.append(float(opp))
                    params += [team, float(row["Game Id"])]
                    with store.conn:
                        store.conn.execute(
                            f'UPDATE team_games SET {", ".join(sets)} '
                            'WHERE team_key = ? AND "Game Id" = ?', params)
                    updated += 1
        log(f"  {season}: rewritten from {season - 1} finals")
    return {"updated": updated, "kept_weekly": kept_weekly,
            "no_rating": no_rating, "seasons_without_prior": missing_prior}


def main() -> int:
    settings = get_settings()
    work_dir = settings.output_root / "experiments"
    work_dir.mkdir(parents=True, exist_ok=True)
    copy_path = work_dir / "lagged_sp.db"

    print(f"Copying the database to {copy_path} (the live one is not touched)")
    shutil.copy2(settings.db_path, copy_path)
    for suffix in ("-wal", "-shm"):
        side = Path(str(settings.db_path) + suffix)
        if side.exists():
            shutil.copy2(side, str(copy_path) + suffix)

    store = Store(copy_path)
    client = CfbdClient(settings)

    seasons = sorted(
        int(y) for (y,) in store.conn.execute(
            'SELECT DISTINCT "Year" FROM team_games WHERE "Year" IS NOT NULL'))
    need = sorted({y - 1 for y in seasons if y not in WEEKLY_SEASONS})
    print(f"\nFetching final SP+ for {need[0]}-{need[-1]} ({len(need)} calls)")
    finals = season_finals(client, need)

    print("\nRewriting pre-2023 seasons to the prior year's finals")
    stats = apply_lagged(store, finals)
    print(f"  rows rewritten: {stats['updated']}")
    print(f"  rows left as genuine weekly: {stats['kept_weekly']}")
    print(f"  rows with no prior rating for either team: {stats['no_rating']}")
    print(f"API calls: {client.calls_made}")

    rows = []
    for year in (2024, 2025):
        print(f"\n=== backtesting {year} on the experimental copy ===")
        results = run_backtest(store, year, random_state=50, settings=settings,
                               label=f"laggedsp_{year}")
        if not results.empty:
            rows.append(tier_row(results, f"{year} lagged prior-year SP+"))

    baseline = {
        "2024 current (season-final SP+)":
            settings.output_root / "backtests" / "2024" / "backtest_2024_results.csv",
        "2025 current (season-final SP+)":
            settings.output_root / "backtests" / "2025" / "backtest_2025_results.csv",
    }
    base_rows, _ = compare_backtests(baseline)
    from cfb_model.analysis.compare import format_table
    print("\n\n=== TIER COMPARISON ===\n")
    print(format_table(base_rows + rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
