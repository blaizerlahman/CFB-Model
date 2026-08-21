"""Single CLI entry point: python -m cfb_model <command>.

All commands are non-interactive and cron-safe: no prompts, exit code 0 on
success (including "no slate today"), non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys


def _not_implemented(phase: str):
    def handler(args: argparse.Namespace) -> int:
        print(f"'{args.command}' is not implemented yet (lands in {phase}).", file=sys.stderr)
        return 2

    return handler


def _locked(fn):
    """Cron safety: overlapping runs no-op instead of colliding."""
    def wrapper(args: argparse.Namespace) -> int:
        import os
        import time

        from cfb_model.config import get_settings

        settings = get_settings()
        settings.ensure_dirs()
        lock = settings.logs_dir / ".cfb.lock"
        if lock.exists() and time.time() - lock.stat().st_mtime < 2 * 3600:
            print("Another cfb_model run appears to be active; exiting.", file=sys.stderr)
            return 0
        lock.write_text(str(os.getpid()))
        try:
            return fn(args)
        finally:
            lock.unlink(missing_ok=True)

    return wrapper


@_locked
def _cmd_predict(args: argparse.Namespace) -> int:
    from cfb_model.api.client import CfbdClient
    from cfb_model.config import get_settings
    from cfb_model.data.store import Store
    from cfb_model.pipeline import predict_run

    settings = get_settings()
    store = Store(settings.db_path)
    client = CfbdClient(settings)
    predict_run(store, client, args.year, args.week, args.day, settings)
    return 0


@_locked
def _cmd_update(args: argparse.Namespace) -> int:
    from cfb_model.api.client import CfbdClient
    from cfb_model.config import get_settings
    from cfb_model.data.store import Store
    from cfb_model.pipeline import ingest_week, latest_ingested_week

    settings = get_settings()
    store = Store(settings.db_path)
    client = CfbdClient(settings)

    year, week = args.year, args.week
    if year is None or week is None:
        from cfb_model.api.mapping import current_week

        detected = current_week(client.calendar(year or __import__("datetime").datetime.now().year))
        if detected is None:
            print("Off-season: nothing to ingest.")
            return 0
        year = year or detected[0]
        # Default: ingest the most recent COMPLETED week.
        week = week if week is not None else detected[1] - 1
    if week < 1:
        print("No completed week to ingest yet.")
        return 0
    latest = latest_ingested_week(store, year) or 0
    empty_streak = 0
    for w in range(min(latest + 1, week), week + 1):
        summary = ingest_week(store, client, year, w, settings)
        print(f"{year} week {w}: updated {summary['teams_updated']} teams")
        # Past the end of a season every further week costs 3 API calls and
        # returns nothing; give up after two in a row.
        empty_streak = empty_streak + 1 if summary["teams_updated"] == 0 else 0
        if empty_streak >= 2:
            print(f"No games for {year} weeks {w - 1}-{w}; stopping.")
            break
    print(f"API calls: {client.calls_made}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from cfb_model.analysis.evaluate import (
        big_spread_report,
        day_breakdown,
        grade_week,
        since_week_report,
        tier_breakdown,
    )
    from cfb_model.config import get_settings
    from cfb_model.data.store import Store
    from cfb_model.export import results_csv_path, write_results_csv
    from cfb_model.pipeline import load_gated_frames

    settings = get_settings()
    store = Store(settings.db_path)

    year = args.year
    if year is None:
        row = store.conn.execute("SELECT MAX(season) FROM predictions").fetchone()
        if not row or row[0] is None:
            print("No predictions stored yet.")
            return 1
        year = int(row[0])

    if args.week is not None:
        preds = store.load_predictions(year, args.week)
        if preds.empty:
            print(f"No stored predictions for {year} week {args.week}.")
            return 1
        fbs, fcs = load_gated_frames(store)
        results = grade_week(preds, store.load_bins(), fbs, set(fcs))
        if results.empty:
            print("No gradable games (are the week's results ingested?).")
            return 1
        print(f"=== {year} week {args.week} ===")
        print(tier_breakdown(results))
        print()
        print(day_breakdown(results))
        store.upsert_results(year, args.week, results)
        out = write_results_csv(results, results_csv_path(settings.output_root, year, args.week))
        print(f"\nSaved -> {out}")
        return 0

    results = store.load_results(year)
    if results.empty:
        print(f"No stored results for {year} — grade weeks first (analyze --week N).")
        return 1
    print(f"=== {year} season ({results['week'].nunique()} weeks, {len(results)} games) ===")
    print(tier_breakdown(results))
    print()
    print(day_breakdown(results))
    print()
    print(since_week_report(results))
    print()
    print(big_spread_report(results))
    return 0


@_locked
def _cmd_setup_season(args: argparse.Namespace) -> int:
    from cfb_model.api.client import CfbdClient
    from cfb_model.config import get_settings
    from cfb_model.data.store import Store
    from cfb_model.model.train import setup_season

    settings = get_settings()
    setup_season(Store(settings.db_path), CfbdClient(settings), args.year, settings)
    return 0


@_locked
def _cmd_backtest(args: argparse.Namespace) -> int:
    from cfb_model.analysis.backtest import run_backtest
    from cfb_model.config import get_settings
    from cfb_model.data.store import Store

    settings = get_settings()
    exclude = ("SP",) if getattr(args, "no_sp", False) else ()
    run_backtest(Store(settings.db_path), args.year, args.seed, settings,
                 exclude_features=exclude, label="nosp" if exclude else "")
    return 0


def _cmd_matchup(args: argparse.Namespace) -> int:
    import glob as _glob
    import re as _re

    from cfb_model.config import get_settings
    from cfb_model.data.store import Store
    from cfb_model.model.predict import predict_matchup
    from cfb_model.pipeline import load_models

    settings = get_settings()
    store = Store(settings.db_path)

    season = args.season
    if season is None:
        seasons = [
            int(m.group(1))
            for p in _glob.glob(str(settings.models_dir / "*_model_*.pkl"))
            if (m := _re.search(r"_model_(\d{4})\.pkl$", p))
        ]
        season = max(seasons) if seasons else 2024
    models = load_models(settings, season)
    if not models:
        print(f"No models for season {season} — run setup-season first.", file=sys.stderr)
        return 1

    result = predict_matchup(store, models, args.team1, args.team2, spread=args.spread)
    line = (f"{result['team']} vs {result['opponent']}: predicted score differential "
            f"{result['predicted_score_diff']:+g} (from {result['team']}'s perspective)")
    print(line)
    if result.get("spread") is not None:
        verb = {1: "COVER", -1: "NOT COVER", None: "TOSS UP"}.get(result["cover"], "TOSS UP")
        print(f"Against spread {result['spread']:+g}: {verb} "
              f"(spreadDiff {result['spread_diff']:+g})")
        if result.get("success_rate") is not None:
            print(f"Historical success rate: {result['success_rate'] * 100:.2f}% ({result['tier']})")
    return 0


@_locked
def _cmd_snapshot_sp(args: argparse.Namespace) -> int:
    from datetime import datetime

    from cfb_model.api.client import CfbdClient
    from cfb_model.api.mapping import current_week
    from cfb_model.config import get_settings
    from cfb_model.data.store import Store
    from cfb_model.pipeline import capture_sp_snapshot

    settings = get_settings()
    store = Store(settings.db_path)
    client = CfbdClient(settings)

    year, week = args.year, args.week
    if year is None or week is None:
        detected = current_week(client.calendar(year or datetime.now().year))
        if detected is None:
            print("Off-season: no week to snapshot.")
            return 0
        year = year or detected[0]
        week = week if week is not None else detected[1]

    capture_sp_snapshot(store, client, year, week)
    print(f"API calls: {client.calls_made}")
    return 0


def _cmd_backfill_sp(args: argparse.Namespace) -> int:
    from cfb_model.api.client import CfbdClient
    from cfb_model.config import get_settings
    from cfb_model.data.sp_backfill import backfill_sp
    from cfb_model.data.store import Store

    settings = get_settings()
    store = Store(settings.db_path)
    client = CfbdClient(settings)
    # Deadlines come from real kickoff times, not the calendar's week boundary.
    chosen = backfill_sp(store, args.year, client.games(args.year), settings)
    print(f"Backfilled SP+ for {args.year}: weeks {sorted(chosen)}")
    return 0


def _cmd_migrate(args: argparse.Namespace) -> int:
    from cfb_model.data.migrate import run_migration, validate_migration

    if not args.validate_only:
        run_migration()
    return 0 if validate_migration() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfb_model",
        description="CFB spread-prediction pipeline (CFBD API edition).",
    )
    # GOAL.md compatibility: `python -m cfb_model --analyze` == `analyze`.
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="alias for the 'analyze' command (season roll-up)",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("predict", help="daily one-liner: ingest if needed, predict upcoming week")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.add_argument("--day", type=str, default=None, help="day tag (default: today's weekday, e.g. tue)")
    p.set_defaults(handler=_cmd_predict)

    p = sub.add_parser("update", help="ingest a finished week's box scores and lines")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.set_defaults(handler=_cmd_update)

    p = sub.add_parser("analyze", help="success analysis: week grading or season roll-up")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.set_defaults(handler=_cmd_analyze)

    p = sub.add_parser("setup-season", help="backfill missing data via API and retrain team models")
    p.add_argument("--year", type=int, required=True)
    p.set_defaults(handler=_cmd_setup_season)

    p = sub.add_parser("backtest", help="replay a past season with as-of models")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--no-sp", action="store_true",
                   help="train and predict without SP+ features (historical SP+ rows carry "
                        "the season-final rating, which leaks end-of-season information)")
    p.set_defaults(handler=_cmd_backtest)

    p = sub.add_parser("snapshot-sp",
                       help="record this week's SP+ ratings (run weekly in-season; past weeks "
                            "cannot be recovered once a season ends)")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.set_defaults(handler=_cmd_snapshot_sp)

    p = sub.add_parser("backfill-sp", help="fill weekly SP+ snapshots for a past season from Wayback captures")
    p.add_argument("--year", type=int, required=True)
    p.set_defaults(handler=_cmd_backfill_sp)

    p = sub.add_parser("migrate", help="one-time legacy CSV -> SQLite migration")
    p.add_argument("--validate-only", action="store_true")
    p.set_defaults(handler=_cmd_migrate)

    p = sub.add_parser("matchup", help="on-demand prediction for two teams")
    p.add_argument("team1")
    p.add_argument("team2")
    p.add_argument("--spread", type=float, default=None,
                   help="line from TEAM1's perspective (negative = team1 favored)")
    p.add_argument("--season", type=int, default=None, help="model season (default: latest available)")
    p.set_defaults(handler=_cmd_matchup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        if args.analyze:
            args.command = "analyze"
            args.year = None
            args.week = None
            args.handler = _cmd_analyze
        else:
            parser.print_help()
            return 0

    return args.handler(args)
