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
    for w in range(min(latest + 1, week), week + 1):
        summary = ingest_week(store, client, year, w, settings)
        print(f"{year} week {w}: updated {summary['teams_updated']} teams")
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
    chosen = backfill_sp(store, args.year, client.calendar(args.year), settings)
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
    p.set_defaults(handler=_not_implemented("Phase 6"))

    p = sub.add_parser("setup-season", help="backfill missing data via API and retrain team models")
    p.add_argument("--year", type=int, required=True)
    p.set_defaults(handler=_not_implemented("Phase 7"))

    p = sub.add_parser("backtest", help="replay a past season with as-of models")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--seed", type=int, default=50)
    p.set_defaults(handler=_not_implemented("Phase 7.5"))

    p = sub.add_parser("backfill-sp", help="fill weekly SP+ snapshots for a past season from Wayback captures")
    p.add_argument("--year", type=int, required=True)
    p.set_defaults(handler=_cmd_backfill_sp)

    p = sub.add_parser("migrate", help="one-time legacy CSV -> SQLite migration")
    p.add_argument("--validate-only", action="store_true")
    p.set_defaults(handler=_cmd_migrate)

    p = sub.add_parser("matchup", help="on-demand prediction for two teams")
    p.add_argument("team1")
    p.add_argument("team2")
    p.add_argument("--spread", type=float, default=None)
    p.set_defaults(handler=_not_implemented("Phase 8"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        if args.analyze:
            args.command = "analyze"
            args.year = None
            args.week = None
            args.handler = _not_implemented("Phase 6")
        else:
            parser.print_help()
            return 0

    return args.handler(args)
