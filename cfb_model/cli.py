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
    p.set_defaults(handler=_not_implemented("Phase 5"))

    p = sub.add_parser("update", help="ingest a finished week's box scores and lines")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    p.set_defaults(handler=_not_implemented("Phase 5"))

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

    p = sub.add_parser("migrate", help="one-time legacy CSV -> SQLite migration")
    p.add_argument("--validate-only", action="store_true")
    p.set_defaults(handler=_not_implemented("Phase 3"))

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
