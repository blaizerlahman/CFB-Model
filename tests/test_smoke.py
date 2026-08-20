"""Phase 0 smoke tests: package imports and CLI wiring."""

from cfb_model import constants
from cfb_model.cli import build_parser
from cfb_model.config import get_settings


def test_constants_shape():
    assert len(constants.BASE_STATS) == 25
    assert constants.WINDOWS == (20, 8)
    assert constants.PROVIDER_ORDER[0] == "ESPN Bet"
    assert len(constants.NAME_MAP) == 6


def test_cli_parses_all_commands():
    parser = build_parser()
    for argv in (
        ["predict"],
        ["update", "--year", "2025", "--week", "3"],
        ["analyze", "--year", "2024"],
        ["setup-season", "--year", "2026"],
        ["backtest", "--year", "2024"],
        ["migrate", "--validate-only"],
        ["matchup", "Alabama", "Georgia", "--spread", "-3.5"],
    ):
        args = parser.parse_args(argv)
        assert args.command == argv[0]


def test_settings_paths_are_project_relative():
    s = get_settings()
    assert s.db_path.parent.name == "data"
    assert s.project_root.name == "CFB Model"
