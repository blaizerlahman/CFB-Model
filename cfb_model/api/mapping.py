"""CFBD API JSON -> exporter-convention DataFrames.

The whole pipeline (stored frames, pickled models) speaks the old exporter
CSV dialect: PascalCase columns, compound stats encoded as decimals, one row
per game/team. This module is the single translation point; every field map
is table-driven so API drift is a one-line fix.

Verified live against api.collegefootballdata.com on 2026-08-20:
  /games/teams -> [{id, teams: [{team, teamId, conference, homeAway, points,
                    stats: [{category, stat}]}]}]
  /lines       -> [{id, season, week, homeTeam, homeScore, awayTeam, awayScore,
                    lines: [{provider, spread, formattedSpread, spreadOpen,
                             overUnder, overUnderOpen, homeMoneyline,
                             awayMoneyline}]}]
  /ratings/sp  -> [{year, team, rating, ranking, ...}]  (season granularity only)
  /talent      -> [{year, team, talent}]
  /calendar    -> [{season, week, seasonType, startDate, endDate, ...}]
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cfb_model.constants import NAME_MAP
from cfb_model.features.rolling import encode_stat

# Long-format identity columns produced for each (game, team) row before pivot.
GAME_TEAM_ID_COLUMNS = ("Game Id", "School", "Conference", "HomeAway", "Points", "Week", "Year")

# API line field -> legacy betting column (order = legacy column order).
LINE_FIELD_MAP = {
    "id": "Id",
    "homeTeam": "HomeTeam",
    "homeScore": "HomeScore",
    "awayTeam": "AwayTeam",
    "awayScore": "AwayScore",
    "provider": "LineProvider",       # nested per-line field
    "overUnder": "OverUnder",         # nested
    "spread": "Spread",               # nested
    "formattedSpread": "FormattedSpread",  # nested
    "spreadOpen": "OpeningSpread",    # nested (renamed by API)
    "overUnderOpen": "OpeningOverUnder",   # nested (renamed by API)
    "homeMoneyline": "HomeMoneyline",  # nested
    "awayMoneyline": "AwayMoneyline",  # nested
}
NESTED_LINE_FIELDS = (
    "provider", "overUnder", "spread", "formattedSpread",
    "spreadOpen", "overUnderOpen", "homeMoneyline", "awayMoneyline",
)
BETTING_COLUMNS = tuple(LINE_FIELD_MAP.values())


def normalize_school(name: str) -> str:
    """CFBD school name -> canonical model key."""
    return NAME_MAP.get(name, name)


def team_stats_to_wide(records: list[dict], year: int, week: int) -> pd.DataFrame:
    """/games/teams JSON -> wide frame (one row per game/team), exporter dialect.

    Stat values run through encode_stat ('12-95' -> 12.95). Week/Year come from
    the request (the response carries neither). School names are normalized.
    Missing stat categories become NaN columns so every row shares one schema.
    """
    rows: list[dict] = []
    for game in records:
        for entry in game.get("teams", []):
            row = {
                "Game Id": game["id"],
                "School": normalize_school(entry["team"]),
                "Conference": entry.get("conference"),
                "HomeAway": entry.get("homeAway"),
                "Points": pd.to_numeric(entry.get("points"), errors="coerce"),
                "Week": week,
                "Year": year,
            }
            for stat in entry.get("stats", []):
                row[stat["category"]] = encode_stat(stat["stat"])
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(GAME_TEAM_ID_COLUMNS))

    df = pd.DataFrame(rows)
    # Cross-classification games appear in BOTH the fbs and fcs pulls; keep
    # one row per (game, team) or downstream self-merges explode into
    # duplicate rows that corrupt rolling windows.
    df = df.drop_duplicates(subset=["Game Id", "School"], keep="first")
    # Stable column order: identity columns first, stat categories sorted.
    stat_cols = sorted(c for c in df.columns if c not in GAME_TEAM_ID_COLUMNS)
    return df[list(GAME_TEAM_ID_COLUMNS) + stat_cols].reset_index(drop=True)


def team_stats_to_wide_multi(records: list[dict], year: int,
                             week_by_game: dict) -> pd.DataFrame:
    """Like team_stats_to_wide for a whole-season /games/teams response:
    weeks come from a {game id: week} map (built from /games)."""
    frames = []
    by_week: dict[int, list[dict]] = {}
    for game in records:
        week = week_by_game.get(game["id"])
        if week is None:
            continue
        by_week.setdefault(week, []).append(game)
    for week, games in sorted(by_week.items()):
        frames.append(team_stats_to_wide(games, year=year, week=week))
    if not frames:
        return pd.DataFrame(columns=list(GAME_TEAM_ID_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def lines_to_frame(records: list[dict]) -> pd.DataFrame:
    """/lines JSON -> one row per (game, provider) in the legacy column dialect.

    Team names are left as raw CFBD names (legacy behavior: the notebooks
    never normalized HomeTeam/AwayTeam in betting data). Games without lines
    are dropped, as the exporter did.
    """
    rows: list[dict] = []
    for game in records:
        for line in game.get("lines", []):
            row = {}
            for api_field, column in LINE_FIELD_MAP.items():
                source = line if api_field in NESTED_LINE_FIELDS else game
                row[column] = source.get(api_field)
            rows.append(row)

    df = pd.DataFrame(rows, columns=list(BETTING_COLUMNS))
    for col in ("OverUnder", "Spread", "OpeningSpread", "OpeningOverUnder",
                "HomeMoneyline", "AwayMoneyline", "HomeScore", "AwayScore"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sp_to_frame(records: list[dict]) -> pd.DataFrame:
    """/ratings/sp JSON -> Year/Team/Rating (legacy SP file dialect).

    'nationalAverages' pseudo-rows (if present) are excluded.
    """
    rows = [
        {"Year": r.get("year"), "Team": r.get("team"), "Rating": r.get("rating")}
        for r in records
        if r.get("team") and r.get("team") != "nationalAverages"
    ]
    return pd.DataFrame(rows, columns=["Year", "Team", "Rating"])


def talent_to_frame(records: list[dict]) -> pd.DataFrame:
    """/talent JSON -> Year/School/Talent (legacy talent file dialect).

    The API renamed 'school' to 'team'; the legacy files use School.
    """
    rows = [
        {"Year": r.get("year"), "School": r.get("team"), "Talent": pd.to_numeric(r.get("talent"), errors="coerce")}
        for r in records
    ]
    return pd.DataFrame(rows, columns=["Year", "School", "Talent"])


def current_week(calendar_records: list[dict], now: datetime | None = None) -> tuple[int, int] | None:
    """(season, week) for the calendar entry containing `now`, else the next
    upcoming entry, else None (off-season past the last entry)."""
    now = now or datetime.now(timezone.utc)
    upcoming: tuple[int, int] | None = None
    upcoming_start: datetime | None = None
    for entry in calendar_records:
        if entry.get("seasonType") not in (None, "regular", "postseason"):
            continue
        start = datetime.fromisoformat(entry["startDate"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(entry["endDate"].replace("Z", "+00:00"))
        if start <= now <= end:
            return entry["season"], entry["week"]
        if start > now and (upcoming_start is None or start < upcoming_start):
            upcoming_start = start
            upcoming = (entry["season"], entry["week"])
    return upcoming
