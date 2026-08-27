"""SQLite store — the consolidated replacement for the per-team weekly CSVs.

Design (plan §4): the wide per-team frame is materialized verbatim in
`team_games` (quoted identifiers, e.g. "Game Id", "rolling_sum_Points20") so
frames round-trip exactly. Historical rows are migrated as-is and never
recomputed. `row_order` preserves original row positions; every frame load
is ORDER BY row_order, which guarantees identical rolling windows.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1"

BOOKKEEPING_COLUMNS = ("team_key", "classification", "row_order")

# Data columns that hold text; everything else in team_games is numeric (REAL).
TEXT_COLUMNS = {
    "School", "Conference", "HomeAway",
    "School_opp", "Conference_opp", "HomeAway_opp",
    "HomeTeam", "AwayTeam", "LineProvider", "FormattedSpread",
}
# Win/Win_opp are booleans in the CSVs; stored as INTEGER 0/1.
BOOL_COLUMNS = {"Win", "Win_opp"}

FIXED_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS betting_lines (
  game_id REAL NOT NULL, provider TEXT NOT NULL,
  season INTEGER, week INTEGER, season_type TEXT,
  home_team TEXT, away_team TEXT, home_score REAL, away_score REAL,
  spread REAL, formatted_spread TEXT, over_under REAL,
  opening_spread REAL, opening_over_under REAL,
  home_moneyline REAL, away_moneyline REAL,
  fetched_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (game_id, provider)
);

CREATE TABLE IF NOT EXISTS sp_ratings (
  season INTEGER NOT NULL, week INTEGER NOT NULL, team TEXT NOT NULL,
  rating REAL,
  PRIMARY KEY (season, week, team)
);

CREATE TABLE IF NOT EXISTS talent (
  season INTEGER NOT NULL, school TEXT NOT NULL, talent REAL,
  PRIMARY KEY (season, school)
);

-- Named bin sets. The original table above is left untouched so the shipped
-- legacy calibration stays available for comparison.
CREATE TABLE IF NOT EXISTS bin_sets (
  name TEXT NOT NULL,
  lower_bin REAL NOT NULL, upper_bin REAL NOT NULL,
  success_rate REAL, game_count REAL,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (name, lower_bin, upper_bin)
);

CREATE TABLE IF NOT EXISTS bins (
  lower_bin REAL NOT NULL, upper_bin REAL NOT NULL,
  success_rate REAL, game_count REAL,
  PRIMARY KEY (lower_bin, upper_bin)
);

CREATE TABLE IF NOT EXISTS predictions (
  season INTEGER NOT NULL, week INTEGER NOT NULL, day TEXT NOT NULL,
  game_id REAL NOT NULL, team TEXT NOT NULL, opp_team TEXT,
  pred REAL, spread REAL, spread_diff REAL, cover INTEGER, success_rate REAL,
  pick TEXT, pick_spread REAL,
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (season, week, day, game_id, team)
);

CREATE TABLE IF NOT EXISTS results (
  season INTEGER NOT NULL, week INTEGER NOT NULL, game_id REAL NOT NULL,
  team TEXT, opp_team TEXT, day TEXT,
  pred REAL, spread REAL, spread_diff REAL, cover INTEGER,
  score_diff REAL, result REAL, success_rate REAL,
  pick TEXT, pick_spread REAL,
  PRIMARY KEY (season, week, game_id)
);

CREATE TABLE IF NOT EXISTS matchup_queries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  team1 TEXT, team2 TEXT, spread REAL, pred REAL, spread_diff REAL,
  success_rate REAL,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_registry (
  team_key TEXT NOT NULL, season INTEGER NOT NULL,
  path TEXT, best_alpha REAL, n_train_rows INTEGER,
  sklearn_version TEXT, trained_at TEXT,
  PRIMARY KEY (team_key, season)
);
"""


def _quote(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


class Store:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._ensure_pick_columns()

    def _ensure_pick_columns(self) -> None:
        """Add the game-level pick columns to a database that predates them.

        Cheap enough to run on connect, and it means an existing store — or a
        copy taken for an experiment — self-heals rather than failing on the
        first query that mentions them.
        """
        for table in ("predictions", "results"):
            present = {r[0] for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))}
            if not present:
                continue
            columns = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            with self.conn:
                if "pick" not in columns:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN pick TEXT")
                if "pick_spread" not in columns:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN pick_spread REAL")

    def close(self) -> None:
        self.conn.close()

    # ---- schema ----

    def init_schema(self, team_columns: list[str]) -> None:
        """Create all tables. `team_columns` is the verbatim wide-frame column
        list (union across sources, Unnamed columns already dropped)."""
        with self.conn:
            self.conn.executescript(FIXED_TABLES_SQL)
            col_defs = ['team_key TEXT NOT NULL', "classification TEXT NOT NULL"]
            for col in team_columns:
                if col in TEXT_COLUMNS:
                    sql_type = "TEXT"
                elif col in BOOL_COLUMNS:
                    sql_type = "INTEGER"
                else:
                    sql_type = "REAL"
                col_defs.append(f"{_quote(col)} {sql_type}")
            col_defs.append("row_order INTEGER NOT NULL")
            self.conn.execute(
                f"CREATE TABLE IF NOT EXISTS team_games ({', '.join(col_defs)}, "
                f'PRIMARY KEY (team_key, "Game Id"))'
            )
            self.conn.execute(
                'CREATE INDEX IF NOT EXISTS ix_tg_team_year_week '
                'ON team_games (team_key, "Year", "Week")'
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def team_columns(self) -> list[str]:
        cur = self.conn.execute("PRAGMA table_info(team_games)")
        cols = [row[1] for row in cur.fetchall()]
        return [c for c in cols if c not in BOOKKEEPING_COLUMNS]

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
            )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    # ---- team frames ----

    def replace_team_frame(self, team_key: str, classification: str, df: pd.DataFrame) -> None:
        """Insert a full team frame, assigning row_order 0..n-1. Existing rows
        for (team_key, Game Id) are replaced (idempotent re-migration)."""
        self._insert_rows(team_key, classification, df, start_order=0)

    def append_team_rows(self, team_key: str, classification: str, df: pd.DataFrame) -> None:
        """Append new game rows after the current last row_order."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(row_order), -1) FROM team_games WHERE team_key = ?",
            (team_key,),
        ).fetchone()
        self._insert_rows(team_key, classification, df, start_order=row[0] + 1)

    def _insert_rows(self, team_key: str, classification: str, df: pd.DataFrame, start_order: int) -> None:
        table_cols = self.team_columns()
        insert_cols = ["team_key", "classification"] + table_cols + ["row_order"]
        placeholders = ", ".join(["?"] * len(insert_cols))
        quoted = ", ".join(_quote(c) for c in insert_cols)
        sql = f"INSERT OR REPLACE INTO team_games ({quoted}) VALUES ({placeholders})"

        records = []
        for offset, (_, row) in enumerate(df.iterrows()):
            values: list = [team_key, classification]
            for col in table_cols:
                v = row[col] if col in row.index else None
                if pd.isna(v):
                    v = None
                elif col in BOOL_COLUMNS:
                    v = int(bool(v))
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, (np.bool_, bool)):
                    v = int(v)
                values.append(v)
            values.append(start_order + offset)
            records.append(values)

        with self.conn:
            self.conn.executemany(sql, records)

    def load_team_frame(self, team_key: str) -> pd.DataFrame | None:
        df = pd.read_sql_query(
            "SELECT * FROM team_games WHERE team_key = ? ORDER BY row_order",
            self.conn,
            params=(team_key,),
        )
        if df.empty:
            return None
        df = df.drop(columns=list(BOOKKEEPING_COLUMNS))
        return self._restore_dtypes(df)

    def load_all_team_frames(self, classification: str) -> dict[str, pd.DataFrame]:
        keys = [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT team_key FROM team_games WHERE classification = ? ORDER BY team_key",
                (classification,),
            )
        ]
        return {k: self.load_team_frame(k) for k in keys}

    def team_keys(self, classification: str | None = None) -> list[str]:
        if classification:
            cur = self.conn.execute(
                "SELECT DISTINCT team_key FROM team_games WHERE classification = ? ORDER BY team_key",
                (classification,),
            )
        else:
            cur = self.conn.execute("SELECT DISTINCT team_key FROM team_games ORDER BY team_key")
        return [r[0] for r in cur.fetchall()]

    @staticmethod
    def _restore_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col in TEXT_COLUMNS:
                # None -> NaN so text columns match read_csv object columns.
                df[col] = df[col].where(df[col].notna(), np.nan)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ---- reference tables ----

    def upsert_bins(self, bins: pd.DataFrame) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO bins (lower_bin, upper_bin, success_rate, game_count) VALUES (?,?,?,?)",
                bins[["lowerBin", "upperBin", "successRate", "gameCount"]].itertuples(index=False, name=None),
            )

    def load_bins(self) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT lower_bin AS lowerBin, upper_bin AS upperBin, "
            "success_rate AS successRate, game_count AS gameCount "
            "FROM bins ORDER BY lower_bin",
            self.conn,
        )
        return df

    def upsert_bin_set(self, name: str, bins: pd.DataFrame) -> None:
        """Store a named calibration. Replaces only that name."""
        rows = [
            (name, r.lowerBin, r.upperBin,
             None if pd.isna(r.successRate) else float(r.successRate),
             float(r.gameCount))
            for r in bins.itertuples(index=False)
        ]
        with self.conn:
            self.conn.execute("DELETE FROM bin_sets WHERE name = ?", (name,))
            self.conn.executemany(
                "INSERT INTO bin_sets (name, lower_bin, upper_bin, success_rate, game_count)"
                " VALUES (?,?,?,?,?)", rows)

    def load_bin_set(self, name: str) -> pd.DataFrame:
        """Load a named calibration; 'legacy' falls back to the original table."""
        if name == "legacy":
            return self.load_bins()
        return pd.read_sql_query(
            "SELECT lower_bin AS lowerBin, upper_bin AS upperBin,"
            " success_rate AS successRate, game_count AS gameCount"
            " FROM bin_sets WHERE name = ? ORDER BY lower_bin",
            self.conn, params=(name,))

    def bin_set_names(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT name FROM bin_sets ORDER BY name")]

    def upsert_talent(self, season: int, talent: pd.DataFrame) -> None:
        rows = [(season, r.School, float(r.Talent)) for r in talent.itertuples(index=False)]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO talent (season, school, talent) VALUES (?,?,?)", rows
            )

    def load_talent(self, season: int) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT school AS School, talent AS Talent FROM talent WHERE season = ?",
            self.conn,
            params=(season,),
        )

    def upsert_sp(self, season: int, week: int, sp: pd.DataFrame,
                  overwrite: bool = True) -> bool:
        """Store a weekly SP+ snapshot. Returns True if anything was written.

        `overwrite=False` makes the first snapshot of a week final. In-season
        that is the safe default: SP+ publishes once a week, so the first
        capture holds the ratings that were public before the week's games,
        while a later re-run (say Sunday, once results are in) could otherwise
        replace them with ratings that already know those results.
        """
        if not overwrite and self.load_sp(season, week).shape[0] > 0:
            return False
        rows = [
            (season, week, r.Team, None if pd.isna(r.Rating) else float(r.Rating))
            for r in sp.itertuples(index=False)
        ]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO sp_ratings (season, week, team, rating) VALUES (?,?,?,?)", rows
            )
        return True

    def load_sp(self, season: int, week: int) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT team AS Team, rating AS Rating FROM sp_ratings WHERE season = ? AND week = ?",
            self.conn,
            params=(season, week),
        )

    def sp_weeks(self, season: int) -> list[int]:
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT week FROM sp_ratings WHERE season = ? ORDER BY week", (season,)
            )
        ]

    # ---- betting lines ----

    def upsert_lines(self, season: int, week: int, lines: pd.DataFrame,
                     season_type: str = "regular") -> None:
        rows = []
        for r in lines.itertuples(index=False):
            rows.append((
                float(r.Id), r.LineProvider, season, week, season_type,
                r.HomeTeam, r.AwayTeam,
                None if pd.isna(r.HomeScore) else float(r.HomeScore),
                None if pd.isna(r.AwayScore) else float(r.AwayScore),
                None if pd.isna(r.Spread) else float(r.Spread),
                None if pd.isna(r.FormattedSpread) else str(r.FormattedSpread),
                None if pd.isna(r.OverUnder) else float(r.OverUnder),
                None if pd.isna(r.OpeningSpread) else float(r.OpeningSpread),
                None if pd.isna(r.OpeningOverUnder) else float(r.OpeningOverUnder),
                None if pd.isna(r.HomeMoneyline) else float(r.HomeMoneyline),
                None if pd.isna(r.AwayMoneyline) else float(r.AwayMoneyline),
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO betting_lines (game_id, provider, season, week, season_type,"
                " home_team, away_team, home_score, away_score, spread, formatted_spread, over_under,"
                " opening_spread, opening_over_under, home_moneyline, away_moneyline)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_lines(self, season: int, week: int) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT game_id AS Id, home_team AS HomeTeam, home_score AS HomeScore,"
            " away_team AS AwayTeam, away_score AS AwayScore, provider AS LineProvider,"
            " over_under AS OverUnder, spread AS Spread, formatted_spread AS FormattedSpread,"
            " opening_spread AS OpeningSpread, opening_over_under AS OpeningOverUnder,"
            " home_moneyline AS HomeMoneyline, away_moneyline AS AwayMoneyline"
            " FROM betting_lines WHERE season = ? AND week = ?",
            self.conn,
            params=(season, week),
        )

    # ---- predictions / results ----

    def upsert_predictions(self, season: int, week: int, day: str, preds: pd.DataFrame) -> None:
        rows = []
        for r in preds.itertuples(index=False):
            rows.append((
                season, week, day, float(r.gameID), r.team, r.oppTeam,
                None if pd.isna(r.pred) else float(r.pred),
                None if pd.isna(r.spread) else float(r.spread),
                None if pd.isna(r.spreadDiff) else float(r.spreadDiff),
                None if pd.isna(r.cover) else int(r.cover),
                None if not hasattr(r, "successRate") or pd.isna(r.successRate) else float(r.successRate),
                getattr(r, "pick", None) if not pd.isna(getattr(r, "pick", None)) else None,
                None if pd.isna(getattr(r, "pickSpread", None)) else float(getattr(r, "pickSpread")),
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO predictions (season, week, day, game_id, team, opp_team,"
                " pred, spread, spread_diff, cover, success_rate, pick, pick_spread)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_predictions(self, season: int, week: int) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT pred, spread, spread_diff AS spreadDiff, cover, game_id AS gameID,"
            " team, opp_team AS oppTeam, day, success_rate AS successRate,"
            " pick, pick_spread AS pickSpread"
            " FROM predictions WHERE season = ? AND week = ?",
            self.conn,
            params=(season, week),
        )

    def prediction_weeks(self, season: int) -> list[int]:
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT DISTINCT week FROM predictions WHERE season = ? ORDER BY week", (season,)
            )
        ]

    def upsert_results(self, season: int, week: int, results: pd.DataFrame) -> None:
        rows = []
        for r in results.itertuples(index=False):
            rows.append((
                season, week, float(r.gameID), r.team, r.oppTeam, r.day,
                None if pd.isna(r.pred) else float(r.pred),
                None if pd.isna(r.spread) else float(r.spread),
                None if pd.isna(r.spreadDiff) else float(r.spreadDiff),
                None if pd.isna(r.cover) else int(r.cover),
                None if pd.isna(r.scoreDiff) else float(r.scoreDiff),
                None if pd.isna(r.result) else float(r.result),
                None if pd.isna(r.successRate) else float(r.successRate),
                getattr(r, "pick", None) if not pd.isna(getattr(r, "pick", None)) else None,
                None if pd.isna(getattr(r, "pickSpread", None)) else float(getattr(r, "pickSpread")),
            ))
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO results (season, week, game_id, team, opp_team, day,"
                " pred, spread, spread_diff, cover, score_diff, result, success_rate,"
                " pick, pick_spread)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_results(self, season: int, week: int | None = None) -> pd.DataFrame:
        sql = (
            "SELECT pred, spread, spread_diff AS spreadDiff, cover, game_id AS gameID,"
            " team, opp_team AS oppTeam, day, score_diff AS scoreDiff, result,"
            " success_rate AS successRate, pick, pick_spread AS pickSpread, week"
            " FROM results WHERE season = ?"
        )
        params: tuple = (season,)
        if week is not None:
            sql += " AND week = ?"
            params = (season, week)
        return pd.read_sql_query(sql + " ORDER BY week, game_id", self.conn, params=params)
