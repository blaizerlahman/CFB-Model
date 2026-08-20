"""Full FBS-era history backfill for former skip-list teams (plan §11).

These teams have no usable frames (only stale 2-20-row FCS-era fragments):
their complete FBS history is fetched from the API, built with the standard
pipeline (same rolling/derived/week-0 semantics as every other frame), and
REPLACES the stale fragment. The hardcoded skip list then gives way to the
derived rule: an FBS team without a >=56-game frame gets no model, and games
against it are skipped.

Conventions preserved: 2020 skipped entirely (legacy COVID rule); historical
seasons use the year's final SP+ (exactly how p4 built every legacy training
frame); 2024/2025 rows use the stored weekly SP+ snapshots.
"""

from __future__ import annotations

import pandas as pd

from cfb_model.api import mapping
from cfb_model.api.client import CfbdClient
from cfb_model.data.store import Store
from cfb_model.features.assemble import build_team_frame, fix_week_zero

# CFBD school name -> first FBS season. 2025 newcomers included so their
# frames exist and grow; they stay below the model gate until ~2029.
FBS_SINCE = {
    "Coastal Carolina": 2017,
    "Liberty": 2018,
    "James Madison": 2022,
    "Jacksonville State": 2023,
    "Sam Houston": 2023,
    "Kennesaw State": 2024,
    "Delaware": 2025,
    "Missouri State": 2025,
}


def ensure_ratings_years(store: Store, client: CfbdClient, years: list[int], log=print) -> None:
    """Fetch talent and final-year SP+ (stored as week 0) for any season the
    store lacks."""
    for year in years:
        if store.load_talent(year).empty:
            talent = mapping.talent_to_frame(client.talent(year))
            if not talent.empty:
                store.upsert_talent(year, talent)
                log(f"  talent {year}: {len(talent)} teams")
        if not store.sp_weeks(year):
            sp = mapping.sp_to_frame(client.sp_ratings(year))
            if not sp.empty:
                store.upsert_sp(year, 0, sp)
                log(f"  SP+ {year} (final, week 0): {len(sp)} teams")


def _sp_for(store: Store, year: int, week: int) -> pd.DataFrame:
    weeks = [w for w in store.sp_weeks(year) if w <= week]
    if not weeks:
        weeks = store.sp_weeks(year)[:1]
    return store.load_sp(year, weeks[-1]) if weeks else pd.DataFrame(columns=["Team", "Rating"])


def backfill_team_history(store: Store, client: CfbdClient, team: str,
                          first_year: int, last_year: int, log=print) -> pd.DataFrame | None:
    """Build and store one team's complete FBS-era frame. Returns the frame."""
    wide_parts = []
    for year in range(first_year, last_year + 1):
        if year == 2020:  # legacy COVID skip
            continue
        games = client.games(year, team=team)
        week_by_game = {g["id"]: g.get("week") for g in games}
        records = client.team_game_stats(year, team=team)
        wide = mapping.team_stats_to_wide_multi(records, year, week_by_game)
        if not wide.empty:
            wide_parts.append(wide)
    if not wide_parts:
        log(f"  {team}: no games returned")
        return None
    wide = pd.concat(wide_parts, ignore_index=True)
    wide = wide.drop_duplicates(subset=["Game Id", "School"]).reset_index(drop=True)

    frame = build_team_frame(wide, team)
    frame = fix_week_zero(frame)
    frame = frame.reset_index(drop=True)

    # _opp rolling features from opponents' stored frames (matched by game).
    opp_cache: dict[str, pd.DataFrame | None] = {}
    rolling_cols = [c for c in frame.columns if c.startswith("rolling_sum") and not c.endswith("_opp")]
    for index, row in frame.iterrows():
        opp = row["School_opp"]
        if opp not in opp_cache:
            opp_cache[opp] = store.load_team_frame(opp)
        opp_df = opp_cache[opp]
        if opp_df is None:
            continue
        opp_row = opp_df[opp_df["Game Id"] == row["Game Id"]]
        if opp_row.empty:
            continue
        for col in rolling_cols:
            if col in opp_row.columns:
                frame.loc[index, col + "_opp"] = opp_row.iloc[0][col]

    # talent / SP (own and opponent), per year — p4 semantics.
    for col in ("talent", "talent_opp", "SP", "SP_opp"):
        if col not in frame.columns:
            frame[col] = pd.NA
    for year in sorted(frame["Year"].unique()):
        talent = store.load_talent(int(year)).set_index("School")["Talent"]
        year_mask = frame["Year"] == year
        own_talent = talent.get(team)
        if own_talent is not None:
            frame.loc[year_mask, "talent"] = own_talent
        for index in frame.index[year_mask]:
            week = int(frame.at[index, "Week"])
            sp = _sp_for(store, int(year), week).set_index("Team")["Rating"]
            opp = frame.at[index, "School_opp"]
            if sp.get(team) is not None:
                frame.at[index, "SP"] = sp.get(team)
            opp_talent = talent.get(opp)
            if opp_talent is not None:
                frame.at[index, "talent_opp"] = opp_talent
            if sp.get(opp) is not None:
                frame.at[index, "SP_opp"] = sp.get(opp)

    # Replace the stale fragment (derived DB rows only; legacy CSVs untouched).
    with store.conn:
        removed = store.conn.execute(
            "SELECT COUNT(*) FROM team_games WHERE team_key = ?", (team,)
        ).fetchone()[0]
        store.conn.execute("DELETE FROM team_games WHERE team_key = ?", (team,))
    store.replace_team_frame(team, "fbs", frame)
    log(f"  {team}: {len(frame)} games ({first_year}-{last_year}); replaced {removed} stale rows")
    return frame


def backfill_all_skip_teams(store: Store, client: CfbdClient, through_year: int, log=print) -> dict[str, int]:
    """Backfill every FBS_SINCE team through `through_year`. Returns
    {team: game count} for the §11 gate report."""
    years_needed = sorted({y for start in FBS_SINCE.values() for y in range(start, through_year + 1) if y != 2020})
    ensure_ratings_years(store, client, years_needed, log)

    counts: dict[str, int] = {}
    for team, since in FBS_SINCE.items():
        frame = backfill_team_history(store, client, team, since, through_year, log)
        counts[team] = 0 if frame is None else len(frame)
    return counts
