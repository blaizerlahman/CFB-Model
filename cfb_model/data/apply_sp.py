"""Rewrite a season's SP+ features from its weekly snapshots.

Importing weekly ratings only fills `sp_ratings`. The model reads SP and
SP_opp off each game row in `team_games`, and for seasons before 2024 those
carry the season-FINAL rating — a number that did not exist when the game was
played. This walks a season's rows and replaces them with the snapshot for
that row's own week, which is what a bettor had at kickoff.

Only rows whose week has a snapshot are touched; the rest are reported and
left alone. Nothing here reads or writes the legacy CSVs, and re-running
`migrate` restores the original values.
"""

from __future__ import annotations

import pandas as pd

from cfb_model.data.store import Store


def apply_weekly_sp(store: Store, season: int, log=print) -> dict:
    weeks = store.sp_weeks(season)
    if not weeks:
        raise ValueError(f"No weekly SP+ snapshots stored for {season}.")

    snapshots = {w: store.load_sp(season, w).set_index("Team")["Rating"].to_dict()
                 for w in weeks}
    log(f"{season}: snapshots for weeks {weeks}")

    updated = skipped_no_week = skipped_no_team = 0
    changed_teams: set[str] = set()

    for classification in ("fbs", "fcs"):
        for team in store.team_keys(classification):
            frame = store.load_team_frame(team)
            if frame is None or frame.empty:
                continue
            rows = frame[frame["Year"] == season]
            if rows.empty:
                continue

            for idx, row in rows.iterrows():
                week = row.get("Week")
                if pd.isna(week):
                    continue
                table = snapshots.get(int(week))
                if table is None:
                    skipped_no_week += 1
                    continue
                own = table.get(team)
                opp = table.get(row.get("School_opp"))
                if own is None and opp is None:
                    skipped_no_team += 1
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
                changed_teams.add(team)

    log(f"  rows updated: {updated} across {len(changed_teams)} teams")
    if skipped_no_week:
        log(f"  rows left alone (no snapshot for their week): {skipped_no_week}")
    if skipped_no_team:
        log(f"  rows left alone (team absent from the snapshot): {skipped_no_team}")
    return {"season": season, "updated": updated, "teams": len(changed_teams),
            "skipped_no_week": skipped_no_week, "skipped_no_team": skipped_no_team}


def sp_variety(store: Store, season: int) -> pd.DataFrame:
    """Distinct SP values per team for a season — 1 means season-final is still
    baked in, more means the weekly values landed."""
    return pd.read_sql_query(
        'SELECT team_key, COUNT(DISTINCT "SP") AS distinct_sp, COUNT(*) AS rows '
        'FROM team_games WHERE "Year" = ? AND "SP" IS NOT NULL GROUP BY team_key',
        store.conn, params=(season,))
