"""Season replay backtest — the refactor's acceptance test (plan Phase 7.5).

For a past season Y: train as-of models (Year < Y, seeded for
reproducibility), then for each completed week rebuild the slate exactly as
the daily `predict` would have seen it (frames truncated to prior games,
that week's stored lines, that week's stored SP+ snapshot — for 2025 these
are the true weekly Wayback snapshots, never final-season values), predict,
grade, and report the standard tier breakdown.

Outputs go to output/backtests/{year}/ and are NEVER written to the
predictions/results tables (those hold real slate runs only).
"""

from __future__ import annotations

import pandas as pd

from cfb_model.analysis.evaluate import grade_week, tier_breakdown
from cfb_model.api.mapping import BETTING_COLUMNS
from cfb_model.config import Settings, get_settings
from cfb_model.constants import MIN_FBS_GAMES, MIN_FCS_GAMES, RESULTS_COLUMNS
from cfb_model.data.history import FBS_SINCE
from cfb_model.data.store import Store
from cfb_model.features.assemble import feature_columns
from cfb_model.model.predict import build_upcoming_frames, predict_week, select_preferred_lines
from cfb_model.model.train import train_all


def truncate_before(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    keep = (df["Year"] < year) | ((df["Year"] == year) & (df["Week"] < week))
    return df[keep].reset_index(drop=True)


def preferred_lines_for(store: Store, fbs_full: dict, year: int, week: int) -> pd.DataFrame:
    """Stored lines when available (2025+); otherwise reconstructed from the
    played rows' betting columns (2024 — same information the legacy weekly
    frames carried)."""
    lines = store.load_lines(year, week)
    if not lines.empty:
        return select_preferred_lines(lines)

    rows: dict[float, dict] = {}
    for team, df in fbs_full.items():
        played = df[(df["Year"] == year) & (df["Week"] == week)]
        for _, row in played.iterrows():
            game_id = row["Id"]
            if pd.isna(game_id) or game_id in rows:
                continue
            raw_spread = -row["Spread"] if row["HomeTeam"] == team else row["Spread"]
            rec = {c: row[c] if c in row.index else pd.NA for c in BETTING_COLUMNS}
            rec["Spread"] = raw_spread
            rows[game_id] = rec
    return pd.DataFrame(list(rows.values()), columns=list(BETTING_COLUMNS))


def run_backtest(store: Store, year: int, random_state: int = 50,
                 settings: Settings | None = None, log=print) -> pd.DataFrame:
    settings = settings or get_settings()
    out_dir = settings.output_root / "backtests" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Training as-of-{year} models (seed={random_state}) — this takes a while...")
    trained = train_all(store, year, random_state=random_state, log=lambda s: None)
    models = {team: pipe for team, (pipe, _, _) in trained.items()}
    log(f"  {len(models)} models trained")

    # The derived skip rule, applied as-of this season.
    skip = {team for team in FBS_SINCE
            if team not in models}

    fbs_full = store.load_all_team_frames("fbs")
    fcs_full = store.load_all_team_frames("fcs")

    weeks = sorted(
        int(w) for (w,) in store.conn.execute(
            'SELECT DISTINCT "Week" FROM team_games WHERE "Year" = ? AND classification = ?',
            (year, "fbs"),
        )
    )

    all_results = []
    for week in weeks:
        preferred = preferred_lines_for(store, fbs_full, year, week)
        if preferred.empty:
            continue
        # Strictly the snapshot captured before this week's first kickoff.
        # Substituting a later week's ratings (or the season-final ones) would
        # leak results of the very games being predicted, so a week without
        # its own snapshot is skipped outright.
        sp = store.load_sp(year, week)
        if sp.empty:
            log(f"  week {week:>2}: SKIPPED — no pre-kickoff SP+ snapshot for this week")
            continue
        talent = store.load_talent(year)

        fbs = {k: truncate_before(v, year, week) for k, v in fbs_full.items()}
        fcs = {k: truncate_before(v, year, week) for k, v in fcs_full.items()}
        fbs = {k: v for k, v in fbs.items() if len(v) >= MIN_FBS_GAMES}
        fcs = {k: v for k, v in fcs.items() if len(v) >= MIN_FCS_GAMES}

        fbs, fcs = build_upcoming_frames(fbs, fcs, preferred, sp, talent, year, week)
        features = feature_columns(next(iter(fbs.values())))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, _ = predict_week(models, fbs, features, skip_teams=skip)
        if preds.empty:
            continue
        preds = preds.copy()
        preds["day"] = "bt"

        graded = grade_week(preds, store.load_bins(), fbs_full, set(fcs_full), log=lambda s: None)
        graded["week"] = week
        all_results.append(graded)
        wins = int((graded["result"] == 1).sum())
        losses = int((graded["result"] == -1).sum())
        log(f"  week {week:>2}: {len(graded)} graded, {wins}-{losses}")

    if not all_results:
        log("Nothing to backtest.")
        return pd.DataFrame(columns=list(RESULTS_COLUMNS) + ["week"])

    season_results = pd.concat(all_results, ignore_index=True)
    season_results.to_csv(out_dir / f"backtest_{year}_results.csv")
    log(f"\n=== {year} backtest ({len(season_results)} games) ===")
    log(tier_breakdown(season_results, label_best_empty="No best bets"))
    log(f"\nSaved -> {out_dir / f'backtest_{year}_results.csv'}")
    return season_results
