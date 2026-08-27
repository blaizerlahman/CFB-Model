"""Prediction core — pure functions, no CLI/IO coupling (a future FastAPI
service imports these directly).

Ports of MidSeason's getPreferredLine / grabUpcomingWeekData (sentinel-row
construction, minus Selenium) / predictUpcomingWeek / savePredictions /
printPredictions. Every modeling semantic is preserved; the deliberate
deviations (approved bug fixes) are marked LEGACY-FIX comments.
"""

from __future__ import annotations

import pandas as pd

from cfb_model.constants import PREDS_COLUMNS, PROVIDER_ORDER
from cfb_model.features.assemble import feature_columns, own_rolling_columns
from cfb_model.features.rolling import update_last_row_rolling
from cfb_model.model.classify import lookup_success_rate


def select_preferred_lines(lines: pd.DataFrame,
                           providers: tuple[str, ...] = PROVIDER_ORDER) -> pd.DataFrame:
    """One row per game Id: first provider in preference order, else the
    first available line (exact getPreferredLine semantics)."""

    rank = {p: i for i, p in enumerate(providers)}
    ranked = lines.copy()
    ranked["_rank"] = ranked["LineProvider"].map(rank).fillna(len(providers))
    ranked["_pos"] = range(len(ranked))
    ranked = ranked.sort_values(["Id", "_rank", "_pos"], kind="stable")
    preferred = ranked.drop_duplicates(subset=["Id"], keep="first")
    return preferred.drop(columns=["_rank", "_pos"]).reset_index(drop=True)


def build_upcoming_frames(
    fbs: dict[str, pd.DataFrame],
    fcs: dict[str, pd.DataFrame],
    preferred_lines: pd.DataFrame,
    sp: pd.DataFrame,
    talent: pd.DataFrame,
    year: int,
    week: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Append a sentinel row (NaN School, betting columns filled) per upcoming
    game, recompute rolling sums for it, copy the opponent's _opp features,
    and attach talent/SP. Mutates copies; returns the new dicts.

    Spread sign convention (verbatim): home team's Spread is flipped (-raw),
    away team keeps the raw home-perspective value. The original away-branch
    "double flip" of a shared rowDF produced exactly this.
    """
    fbs = {k: v.copy() for k, v in fbs.items()}
    fcs = {k: v.copy() for k, v in fcs.items()}

    fcs_away_opponents: set[str] = set()

    # --- collect the upcoming rows for each team ---
    # A team can have more than one: CFBD's week 1 covers roughly ten days, so
    # a side opening in week 0 and again the next weekend has both games filed
    # under week 1.
    pending: dict[str, list[pd.DataFrame]] = {}
    for _, row in preferred_lines.iterrows():
        row_df = pd.DataFrame([row])
        home, away = row["HomeTeam"], row["AwayTeam"]
        for name, is_home in ((home, True), (away, False)):
            if name in fbs:
                pass
            elif name in fcs:
                # LEGACY-FIX (bug 3): away-FCS frames were stored into
                # teamDict; here every FCS frame stays in the FCS dict.
                if not is_home:
                    fcs_away_opponents.add(name)
            else:
                continue
            sentinel = row_df.copy()
            if is_home:
                sentinel.iloc[0, sentinel.columns.get_loc("Spread")] = (
                    float(sentinel.iloc[0, sentinel.columns.get_loc("Spread")]) * -1
                )
            pending.setdefault(name, []).append(sentinel)

    sample = next(iter(fbs.values()))
    rolling_cols = own_rolling_columns(sample)

    # --- append each upcoming row with its own rolling sums ---
    # Each is computed against the played games alone, so a second upcoming row
    # sees the same trailing window as the first rather than counting the
    # other one as a game.
    for frames in (fbs, fcs):
        for team, sentinels in pending.items():
            if team not in frames:
                continue
            played = frames[team]
            filled = []
            for sentinel in sentinels:
                temp = pd.concat([played, sentinel], ignore_index=True)
                temp = update_last_row_rolling(temp, rolling_cols)
                filled.append(temp.iloc[[-1]])
            frames[team] = pd.concat([played, *filled], ignore_index=True)

    # --- _opp rolling + own talent/SP, for every upcoming row ---
    # Verbatim control flow, applied per row: the fills are staged and only
    # committed once the talent AND SP lookups succeed. When either comes up
    # empty the notebook discarded its working copy, losing the _opp fill too,
    # and that shows up in predictions, so it is preserved.
    for team in list(fbs):
        df = fbs[team].copy()
        changed = False
        for idx in upcoming_rows(df):
            row = df.loc[idx]
            game_id = row["Id"]
            opp_name = row["AwayTeam"] if team == row["HomeTeam"] else row["HomeTeam"]

            if opp_name in fbs:
                opp_df = fbs[opp_name]
            elif opp_name in fcs:
                opp_df = fcs[opp_name]
            else:
                continue

            staged: dict[str, object] = {}
            opp_row = opp_df[opp_df["Id"] == game_id]
            if not opp_row.empty:
                for col in rolling_cols:
                    staged[col + "_opp"] = opp_row.iloc[0][col]

            if week == 1:
                for col, value in staged.items():
                    df.loc[idx, col] = value
                changed = changed or bool(staged)
                continue

            team_talent = talent.loc[talent["School"] == team, "Talent"]
            if len(team_talent.values) == 0:
                continue  # verbatim: the row's staged fills are discarded
            team_sp = sp.loc[sp["Team"] == team, "Rating"]
            if len(team_sp.values) == 0:
                continue  # verbatim: discarded

            staged["talent"] = team_talent.values[0]
            staged["SP"] = team_sp.values[0]
            for col, value in staged.items():
                df.loc[idx, col] = value
            changed = True
        if changed:
            fbs[team] = df

    # --- opponent talent/SP, for every upcoming row ---
    if week != 1:
        for team in list(fbs):
            df = fbs[team].copy()
            changed = False
            for idx in upcoming_rows(df):
                row = df.loc[idx]
                opp_name = row["AwayTeam"] if team == row["HomeTeam"] else row["HomeTeam"]

                if opp_name in fbs:
                    opp_df = fbs[opp_name]
                elif opp_name in fcs:
                    opp_df = fcs[opp_name]
                else:
                    continue

                if opp_name in fbs or opp_name in fcs_away_opponents:
                    # FBS opponents always; FCS opponents only when they are the
                    # away team — replicating the notebook, where the away-FCS
                    # frame sat in teamDict and took the numeric-Year branch
                    # while home-FCS frames hit a str(year) comparison that
                    # never matched (talent_opp stayed NaN for those).
                    filtered = opp_df[opp_df["Year"] == year]
                    if filtered.empty:
                        continue
                    df.loc[idx, "talent_opp"] = filtered.iloc[0].get("talent", pd.NA)
                    changed = True

                opp_sp = sp[sp["Team"] == opp_name]
                if not opp_sp.empty:
                    df.loc[idx, "SP_opp"] = opp_sp.iloc[0]["Rating"]
                    changed = True
            if changed:
                fbs[team] = df

    if week == 1:
        _seed_week_one(fbs, fcs, sp, talent, year)

    return fbs, fcs


def _seed_week_one(fbs: dict[str, pd.DataFrame], fcs: dict[str, pd.DataFrame],
                   sp: pd.DataFrame, talent: pd.DataFrame, year: int) -> None:
    """grabUpcomingYearTalent + grabUpcomingYearSP, upcoming rows only:
    the new season's talent/SP (and the opponents') come straight from the
    ratings tables rather than from prior rows.

    Week 1 is exactly where a team can hold two upcoming rows, so every one of
    them is seeded rather than only the last.
    """
    for frames in (fbs, fcs):
        for team in list(frames):
            df = frames[team]
            rows = upcoming_rows(df)
            if not rows:
                continue
            df = df.copy()
            for idx in rows:
                df.loc[idx, "Year"] = year

                team_talent = talent.loc[talent["School"] == team, "Talent"]
                if len(team_talent.values):
                    df.loc[idx, "talent"] = team_talent.values[0]

                if frames is fbs:
                    team_sp = sp.loc[sp["Team"] == team, "Rating"]
                    if len(team_sp.values):
                        df.loc[idx, "SP"] = team_sp.values[0]

                    row = df.loc[idx]
                    opp_name = row["AwayTeam"] if team == row["HomeTeam"] else row["HomeTeam"]
                    opp_talent = talent[talent["School"] == opp_name]
                    if not opp_talent.empty:
                        df.loc[idx, "talent_opp"] = opp_talent.iloc[0]["Talent"]
                    opp_sp = sp[sp["Team"] == opp_name]
                    if not opp_sp.empty:
                        df.loc[idx, "SP_opp"] = opp_sp.iloc[0]["Rating"]

            frames[team] = df


def upcoming_rows(frame: pd.DataFrame) -> list:
    """Every not-yet-played row in a team's frame, newest last.

    A team can hold more than one: CFBD's week 1 spans ten days, so a side
    opening in week 0 and again the next weekend has both games filed under
    week 1. Taking only the last row would drop the earlier game.
    """
    sentinel = frame["School"].isna()
    return [frame.index[i] for i, flag in enumerate(sentinel) if flag]


def games_by_id(frames: dict) -> dict:
    """{game id: {team: row}} across every upcoming row in every frame."""
    games: dict = {}
    for team, frame in frames.items():
        for idx in upcoming_rows(frame):
            row = frame.loc[idx]
            game_id = row.get("Id")
            if pd.isna(game_id):
                continue
            games.setdefault(game_id, {})[team] = row
    return games



def resolve_pick(team: str, opp_team: str, spread, cover):
    """Which side to back, and the line it must cover.

    The model speaks from one team's perspective: `cover` = +1 means that team
    beats the spread, -1 means the other side does. A game-level call names the
    side directly. `spread` is stored score-differential style (positive = that
    team is favoured by that much), so the returned line is negated into the
    convention a sportsbook quotes, where the favourite is negative.
    """
    if pd.isna(cover) or pd.isna(spread):
        return None, float("nan")
    if cover >= 0:                      # a push still leaves the call on `team`
        return team, -float(spread)
    return opp_team, float(spread)


def predict_week(
    model_dict: dict,
    fbs_frames: dict[str, pd.DataFrame],
    features: list[str] | None = None,
    skip_teams: set[str] | frozenset[str] = frozenset(),
    average_sides: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Exact port of predictUpcomingWeek + savePredictions.

    Returns (preds frame in legacy column order, status dict for teams
    without a prediction row).

    LEGACY-FIX (bug 2): the skip check now tests both participants' names
    (the notebook compared a boolean against the list, so only the opponent
    check ever fired). LEGACY-FIX (bug 4): no hardcoded Baylor exception.
    """
    statuses: dict[str, str] = {}
    rows: list[list] = []
    games = games_by_id(fbs_frames)

    for team in sorted(model_dict):
        frame = fbs_frames.get(team)
        if frame is None or not upcoming_rows(frame):
            statuses[team] = "No game this week"

    # Keyed by game rather than by each team's last row, so a side playing
    # twice in one week gets both games predicted instead of just the later.
    for game_id, sides in sorted(games.items(), key=lambda kv: (sorted(kv[1])[0], kv[0])):
        modelled = [t for t in sorted(sides) if t in model_dict]
        if not modelled:
            continue
        owner = modelled[0]
        row = sides[owner]
        opp_team = row["AwayTeam"] if row["HomeTeam"] == owner else row["HomeTeam"]

        # The alphabetically-first modelled side still owns the row, but with
        # averaging both models feed the same game-level call, so there is no
        # longer a meaningful sense in which the other team "was not predicted".

        if owner in skip_teams or opp_team in skip_teams:
            statuses[f"{owner} vs {opp_team}"] = "no pick — a side has incomplete data"
            continue

        if features is None:
            features = feature_columns(fbs_frames[owner])

        if average_sides and len(modelled) > 1:
            values = []
            for team in modelled:
                value = model_dict[team].predict(sides[team][features].to_frame().T)[0]
                values.append(value if team == owner else -value)
            pred = round(sum(values) / len(values) * 2) / 2
        else:
            pred = model_dict[owner].predict(row[features].to_frame().T)
            pred = round(pred[0] * 2) / 2  # builtin round, verbatim

        spread = row["Spread"]
        spread_diff = pred - spread
        cover = (float("nan") if pd.isna(spread_diff)
                 else -1 if spread_diff < 0 else 1 if spread_diff > 0 else 0)
        pick, pick_spread = resolve_pick(owner, opp_team, spread, cover)
        rows.append([pred, spread, spread_diff, cover, game_id, owner, opp_team,
                     pick, pick_spread])

    preds = pd.DataFrame(rows, columns=list(PREDS_COLUMNS))
    return preds, statuses


def predict_matchup(
    store,
    models: dict,
    team1: str,
    team2: str,
    spread: float | None = None,
    year: int | None = None,
    week: int | None = None,
) -> dict:
    """On-demand prediction for an arbitrary pairing, from current DB state.

    Uses the production semantics: the alphabetically-earlier team's model
    owns the game (falling back to the other model if the owner has none).
    `spread` is from team1's perspective (negative = team1 favored); without
    it only the predicted score differential is returned. Logged to
    matchup_queries only — NEVER to the predictions table, and success
    analyses never see these.
    """
    import numpy as np

    from cfb_model.model.classify import lookup_success_rate, tier

    owner, other = sorted([team1, team2])
    if owner not in models and other in models:
        owner, other = other, owner
    if owner not in models:
        raise ValueError(f"No model for either {team1} or {team2}")

    frames = {}
    for name in (team1, team2):
        frame = store.load_team_frame(name)
        if frame is None:
            raise ValueError(f"No stored data for {name}")
        frames[name] = frame[frame["School"].notna()].reset_index(drop=True)

    if year is None:
        year = int(max(f["Year"].max() for f in frames.values()))
    sp_weeks = store.sp_weeks(year)
    sp = store.load_sp(year, sp_weeks[-1]) if sp_weeks else None
    talent = store.load_talent(year)

    # Synthetic line row: team1 as home. Raw (home-perspective) spread is the
    # team1-perspective value; the standard build flips it for the home team.
    from cfb_model.api.mapping import BETTING_COLUMNS

    row = {c: np.nan for c in BETTING_COLUMNS}
    row.update({
        "Id": -1.0,
        "HomeTeam": team1,
        "AwayTeam": team2,
        "LineProvider": "matchup",
        "Spread": -spread if spread is not None and team1 == owner else
                  (spread if spread is not None else np.nan),
    })
    preferred = pd.DataFrame([row], columns=list(BETTING_COLUMNS))

    fbs = {k: v for k, v in frames.items()}
    fbs, _ = build_upcoming_frames(
        fbs, {}, preferred, sp if sp is not None else pd.DataFrame(columns=["Team", "Rating"]),
        talent, year, week if week is not None else 99,
    )

    features = feature_columns(fbs[owner])
    preds, _ = predict_week({owner: models[owner]}, fbs, features)
    if preds.empty:
        raise RuntimeError("Matchup prediction produced no output")
    p = preds.iloc[0]

    result = {
        "team": p["team"],
        "opponent": p["oppTeam"],
        "predicted_score_diff": float(p["pred"]),
        "spread": None if pd.isna(p["spread"]) else float(p["spread"]),
        "spread_diff": None if pd.isna(p["spreadDiff"]) else float(p["spreadDiff"]),
        "cover": None if pd.isna(p["cover"]) else int(p["cover"]),
    }
    if result["spread_diff"] is not None:
        bins = store.load_bins()
        rate = lookup_success_rate(result["spread_diff"], bins)
        result["success_rate"] = rate
        result["tier"] = tier(rate) if rate is not None else None

    with store.conn:
        store.conn.execute(
            "INSERT INTO matchup_queries (team1, team2, spread, pred, spread_diff, success_rate)"
            " VALUES (?,?,?,?,?,?)",
            (team1, team2, spread, result["predicted_score_diff"],
             result["spread_diff"], result.get("success_rate")),
        )
    return result


def classification_report(
    preds: pd.DataFrame,
    bins: pd.DataFrame,
    fbs_frames: dict[str, pd.DataFrame],
    fcs_frames: dict[str, pd.DataFrame],
    statuses: dict[str, str] | None = None,
) -> str:
    """One line per game, naming the side to back and the line it must cover.

    Grouped into the same exclusive tiers as before (>70%, 65-70%, 60-65%,
    <60%), sorted by success rate, with toss-ups separate. Games against FCS
    opposition are left out, as they always were.
    """
    legend = (
        "Each line: the side to back and its line, the margin this model "
        "projects for that\nside, and the historical hit rate for a "
        "disagreement of this size.\n"
    )
    best, great, good, normal, toss_up = [], [], [], [], []

    for _, p in preds.iterrows():
        if p["oppTeam"] in fcs_frames or p["oppTeam"] not in fbs_frames:
            continue

        spread_diff = p["spreadDiff"]
        if pd.isna(spread_diff):
            continue
        cover = 1 if spread_diff > 0 else -1 if spread_diff < 0 else 0

        pick = p.get("pick")
        pick_spread = p.get("pickSpread")
        if pick is None or (isinstance(pick, float) and pd.isna(pick)):
            pick, pick_spread = resolve_pick(p["team"], p["oppTeam"], p["spread"], cover)
        other = p["oppTeam"] if pick == p["team"] else p["team"]
        # Projected margin from the backed side's point of view.
        margin = p["pred"] if pick == p["team"] else -p["pred"]

        if cover == 0:
            toss_up.append((pick, other, pick_spread, margin))
            continue

        rate = lookup_success_rate(spread_diff, bins)
        if rate is None:
            continue
        entry = (pick, other, pick_spread, margin, abs(spread_diff), rate)
        if rate < 0.595:
            normal.append(entry)
        elif rate <= 0.645:
            good.append(entry)
        elif rate <= 0.695:
            great.append(entry)
        else:
            best.append(entry)

    for bucket in (normal, good, great, best):
        bucket.sort(key=lambda x: x[5], reverse=True)

    lines_out: list[str] = [legend]

    def emit(title: str, bucket: list, empty_msg: str) -> None:
        lines_out.append(title)
        if not bucket:
            lines_out.append(empty_msg)
        else:
            for pick, other, line, margin, edge, rate in bucket:
                lines_out.append(
                    f"  {pick} {line:+g} vs {other}  |  model {margin:+g}  |  "
                    f"{rate * 100:.2f}%"
                )
        lines_out.append("")

    emit("Games with a greater than 70% success rate:", best,
         "  No games above a 70% success rate this week.")
    emit("Games with a 65-70% success rate:", great,
         "  No games with a 65-70% success rate this week.")
    emit("Games with a 60-65% success rate:", good,
         "  No games with a 60-65% success rate this week.")
    emit("Games with a less than 60% success rate:", normal,
         "  No games below a 60% success rate this week.")

    lines_out.append("Toss-up games:")
    if not toss_up:
        lines_out.append("  No toss-up games this week.")
    else:
        for pick, other, line, margin in toss_up:
            lines_out.append(f"  {pick} {line:+g} vs {other}  |  projected {margin:+g}  |  toss up")

    if statuses:
        skipped = {t: st for t, st in statuses.items()
                   if st not in ("No game this week",)}
        if skipped:
            lines_out.append("")
            for team, status in skipped.items():
                lines_out.append(f"{team}: {status}")

    return "\n".join(lines_out)
