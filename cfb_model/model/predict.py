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

    # --- sentinel rows ---
    for _, row in preferred_lines.iterrows():
        row_df = pd.DataFrame([row])

        home, away = row["HomeTeam"], row["AwayTeam"]

        for name, is_home in ((home, True), (away, False)):
            if name in fbs:
                target = fbs
            elif name in fcs:
                # LEGACY-FIX (bug 3): away-FCS frames were stored into
                # teamDict; here every FCS frame stays in the FCS dict.
                target = fcs
                if not is_home:
                    fcs_away_opponents.add(name)
            else:
                continue
            sentinel = row_df.copy()
            if is_home:
                sentinel.iloc[0, sentinel.columns.get_loc("Spread")] = (
                    float(sentinel.iloc[0, sentinel.columns.get_loc("Spread")]) * -1
                )
            frame = pd.concat([target[name], sentinel], ignore_index=True)
            target[name] = frame.reset_index(drop=True)

    sample = next(iter(fbs.values()))
    rolling_cols = own_rolling_columns(sample)

    # --- own rolling sums on sentinel rows ---
    for frames in (fbs, fcs):
        for team, df in frames.items():
            if pd.isna(df.iloc[-1]["School"]) == False:  # noqa: E712 (verbatim)
                continue
            frames[team] = update_last_row_rolling(df, rolling_cols)

    # --- _opp rolling + own talent/SP (FBS frames with a sentinel only) ---
    # Verbatim control flow: work on a copy; if the talent or SP lookup comes
    # up empty the copy is DISCARDED (the _opp fill is lost too). That is the
    # notebook's behavior and it affects predictions, so it is preserved.
    for team in list(fbs):
        if pd.isna(fbs[team].iloc[-1]["School"]) == False:  # noqa: E712
            continue

        df = fbs[team].copy()
        last_index = df.index[-1]
        last_row = df.iloc[-1]
        game_id = last_row["Id"]
        opp_name = last_row["AwayTeam"] if team == last_row["HomeTeam"] else last_row["HomeTeam"]

        if opp_name in fbs:
            opp_df = fbs[opp_name]
        elif opp_name in fcs:
            opp_df = fcs[opp_name]
        else:
            continue

        opp_row = opp_df[opp_df["Id"] == game_id]
        if not opp_row.empty:
            for col in rolling_cols:
                df.loc[last_index, col + "_opp"] = opp_row.iloc[0][col]

        if week == 1:
            fbs[team] = df
            continue

        team_talent = talent.loc[talent["School"] == team, "Talent"]
        if len(team_talent.values) == 0:
            continue  # verbatim: copy (incl. _opp fill) discarded
        df.loc[last_index, "talent"] = team_talent.values[0]

        team_sp = sp.loc[sp["Team"] == team, "Rating"]
        if len(team_sp.values) == 0:
            continue  # verbatim: copy discarded
        df.loc[last_index, "SP"] = team_sp.values[0]

        fbs[team] = df

    # --- opponent talent/SP on sentinel rows ---
    # (The notebook loop also touched non-playing teams' last real rows; that
    # mutation never influenced predictions or persisted state, so only
    # sentinel rows are processed here.)
    if week != 1:
        for team in list(fbs):
            df = fbs[team]
            if pd.isna(df.iloc[-1]["School"]) == False:  # noqa: E712
                continue
            df = df.copy()
            last_index = df.index[-1]
            last_row = df.iloc[-1]
            opp_name = last_row["AwayTeam"] if team == last_row["HomeTeam"] else last_row["HomeTeam"]

            if opp_name in fbs:
                opp_df = fbs[opp_name]
            elif opp_name in fcs:
                opp_df = fcs[opp_name]
            else:
                continue

            if opp_name in fbs or opp_name in fcs_away_opponents:
                # FBS opponents always; FCS opponents only when they are the
                # away team — replicating the notebook, where the away-FCS
                # frame sat in teamDict and took the numeric-Year branch while
                # home-FCS frames hit a str(year) comparison that never
                # matched (talent_opp stayed NaN for those).
                filtered = opp_df[opp_df["Year"] == year]
                if filtered.empty:
                    continue
                opp_first = filtered.iloc[0]
                df.loc[last_index, "talent_opp"] = opp_first.get("talent", pd.NA)

            opp_sp = sp[sp["Team"] == opp_name]
            if not opp_sp.empty:
                df.loc[last_index, "SP_opp"] = opp_sp.iloc[0]["Rating"]

            fbs[team] = df

    if week == 1:
        _seed_week_one(fbs, fcs, sp, talent, year)

    return fbs, fcs


def _seed_week_one(fbs: dict[str, pd.DataFrame], fcs: dict[str, pd.DataFrame],
                   sp: pd.DataFrame, talent: pd.DataFrame, year: int) -> None:
    """grabUpcomingYearTalent + grabUpcomingYearSP, sentinel rows only:
    the new season's talent/SP (and the opponents') come straight from the
    ratings tables rather than from prior rows."""
    for frames in (fbs, fcs):
        for team in list(frames):
            df = frames[team]
            if pd.isna(df.iloc[-1]["School"]) == False:  # noqa: E712
                continue
            df = df.copy()
            last_index = df.index[-1]
            df.loc[last_index, "Year"] = year

            team_talent = talent.loc[talent["School"] == team, "Talent"]
            if len(team_talent.values):
                df.loc[last_index, "talent"] = team_talent.values[0]

            if frames is fbs:
                team_sp = sp.loc[sp["Team"] == team, "Rating"]
                if len(team_sp.values):
                    df.loc[last_index, "SP"] = team_sp.values[0]

                last_row = df.iloc[-1]
                opp_name = last_row["AwayTeam"] if team == last_row["HomeTeam"] else last_row["HomeTeam"]
                opp_talent = talent[talent["School"] == opp_name]
                if not opp_talent.empty:
                    df.loc[last_index, "talent_opp"] = opp_talent.iloc[0]["Talent"]
                opp_sp = sp[sp["Team"] == opp_name]
                if not opp_sp.empty:
                    df.loc[last_index, "SP_opp"] = opp_sp.iloc[0]["Rating"]

            frames[team] = df


def predict_week(
    model_dict: dict,
    fbs_frames: dict[str, pd.DataFrame],
    features: list[str] | None = None,
    skip_teams: set[str] | frozenset[str] = frozenset(),
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

    sorted_models = {key: model_dict[key] for key in sorted(model_dict)}

    for team, model in sorted_models.items():
        frame = fbs_frames[team]
        last = frame.iloc[-1]

        if pd.isna(last["School"]) == False:  # noqa: E712 (verbatim)
            statuses[team] = "No game this week"
            continue

        is_home = last["HomeTeam"] == team
        opp_team = last["AwayTeam"] if is_home else last["HomeTeam"]

        if opp_team < team and opp_team in sorted_models:
            statuses[team] = f"Already predicted by {opp_team}"
            continue

        if features is None:
            features = feature_columns(frame)

        X = last[features].to_frame().T
        pred = model.predict(X)
        pred = round(pred[0] * 2) / 2  # built-in round, verbatim

        spread = last["Spread"]
        spread_diff = pred - spread
        game_id = last["Id"]

        if team in skip_teams or opp_team in skip_teams:
            statuses[team] = "Playing team with incomplete data"
            continue

        if pd.isna(spread_diff):
            cover = float("nan")
        else:
            cover = -1 if spread_diff < 0 else 1 if spread_diff > 0 else 0

        rows.append([pred, spread, spread_diff, cover, game_id, team, opp_team])

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
    """Port of printPredictions: exclusive tiers, sorted by success rate
    descending, toss-ups separate, FCS games excluded."""
    best, great, good, normal, toss_up = [], [], [], [], []

    for _, p in preds.iterrows():
        if p["oppTeam"] in fcs_frames or p["oppTeam"] not in fbs_frames:
            continue

        spread_diff = p["spreadDiff"]
        cover = 1 if spread_diff > 0 else -1 if spread_diff < 0 else 0
        if cover == 0:
            toss_up.append((p["team"], p["spread"], p["pred"]))
            continue

        rate = lookup_success_rate(spread_diff, bins)
        if rate is None:
            continue

        entry = (p["team"], cover, p["pred"], rate, p["spread"])
        if rate < 0.595:
            normal.append(entry)
        elif rate <= 0.645:
            good.append(entry)
        elif rate <= 0.695:
            great.append(entry)
        else:
            best.append(entry)

    for bucket in (normal, good, great, best):
        bucket.sort(key=lambda x: x[3], reverse=True)

    lines_out: list[str] = []

    def emit(title: str, bucket: list, empty_msg: str) -> None:
        lines_out.append(title)
        if not bucket:
            lines_out.append(empty_msg)
        else:
            for team, cover, pred, rate, spread in bucket:
                verb = "NOT COVER" if cover == -1 else "COVER"
                lines_out.append(
                    f"{team}: {verb} with spread {spread}. Predicted score differential: "
                    f"{pred}. Historical success rate: {rate * 100:.2f}%."
                )
        lines_out.append("")

    emit("Games with a greater than 70% success rate:", best,
         "No games above a 70% success rate this week.")
    emit("Games with a 65-70% success rate:", great,
         "No games with a 65-70% success rate this week.")
    emit("Games with a 60-65% success rate:", good,
         "No games with a 60-65% success rate this week.")
    emit("Games with a less than 60% success rate:", normal,
         "No games below a 60% success rate this week.")

    lines_out.append("Toss-up games:")
    if not toss_up:
        lines_out.append("No toss-up games this week.")
    else:
        for team, spread, pred in toss_up:
            lines_out.append(f"{team}: TOSS UP with spread {spread}. Predicted score differential: {pred}.")

    if statuses:
        lines_out.append("")
        for team, status in statuses.items():
            if status == "No game this week":
                continue
            lines_out.append(f"{team}: {status}")

    return "\n".join(lines_out)
