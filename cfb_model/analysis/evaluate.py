"""Success analysis — ports of evaluatePreds / evaluateSeason and the
season-slice cells from MidSeason (day-of-week, since-week-N, big spreads).

Grading semantics (verbatim): actual cover = sign(scoreDiff - spread) from
the finished game; result = 0 on push else 1 when the stored cover call
matched, -1 otherwise; successRate re-derived from the bins at grading time;
only the highest-successRate day-version of each game is kept; FCS games are
excluded. The good/great/best evaluation filters are CUMULATIVE (>= tier),
unlike the exclusive buckets of the prediction-time report.
"""

from __future__ import annotations

import pandas as pd

from cfb_model.constants import (
    BIG_SPREAD,
    SINCE_WEEK_RANGE,
    THRESH_BEST,
    THRESH_GOOD,
    THRESH_GREAT,
)
from cfb_model.model.classify import lookup_success_rate
from cfb_model.model.predict import resolve_pick

# Chronological position of a day tag within a betting week (sun..sat).
DAY_CHRONOLOGY = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}


def tier_breakdown(df: pd.DataFrame, label_best_empty: str = "No best bets this week") -> str:
    """The overall/good/great/best win-rate block, shared by week, season,
    and since-week reports (verbatim print format)."""
    lines = []
    wins = len(df[df["result"] == 1])
    losses = len(df) - wins
    win_rate = wins / len(df) if len(df) else float("nan")
    lines.append(f"Overall win rate: {win_rate}, {wins}-{losses}")

    for name, thresh in (("good", THRESH_GOOD), ("great", THRESH_GREAT)):
        bets = df[df["successRate"] >= thresh]
        n_wins = len(bets[bets["result"] == 1])
        n_losses = len(bets[bets["result"] == -1])
        rate = n_wins / len(bets) if len(bets) else float("nan")
        lines.append(f"Win rate on {name} bets: {rate}, {n_wins}-{n_losses}")

    best = df[df["successRate"] >= THRESH_BEST]
    if len(best) != 0:
        n_wins = len(best[best["result"] == 1])
        n_losses = len(best[best["result"] == -1])
        lines.append(f"Win rate on best bets: {n_wins / len(best)}, {n_wins}-{n_losses}")
    else:
        lines.append(label_best_empty)
    return "\n".join(lines)


def grade_week(
    preds: pd.DataFrame,
    bins: pd.DataFrame,
    fbs_frames: dict[str, pd.DataFrame],
    fcs_keys: set[str],
    log=print,
) -> pd.DataFrame:
    """evaluatePreds core: grade every day-version, keep the best per game,
    FBS-only. `preds` needs columns pred/spread/spreadDiff/cover/gameID/team/
    oppTeam/day. Returns the filtered results frame (legacy column order)."""
    preds = preds.reset_index(drop=True).copy()
    preds["scoreDiff"] = pd.NA
    preds["result"] = pd.NA
    preds["successRate"] = pd.NA

    fbs_only = []
    for index, game in preds.iterrows():
        called_cover = game["cover"]
        spread_diff = game["spreadDiff"]
        game_id = game["gameID"]
        team = game["team"]
        opp = game["oppTeam"]

        fbs_only.append(not (opp in fcs_keys or opp not in fbs_frames))

        team_df = fbs_frames.get(team)
        if team_df is None or game_id not in team_df["Id"].values:
            log(f"Could not find game id {game_id} in {team} dataframe.")
            continue
        game_row = team_df.loc[team_df["Id"] == game_id]
        spread = game_row["Spread"].values[0]
        score_diff = game_row["scoreDiff"].values[0]

        cover = 1 if score_diff > spread else -1 if score_diff < spread else 0
        result = 0 if cover == 0 else 1 if called_cover == cover else -1

        preds.at[index, "scoreDiff"] = score_diff
        preds.at[index, "result"] = result

        rate = lookup_success_rate(spread_diff, bins)
        if rate is not None:
            preds.at[index, "successRate"] = rate
        else:
            log(f"Could not find predBin for {team} game.")

    clean = preds.dropna(subset=["successRate"])
    # Highest successRate per game; ties (line didn't move between runs) go
    # to the LATEST day of the betting week — the most-informed line, and the
    # behavior observed in the stored legacy results.
    order = clean["day"].map(DAY_CHRONOLOGY).fillna(-1)
    clean = clean.loc[order.sort_values(ascending=False, kind="stable").index]
    best_version = clean.groupby("gameID")["successRate"].idxmax()
    filtered = clean.loc[best_version]
    filtered = filtered.sort_index()

    mask = pd.Series(fbs_only, index=preds.index)
    filtered = filtered[mask.loc[filtered.index]]

    # Rows stored before the pick columns existed still grade; the call is
    # recoverable from team/oppTeam/spread/cover.
    if "pick" not in filtered.columns or filtered["pick"].isna().any():
        picks, lines = [], []
        for _, r in filtered.iterrows():
            existing = r.get("pick")
            if isinstance(existing, str) and existing:
                picks.append(existing)
                lines.append(r.get("pickSpread"))
                continue
            pick, line = resolve_pick(r["team"], r["oppTeam"], r["spread"], r["cover"])
            picks.append(pick)
            lines.append(line)
        filtered = filtered.copy()
        filtered["pick"] = picks
        filtered["pickSpread"] = lines

    columns = ["pred", "spread", "spreadDiff", "cover", "gameID", "team",
               "oppTeam", "pick", "pickSpread", "day", "scoreDiff", "result", "successRate"]
    return filtered[columns].astype({"scoreDiff": float, "result": float, "successRate": float})


def day_breakdown(results: pd.DataFrame) -> str:
    grouped = results.groupby("day").agg(
        total_games=("result", "size"),
        win_rate=("result", lambda x: (x == 1).mean()),
    )
    good = results[results["successRate"] >= THRESH_GOOD]
    good_grouped = good.groupby("day").agg(
        good_games=("result", "size"),
        win_rate=("result", lambda x: (x == 1).mean()),
    )
    return f"{grouped}\n\n{good_grouped}"


def since_week_report(results: pd.DataFrame) -> str:
    blocks = []
    for week in range(SINCE_WEEK_RANGE):
        temp = results[results["week"] >= week]
        if temp.empty:
            continue
        blocks.append(f"WIN RATE SINCE WEEK {week} \n\n" + tier_breakdown(temp))
    return "\n\n".join(blocks)


def big_spread_report(results: pd.DataFrame) -> str:
    big = results[results["spread"].abs() >= BIG_SPREAD]
    if big.empty:
        return f"No games with |spread| >= {BIG_SPREAD}."
    return f"Games with |spread| >= {BIG_SPREAD}:\n" + tier_breakdown(big)
