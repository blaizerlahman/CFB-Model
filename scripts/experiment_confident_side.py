"""Experiment: predict a game from both teams and bet the more confident side.

The shipped pipeline asks only one model per game — whichever team sorts
first — and bets whatever it says. Here both teams' models are asked, each
prediction is scored against the calibration table, and the side with the
higher historical success rate becomes the pick. The other is discarded.

This is the "is the disagreement as extreme from the other side?" idea: when
one model produces a wild differential and the other does not, this backs the
one the calibration likes.

Read-only: no package behaviour is changed, nothing is written to the
predictions or results tables, and models are trained in memory.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cfb_model.analysis.backtest import preferred_lines_for, truncate_before  # noqa: E402
from cfb_model.analysis.evaluate import grade_week  # noqa: E402
from cfb_model.config import get_settings  # noqa: E402
from cfb_model.constants import MIN_FBS_GAMES, MIN_FCS_GAMES, PREDS_COLUMNS  # noqa: E402
from cfb_model.data.history import FBS_SINCE  # noqa: E402
from cfb_model.data.store import Store  # noqa: E402
from cfb_model.features.assemble import feature_columns  # noqa: E402
from cfb_model.model.classify import lookup_success_rate  # noqa: E402
from cfb_model.model.predict import build_upcoming_frames, games_by_id  # noqa: E402
from cfb_model.model.train import train_all  # noqa: E402

BREAK_EVEN = 0.5238


def confident_side_picks(models, fbs, features, bins, skip_teams):
    """One pick per game: whichever side's prediction the bins rate higher."""
    rows, diagnostics = [], []
    for game_id, sides in games_by_id(fbs).items():
        modelled = [t for t in sorted(sides) if t in models]
        if len(modelled) < 2:
            continue
        first = sides[modelled[0]]
        opp_of_first = (first["AwayTeam"] if first["HomeTeam"] == modelled[0]
                        else first["HomeTeam"])
        if modelled[0] in skip_teams or opp_of_first in skip_teams:
            continue

        options = []
        for team in modelled:
            row = sides[team]
            pred = round(models[team].predict(row[features].to_frame().T)[0] * 2) / 2
            spread = row["Spread"]
            if pd.isna(spread):
                continue
            diff = pred - spread
            rate = lookup_success_rate(diff, bins)
            if rate is None:
                continue
            opp = row["AwayTeam"] if row["HomeTeam"] == team else row["HomeTeam"]
            options.append({"team": team, "opp": opp, "pred": pred,
                            "spread": spread, "diff": diff, "rate": rate})
        if len(options) < 2:
            continue

        # Back the side the calibration rates higher; ties go to the first
        # team alphabetically so the choice is deterministic.
        best = max(options, key=lambda o: (o["rate"], -ord(o["team"][0])))
        other = [o for o in options if o["team"] != best["team"]][0]
        cover = (np.nan if pd.isna(best["diff"])
                 else -1 if best["diff"] < 0 else 1 if best["diff"] > 0 else 0)
        rows.append([best["pred"], best["spread"], best["diff"], cover,
                     game_id, best["team"], best["opp"]])
        diagnostics.append({"gameID": game_id, "chosen": best["team"],
                            "chosen_rate": best["rate"], "other_rate": other["rate"],
                            "rate_gap": best["rate"] - other["rate"],
                            "chosen_diff": best["diff"], "other_diff": other["diff"],
                            "sides_agree": np.sign(best["diff"]) != np.sign(other["diff"])})
    return pd.DataFrame(rows, columns=list(PREDS_COLUMNS)), pd.DataFrame(diagnostics)


def run_season(store, year, settings, log=print):
    log(f"\n=== {year}: training as-of models (seed 50) ===")
    trained = train_all(store, year, random_state=50, log=lambda m: None)
    models = {t: pipe for t, (pipe, _, _) in trained.items()}
    log(f"  {len(models)} models")

    bins = store.load_bin_set(settings.bin_set)
    skip = {t for t in FBS_SINCE if t not in models}
    fbs_full = store.load_all_team_frames("fbs")
    fcs_full = store.load_all_team_frames("fcs")
    weeks = sorted(int(w) for (w,) in store.conn.execute(
        'SELECT DISTINCT "Week" FROM team_games WHERE "Year" = ? AND classification = ?',
        (year, "fbs")))

    graded_all, diag_all = [], []
    for week in weeks:
        preferred = preferred_lines_for(store, fbs_full, year, week)
        sp = store.load_sp(year, week)
        if preferred.empty or sp.empty:
            continue
        talent = store.load_talent(year)
        fbs = {k: truncate_before(v, year, week) for k, v in fbs_full.items()}
        fcs = {k: truncate_before(v, year, week) for k, v in fcs_full.items()}
        fbs = {k: v for k, v in fbs.items() if len(v) >= MIN_FBS_GAMES}
        fcs = {k: v for k, v in fcs.items() if len(v) >= MIN_FCS_GAMES}
        fbs, fcs = build_upcoming_frames(fbs, fcs, preferred, sp, talent, year, week)
        features = feature_columns(next(iter(fbs.values())))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds, diag = confident_side_picks(models, fbs, features, bins, skip)
        if preds.empty:
            continue
        preds = preds.copy()
        preds["day"] = "bt"
        graded = grade_week(preds, bins, fbs_full, set(fcs_full), log=lambda m: None)
        graded["week"] = week
        graded_all.append(graded)
        diag["week"] = week
        diag_all.append(diag)
        w = int((graded.result == 1).sum()); l = int((graded.result == -1).sum())
        log(f"  week {week:>2}: {len(graded)} picks, {w}-{l}")

    if not graded_all:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(graded_all, ignore_index=True), pd.concat(diag_all, ignore_index=True)


def tiers(df, label):
    out = [f"{label:<34}"]
    for name, thr in (("all", None), ("good", 0.595), ("great", 0.645), ("best", 0.695)):
        sub = df if thr is None else df[df.successRate >= thr]
        w = int((sub.result == 1).sum()); l = int((sub.result == -1).sum())
        if w + l == 0:
            out.append(f"{'(none)':>20}")
        else:
            rate = w / (w + l)
            out.append(f"{f'{w}-{l} {rate*100:.1f}%' + ('+' if rate >= BREAK_EVEN else ' '):>20}")
    return "".join(out)


def main():
    settings = get_settings()
    store = Store(settings.db_path)
    out_dir = settings.output_root / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, diags = {}, {}
    for year in (2024, 2025):
        graded, diag = run_season(store, year, settings)
        if graded.empty:
            continue
        frames[year] = graded
        diags[year] = diag
        graded.to_csv(out_dir / f"confident_side_{year}.csv")

    print("\n\n=== PICK THE MORE CONFIDENT SIDE ===\n")
    header = f"{'run':<34}" + "".join(f"{t:>20}" for t in ("all", "good", "great", "best"))
    print(header); print("-" * len(header))
    for year, df in frames.items():
        print(tiers(df, f"{year} confident side"))
    if frames:
        pooled = pd.concat(frames.values(), ignore_index=True)
        print(tiers(pooled, "POOLED confident side"))

    root = settings.output_root / "backtests"
    base = [root / "2024" / "backtest_2024_results.csv", root / "2025" / "backtest_2025_results.csv"]
    avg = [root / "2024" / "backtest_2024_bothsides_results.csv",
           root / "2025" / "backtest_2025_bothsides_results.csv"]
    for label, paths in (("POOLED one model (current)", base), ("POOLED both averaged", avg)):
        got = [pd.read_csv(p) for p in paths if p.exists()]
        if got:
            print(tiers(pd.concat(got, ignore_index=True), label))
    print(f"\n'+' marks a tier at or above the {BREAK_EVEN*100:.2f}% break-even at -110.")

    if diags:
        d = pd.concat(diags.values(), ignore_index=True)
        d.to_csv(out_dir / "confident_side_diagnostics.csv")
        print(f"\nHow often the two models picked the SAME side: "
              f"{d.sides_agree.mean()*100:.1f}% of {len(d)} games")
        print(f"Median gap between the chosen and rejected success rate: "
              f"{d.rate_gap.median()*100:.1f} pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
