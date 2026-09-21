"""Precompute the static JSON the website serves: teams, all matchups, performance.

Because a matchup is deterministic given the current models + DB, every pairing
can be computed up front (~131 teams -> ~8,500 pairs, ~1-2 MB). The website then
serves these files directly with no live model host.

Run (in the weekly GitHub Action, after `update`/`analyze`):

    python scripts/precompute_static.py --out output/static

Outputs teams.json, matchups.json, performance.json into --out.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import multiprocessing
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from cfb_model.analysis.evaluate import tier_records
from cfb_model.config import get_settings
from cfb_model.data.store import Store
from cfb_model.model.predict import predict_matchup
from cfb_model.pipeline import load_models


def _entry_from_result(r: dict) -> dict:
    diff = r["predicted_score_diff"]
    if diff > 0:
        favored, margin = r["team"], diff
    elif diff < 0:
        favored, margin = r["opponent"], -diff
    else:
        favored, margin = None, 0.0
    return {
        "favored_team": favored,
        "margin": round(float(margin), 1),
        "predicted_score_diff": round(float(diff), 1),
    }


# --- parallel worker: each process opens its own store; models are inherited
# from the parent via fork (copy-on-write) so they aren't reloaded per worker.
_W: dict = {}
_PARENT_MODELS: dict | None = None


def _init_worker():
    settings = get_settings()
    _W["store"] = Store(settings.db_path)
    _W["models"] = (
        _PARENT_MODELS
        if _PARENT_MODELS is not None
        else load_models(settings, latest_model_season(settings))
    )


def _predict_pair(pair: tuple[str, str]):
    a, b = pair
    try:
        r = predict_matchup(_W["store"], _W["models"], a, b, log_query=False)
    except Exception as exc:  # noqa: BLE001 — report and skip
        return a, b, None, str(exc)
    return a, b, _entry_from_result(r), None


def latest_model_season(settings) -> int:
    seasons = [
        int(m.group(1))
        for p in glob.glob(str(settings.models_dir / "*_model_*.pkl"))
        if (m := re.search(r"_model_(\d{4})\.pkl$", p))
    ]
    return max(seasons) if seasons else 2024


def build_matchups(store, models: dict, log=print, max_pairs: int | None = None,
                   jobs: int = 1) -> tuple[dict, int]:
    teams = sorted(models.keys())
    pairs = list(itertools.combinations(teams, 2))  # teams sorted -> (a, b) ordered
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    out: dict[str, dict] = {}
    skipped = 0

    def record(a, b, entry, err):
        nonlocal skipped
        if err is not None:
            skipped += 1
            log(f"  skip {a}|{b}: {err}")
        else:
            out[f"{a}|{b}"] = entry

    if jobs <= 1:
        for i, (a, b) in enumerate(pairs):
            try:
                r = predict_matchup(store, models, a, b, log_query=False)
                record(a, b, _entry_from_result(r), None)
            except Exception as exc:  # noqa: BLE001
                record(a, b, None, str(exc))
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(pairs)} pairs...")
    else:
        global _PARENT_MODELS
        _PARENT_MODELS = models  # inherited by forked workers (copy-on-write)
        # fork avoids re-importing pandas/sklearn and reloading models per worker
        # (the default on Linux CI; requested explicitly so macOS behaves too).
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:  # platform without fork -> fall back to default
            ctx = multiprocessing.get_context()
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                 initializer=_init_worker) as pool:
            for i, (a, b, entry, err) in enumerate(pool.map(_predict_pair, pairs, chunksize=32)):
                record(a, b, entry, err)
                if (i + 1) % 500 == 0:
                    log(f"  {i + 1}/{len(pairs)} pairs...")
    return out, skipped


def build_performance(store) -> dict:
    row = store.conn.execute("SELECT MAX(season) FROM results").fetchone()
    if not row or row[0] is None:
        return {"season": None, "season_to_date": None, "last_week": None}
    season = int(row[0])
    season_df = store.load_results(season)
    wk = store.conn.execute(
        "SELECT MAX(week) FROM results WHERE season = ?", (season,)
    ).fetchone()
    last_week = int(wk[0]) if wk and wk[0] is not None else None
    week_df = store.load_results(season, last_week) if last_week is not None else None
    return {
        "season": season,
        "season_to_date": tier_records(season_df) if not season_df.empty else None,
        "last_week": (
            {"week": last_week, "records": tier_records(week_df)}
            if week_df is not None and not week_df.empty
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/static", help="output directory")
    ap.add_argument("--max-pairs", type=int, default=None, help="debug: cap number of pairs")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (use the runner's core count)")
    args = ap.parse_args(argv)

    settings = get_settings()
    store = Store(settings.db_path)
    season = latest_model_season(settings)
    models = load_models(settings, season)
    if not models:
        print(f"No models for season {season}; run setup-season first.", file=sys.stderr)
        return 1

    teams = sorted(models.keys())
    generated_at = datetime.now(timezone.utc).isoformat()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"season {season}, {len(teams)} teams")
    (out_dir / "teams.json").write_text(json.dumps(
        {"season": season, "generated_at": generated_at, "teams": teams}, indent=1))

    print("computing matchups...")
    t0 = time.time()
    matchups, skipped = build_matchups(store, models, max_pairs=args.max_pairs, jobs=args.jobs)
    print(f"  {len(matchups)} matchups ({skipped} skipped) in {time.time() - t0:.0f}s")
    # compact: this file is the largest, keep it small
    (out_dir / "matchups.json").write_text(json.dumps(
        {"season": season, "generated_at": generated_at, "matchups": matchups},
        separators=(",", ":")))

    perf = build_performance(store)
    perf["generated_at"] = generated_at
    (out_dir / "performance.json").write_text(json.dumps(perf, indent=1))

    for name in ("teams.json", "matchups.json", "performance.json"):
        p = out_dir / name
        print(f"  wrote {p} ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
