"""Training — verbatim port of p6's buildModel (the production trainer).

p6 facts (verified against the notebook source, they differ from p4/p5):
  * frames are NOT warm-up-dropped at load (>=56-row gate only);
  * fixCols = features matching the legacy-missing patterns PLUS 'talent'
    and 'SP' substrings (so SP/SP_opp are zero-as-missing imputed too);
  * predictMissing runs on the FULL frame before the Year < season filter;
  * production randomState is None (unseeded) — seeds are for backtests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import BaggingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cfb_model.config import Settings, get_settings
from cfb_model.constants import (
    ALPHA_GRID_ARGS,
    BAGGING_N_ESTIMATORS,
    GRIDSEARCH_CV,
    GRIDSEARCH_SCORING,
    LASSO_MAX_ITER,
    LEGACY_ZERO_AS_MISSING_PATTERNS,
    MIN_FBS_GAMES,
    MIN_TRAIN_ROWS,
)
from cfb_model.data.store import Store
from cfb_model.features.assemble import feature_columns, predict_missing

TRAIN_FIX_PATTERNS = LEGACY_ZERO_AS_MISSING_PATTERNS + ("talent", "SP")


def training_fix_columns(features: list[str]) -> list[str]:
    """p6's fixCols: feature columns matching the legacy-missing patterns
    plus talent/SP (both own and _opp variants match by substring)."""
    return [c for c in features if any(w in c for w in TRAIN_FIX_PATTERNS)]


def train_team_model(team_df, features: list[str], season: int,
                     random_state=None):
    """Exact p6 buildModel. Returns (pipeline, best_alpha, n_train_rows) or
    None when fewer than 8 training rows exist."""
    team = team_df.copy()
    predict_missing(team, training_fix_columns(features), random_state)

    model_team = team[team["Year"] < season]

    X = model_team[features]
    y = model_team[["scoreDiff"]].values.ravel()
    if len(y) < MIN_TRAIN_ROWS:
        return None

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("regressor", Lasso()),
    ])
    grid = GridSearchCV(
        pipe,
        {"regressor__alpha": np.logspace(*ALPHA_GRID_ARGS)},
        cv=GRIDSEARCH_CV,
        scoring=GRIDSEARCH_SCORING,
    )
    grid.fit(X, y)
    best_alpha = grid.best_params_["regressor__alpha"]

    better_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
        ("regressor", BaggingRegressor(
            estimator=Lasso(alpha=best_alpha, max_iter=LASSO_MAX_ITER),
            n_estimators=BAGGING_N_ESTIMATORS,
            n_jobs=-1,
            random_state=random_state,   # None in production == legacy
        )),
    ])
    better_pipe.fit(X, y)
    return better_pipe, float(best_alpha), int(len(y))


def train_all(store: Store, season: int, random_state=None, log=print,
              exclude_features: tuple[str, ...] = ()) -> dict:
    """Train every FBS team passing the >=56-game gate, as-of `season`:
    frames are truncated to Year < season BEFORE training so predictMissing
    sees exactly what p6 saw (its frames ended at the prior season). Returns
    {team_key: (pipeline, best_alpha, n_rows)}."""
    import warnings

    frames = store.load_all_team_frames("fbs")
    trained: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for team, df in sorted(frames.items()):
            df = df[df["Year"] < season].reset_index(drop=True)
            if len(df) < MIN_FBS_GAMES:
                log(f"  {team}: {len(df)} games < {MIN_FBS_GAMES} — no model (skip rule)")
                continue
            features = feature_columns(df, exclude=exclude_features)
            result = train_team_model(df, features, season, random_state)
            if result is None:
                log(f"  {team}: fewer than {MIN_TRAIN_ROWS} training rows — no model")
                continue
            trained[team] = result
            log(f"  {team}: alpha={result[1]:.4g}, n_train={result[2]}")
    return trained


def setup_season(store: Store, client, season: int,
                 settings: Settings | None = None, log=print) -> None:
    """Backfill skip-team histories through season-1, ensure the season's
    ratings exist, retrain and persist every qualifying model."""
    from cfb_model.data.history import FBS_SINCE, backfill_all_skip_teams

    settings = settings or get_settings()

    log("Backfilling skip-team FBS histories...")
    counts = backfill_all_skip_teams(store, client, season - 1, log)
    log("\nSkip-rule gate report (>=56 games gets a model):")
    for team, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        verdict = "MODEL" if count >= MIN_FBS_GAMES else "still skipped"
        log(f"  {team} (FBS since {FBS_SINCE[team]}): {count} games -> {verdict}")

    log(f"\nTraining models for {season} (unseeded, verbatim p6 recipe)...")
    trained = train_all(store, season, random_state=None, log=log)
    manifest = save_models(trained, season, store, settings)
    log(f"\n{len(trained)} models saved -> {settings.models_dir} (manifest: {manifest.name})")
    log(f"API calls: {client.calls_made}")


def save_models(trained: dict, season: int, store: Store,
                settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    now = datetime.now(timezone.utc).isoformat()
    for team, (pipe, alpha, n_rows) in trained.items():
        name = team.replace(" ", "_") + f"_model_{season}.pkl"
        path = settings.models_dir / name
        joblib.dump(pipe, path)
        manifest[team] = {"path": name, "best_alpha": alpha, "n_train_rows": n_rows}
        with store.conn:
            store.conn.execute(
                "INSERT OR REPLACE INTO model_registry (team_key, season, path, best_alpha,"
                " n_train_rows, sklearn_version, trained_at) VALUES (?,?,?,?,?,?,?)",
                (team, season, str(path), alpha, n_rows, sklearn.__version__, now),
            )
    manifest_path = settings.models_dir / f"manifest_{season}.json"
    manifest_path.write_text(json.dumps(
        {"season": season, "sklearn": sklearn.__version__, "trained_at": now,
         "teams": manifest}, indent=1))
    return manifest_path
