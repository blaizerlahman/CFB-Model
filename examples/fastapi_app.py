"""Proof of the service import path (plan Phase 8): a minimal FastAPI app
exposing on-demand matchup predictions for e.g. a personal website.

Run:  pip install fastapi uvicorn
      uvicorn examples.fastapi_app:app --reload

The prediction core is imported directly — no CLI coupling.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from cfb_model.config import get_settings
from cfb_model.data.store import Store
from cfb_model.model.predict import predict_matchup
from cfb_model.pipeline import load_models

app = FastAPI(title="CFB Model")


@lru_cache(maxsize=1)
def _resources():
    settings = get_settings()
    store = Store(settings.db_path)
    import glob
    import re

    seasons = [
        int(m.group(1))
        for p in glob.glob(str(settings.models_dir / "*_model_*.pkl"))
        if (m := re.search(r"_model_(\d{4})\.pkl$", p))
    ]
    season = max(seasons) if seasons else 2024
    return store, load_models(settings, season), season


@app.get("/matchup")
def matchup(team1: str, team2: str, spread: float | None = None):
    store, models, season = _resources()
    try:
        result = predict_matchup(store, models, team1, team2, spread=spread)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    result["model_season"] = season
    return result
