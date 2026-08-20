"""Thin CFBD REST client (plain requests, Bearer auth).

Deliberately not the generated `cfbd` package: we need raw JSON visibility
and table-driven field mapping to survive API drift (see plan §3).
"""

from __future__ import annotations

import logging
import time

import requests

from cfb_model.config import Settings, get_settings

logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0


class CfbdClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.settings.api_key()}",
                "Accept": "application/json",
            }
        )
        self.calls_made = 0

    def _get(self, path: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        url = f"{self.settings.api_base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._session.get(url, params=params, timeout=60)
                self.calls_made += 1
                if response.status_code in RETRY_STATUSES:
                    logger.warning(
                        "CFBD %s returned %s (attempt %d/%d)",
                        path, response.status_code, attempt, MAX_RETRIES,
                    )
                    time.sleep(BACKOFF_SECONDS * attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:  # includes HTTPError
                last_error = exc
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"CFBD request failed after {MAX_RETRIES} attempts: GET {path}") from last_error

    # --- endpoints (paths verified live in Phase 2) ---

    def team_game_stats(self, year: int, week: int | None = None,
                        classification: str | None = None,
                        season_type: str = "regular", team: str | None = None):
        return self._get(
            "/games/teams",
            year=year, week=week, classification=classification,
            seasonType=season_type, team=team,
        )

    def games(self, year: int, week: int | None = None,
              season_type: str = "regular", team: str | None = None,
              classification: str | None = None):
        return self._get(
            "/games",
            year=year, week=week, seasonType=season_type, team=team,
            classification=classification,
        )

    def lines(self, year: int, week: int | None = None, season_type: str = "regular",
              team: str | None = None):
        return self._get("/lines", year=year, week=week, seasonType=season_type, team=team)

    def sp_ratings(self, year: int | None = None, team: str | None = None):
        return self._get("/ratings/sp", year=year, team=team)

    def talent(self, year: int):
        return self._get("/talent", year=year)

    def calendar(self, year: int):
        return self._get("/calendar", year=year)
