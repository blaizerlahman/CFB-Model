"""Central configuration: paths and API-key resolution.

Every path in the package comes from here — no absolute paths anywhere else
(replaces the notebooks' mix of Desktop/Development hardcoded roots).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_KEY_ENV_VAR = "CFBD_API_KEY"
API_KEY_FILE = PROJECT_ROOT / "Repository" / "Personal Access" / "cfb_api.txt"


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT

    # New consolidated stores.
    db_path: Path = PROJECT_ROOT / "data" / "cfb.db"
    models_dir: Path = PROJECT_ROOT / "models"
    output_root: Path = PROJECT_ROOT / "output"
    logs_dir: Path = PROJECT_ROOT / "logs"

    # Legacy read-only sources (migration + parity oracles). Never written to.
    legacy_season_dir: Path = PROJECT_ROOT / "2024 Season"
    legacy_updated_data: Path = PROJECT_ROOT / "Updated data"
    legacy_fcs_data: Path = PROJECT_ROOT / "FCS Data"
    legacy_team_models: Path = PROJECT_ROOT / "Team Models"
    legacy_bin_csv: Path = PROJECT_ROOT / "Bin Data" / "Bin_Data.csv"
    legacy_sp_dir: Path = PROJECT_ROOT / "SP+ Data"
    legacy_talent_dir: Path = PROJECT_ROOT / "Talent Data"

    # Calibration used when reporting/grading. 'legacy' is the original
    # Bin_Data.csv, whose rates were computed from replays that leaked
    # end-of-season SP+ and run as high as 100%.
    bin_set: str = "leakfree"

    # Ask both teams' models for each game and average them. Measurably better
    # at predicting the game itself (pooled MAE 14.06 -> 13.36); against the
    # spread the two are indistinguishable.
    average_sides: bool = True

    api_base_url: str = "https://api.collegefootballdata.com"
    _api_key: str | None = field(default=None, repr=False)

    def api_key(self) -> str:
        """Resolve the CFBD API key: env var -> .env -> key file.

        The key is never logged; callers must not print it.
        """
        if self._api_key:
            return self._api_key
        key = os.environ.get(API_KEY_ENV_VAR)
        if not key:
            try:
                from dotenv import load_dotenv

                load_dotenv(self.project_root / ".env")
                key = os.environ.get(API_KEY_ENV_VAR)
            except ImportError:
                pass
        if not key and API_KEY_FILE.exists():
            key = API_KEY_FILE.read_text().strip()
        if not key:
            raise RuntimeError(
                "No CFBD API key found. Provide one via the CFBD_API_KEY env var, "
                f"a .env file at {self.project_root / '.env'}, or {API_KEY_FILE}."
            )
        self._api_key = key
        return key

    def ensure_dirs(self) -> None:
        for d in (self.db_path.parent, self.models_dir, self.output_root, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
