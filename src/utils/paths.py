"""
Centralised path configuration — reads from environment variables.

Copy .env.example to .env and set the variables before running any script.
Or export them in your shell / SLURM submit script.

Required:
    MHW_DATA_FILE   — path to merged_daily.nc
    MHW_CLIM_FILE   — path to sst_climatology_doy.nc

Optional (default: ./experiments):
    MHW_EXPERIMENTS_DIR — root for all experiment outputs
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def _require(var: str) -> Path:
    val = os.environ.get(var, "")
    if not val:
        raise EnvironmentError(
            f"Environment variable {var} is not set. "
            f"Copy .env.example to .env and fill in the paths."
        )
    return Path(val)


def _optional(var: str, default: Path) -> Path:
    val = os.environ.get(var, "")
    return Path(val) if val else default


DATA_FILE = _require("MHW_DATA_FILE")
CLIM_FILE = _require("MHW_CLIM_FILE")
EXPERIMENTS_DIR = _optional("MHW_EXPERIMENTS_DIR", REPO_ROOT / "experiments")
