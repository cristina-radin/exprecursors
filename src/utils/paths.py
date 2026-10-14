"""
Centralised path configuration, read from environment variables.

Copy .env.example to .env, fill in the paths and run `source .env`
before any script (or export the variables in your shell / SLURM script).

Required:
    MHW_DATA_FILE        path to merged_daily.nc
    MHW_CLIM_FILE        path to sst_climatology_doy.nc
    MHW_EXPERIMENTS_DIR  root for all experiment outputs

All three are required: if one is missing, importing this module fails
immediately with a clear error (no silent defaults).
"""

import os
from pathlib import Path


def _require(var: str) -> Path:
    val = os.environ.get(var, "")
    if not val:
        raise EnvironmentError(
            f"Environment variable {var} is not set. "
            f"Copy .env.example to .env, fill in the paths and run `source .env`."
        )
    return Path(val)


DATA_FILE = _require("MHW_DATA_FILE")
CLIM_FILE = _require("MHW_CLIM_FILE")
EXPERIMENTS_DIR = _require("MHW_EXPERIMENTS_DIR")
