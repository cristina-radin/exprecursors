"""
Tiny synthetic stand-ins for merged_daily.nc and sst_climatology_doy.nc.

Importing this module points MHW_DATA_FILE / MHW_CLIM_FILE at the synthetic
files (src.utils.paths reads them at import time), so tests exercise the real
LazyDataset / LazyDataModule code without any real data.

Grid: 6 x 8 pixels. Two land masks that differ on purpose:
  land_mask          (SST/atmosphere): land = column 0 and pixel (2, 3)
  land_mask_tbottom  (ptho_bot only) : land = column 0 and pixel (4, 5)
ptho_bot holds a huge value (50.0) on its own land pixels, so any statistic
that leaks land pixels is obviously wrong.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

TMP = Path(tempfile.mkdtemp(prefix="mhw_synthetic_"))
DATA_FILE = TMP / "merged_daily.nc"
DATA_FILE_NO_TBOTTOM = TMP / "merged_daily_no_tbottom.nc"
CLIM_FILE = TMP / "sst_climatology_doy.nc"

os.environ["MHW_DATA_FILE"] = str(DATA_FILE)
os.environ["MHW_CLIM_FILE"] = str(CLIM_FILE)

NLAT, NLON = 6, 8
YEARS = range(2000, 2012)
WINDOW, LEAD = 5, 2
VARIABLES = ["ptho_bot", "u10", "v10", "msl", "ssr"]
LAND_MASK_LAND_PIXELS = [(r, 0) for r in range(NLAT)] + [(2, 3)]
TBOTTOM_LAND_PIXELS = [(r, 0) for r in range(NLAT)] + [(4, 5)]
PTHO_LAND_VALUE = 50.0


def _mask(land_pixels):
    m = np.ones((NLAT, NLON), dtype=np.float32)  # 1 = ocean, 0 = land
    for r, c in land_pixels:
        m[r, c] = 0.0
    return m


def _build_data_file(path: Path, with_tbottom: bool) -> None:
    rng = np.random.default_rng(0)
    time = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    n = len(time)
    shape = (n, NLAT, NLON)
    data = {v: rng.normal(size=shape).astype(np.float32) for v in VARIABLES}
    data["to_anom"] = rng.normal(size=shape).astype(np.float32)
    for r, c in TBOTTOM_LAND_PIXELS:
        data["ptho_bot"][:, r, c] = PTHO_LAND_VALUE

    # Persistent AR(1) target with a seasonal cycle, so Hobday finds real
    # events (>= 5 consecutive days) with different counts per year.
    doy = time.dayofyear.values
    noise = rng.normal(scale=0.35, size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.93 * ar[i - 1] + noise[i]
    target = (0.5 * np.sin(2 * np.pi * doy / 365) + ar).astype(np.float32)

    ds = xr.Dataset(
        {k: (("time", "lat", "lon"), v) for k, v in data.items()}
        | {
            "land_mask": (("lat", "lon"), _mask(LAND_MASK_LAND_PIXELS)),
            "target": (("time",), target),
        },
        coords={
            "time": time,
            "lat": np.arange(NLAT, dtype=float),
            "lon": np.arange(NLON, dtype=float),
        },
    )
    if with_tbottom:
        ds["land_mask_tbottom"] = (("lat", "lon"), _mask(TBOTTOM_LAND_PIXELS))
    ds.to_netcdf(path)


def _build_clim_file(path: Path) -> None:
    doys = np.arange(1, 366)
    lat = np.arange(50.0, 64.0)
    lon = np.arange(-5.0, 14.0)
    seasonal = np.sin(2 * np.pi * doys / 365)
    mean_clim = 1.5 * seasonal[:, None, None] + np.zeros((365, lat.size, lon.size))
    p90 = 0.4 + 0.3 * seasonal[:, None, None] + np.zeros((365, lat.size, lon.size))
    xr.Dataset(
        {
            "mean_clim": (("doy", "lat", "lon"), mean_clim.astype(np.float32)),
            "p90_thresh": (("doy", "lat", "lon"), p90.astype(np.float32)),
        },
        coords={"doy": doys, "lat": lat, "lon": lon},
    ).to_netcdf(path)


_build_data_file(DATA_FILE, with_tbottom=True)
_build_data_file(DATA_FILE_NO_TBOTTOM, with_tbottom=False)
_build_clim_file(CLIM_FILE)


def write_config(directory: Path, name: str = "cfg.yaml", **overrides) -> str:
    """Write a minimal training-style yaml (defaults match the synthetic grid)."""
    cfg = dict(
        variables=VARIABLES,
        ocean_variables=["ptho_bot"],
        window_size=WINDOW,
        lead_time=LEAD,
        normalize=True,
        batch_size=4,
        num_workers=0,
        seed=42,
        split_mode="stratified_kfold",
        n_folds=5,
        fold=0,
        land_fill_mode="zero",
        hobday_smooth_target=False,
    )
    cfg.update(overrides)
    path = Path(directory) / name
    path.write_text(yaml.safe_dump(cfg))
    return str(path)
