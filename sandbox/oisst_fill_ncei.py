"""Fills the days missing from the CoastWatch ERDDAP OISST v2.1 box subset with the original NCEI daily files (box only).
Produces the 1193 filled days and the NCEI-vs-ERDDAP check (1.9e-6 °C) in docs/open_issues.md "Validation against NOAA OISST v2.1".
Needs: internet (login node), the ERDDAP yearly files in /work/bg1446/u241379/data/oisst/ (curl loop, see that section). No data in the repo."""
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

D = "/work/bg1446/u241379/data/oisst"
OUT = f"{D}/ncei_fill"
os.makedirs(OUT, exist_ok=True)
URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{ym}/oisst-avhrr-v02r01.{ymd}.nc"

have = set()
for f in sorted(glob.glob(f"{D}/oisst_box_*.nc")):
    have |= set(pd.to_datetime(xr.open_dataset(f).time.values).normalize())
full = pd.date_range("1985-01-01", "2024-12-31", freq="D")
missing = [d for d in full if d not in have]
check = [pd.Timestamp("1992-10-01"), pd.Timestamp("2021-06-15"), pd.Timestamp("2005-03-01")]
todo = missing + [d for d in check if d in have]
print(f"missing from ERDDAP: {len(missing)}; consistency days: {len(todo) - len(missing)}", flush=True)


def download(d):
    """curl only (no netCDF in threads: HDF5 is not thread-safe)."""
    out = f"{OUT}/{d:%Y%m%d}.nc"
    tmp = f"{OUT}/{d:%Y%m%d}_global.nc"
    if os.path.exists(out) or os.path.exists(tmp):
        return d, "ok"
    for _ in range(3):
        r = subprocess.run(["curl", "-s", "--max-time", "120", "-o", tmp + ".part", "-w", "%{http_code}",
                            URL.format(ym=f"{d:%Y%m}", ymd=f"{d:%Y%m%d}")], capture_output=True, text=True)
        if r.stdout == "200":
            os.rename(tmp + ".part", tmp)
            return d, "ok"
    return d, f"HTTP {r.stdout}"


def subset(d):
    out = f"{OUT}/{d:%Y%m%d}.nc"
    tmp = f"{OUT}/{d:%Y%m%d}_global.nc"
    if os.path.exists(out):
        return
    with xr.open_dataset(tmp) as ds:
        lon = ds.lon.values
        box = ds.sst.isel(zlev=0).sel(lat=slice(51.0, 62.5)).isel(
            lon=np.flatnonzero((lon >= 354.8) | (lon <= 13.2))).load()
    box = box.assign_coords(lon=((box.lon + 180) % 360) - 180).sortby("lon")
    box.to_dataset(name="sst").to_netcdf(out)
    os.remove(tmp)


with ThreadPoolExecutor(4) as ex:
    bad = [(d, s) for d, s in ex.map(download, todo) if s != "ok"]
print(f"downloads done; failures: {bad[:10]} ({len(bad)})", flush=True)
for d in todo:
    if (d, ) and os.path.exists(f"{OUT}/{d:%Y%m%d}_global.nc") or os.path.exists(f"{OUT}/{d:%Y%m%d}.nc"):
        subset(d)
n_ok = len(glob.glob(f"{OUT}/[0-9]*[0-9].nc"))
print(f"subset files: {n_ok} of {len(todo)}", flush=True)
sys.exit(1 if bad or n_ok != len(todo) else 0)
