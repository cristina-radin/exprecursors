"""Compares model MHW detections (A, B) with NOAA OISST v2.1 on the same box: r, MHW days, paired events, POD/FAR/F1 by season.
Produces docs/open_issues.md "Validation against NOAA OISST v2.1" (r 0.938, F1 0.71 A / 0.70 B, SON 0.83, MAM 0.61, 1822 vs 2303 days).
Needs: `source .env`, OISST files in /work/bg1446/u241379/data/oisst/ (+ ncei_fill/), abs_vs_anom.py next to it; run from the repo root."""
import contextlib
import glob
import io
import runpy

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label, uniform_filter1d

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("abs_vs_anom.py")))
years, doys, run = g["years"], g["doys"], g["run"]
REF, HW, SW = g["REF_YEARS"], g["WINDOW_HALF_WIDTH"], g["SMOOTH_WIDTH"]
mdates = pd.to_datetime(g["time"].values)
model = {"A": (g["anom"].astype(np.float64), g["m_a"]), "B": (g["sst"], g["m_s"])}
raw_target = g["raw"].astype(np.float64)

# ── OISST box means ──────────────────────────────────────────────────────────
files = sorted(glob.glob("/work/bg1446/u241379/data/oisst/oisst_box_*.nc"))
assert len(files) == 40, len(files)
parts = []
for f in files:
    s = xr.open_dataset(f).sst.squeeze("zlev", drop=True).rename(latitude="lat", longitude="lon").load()
    s = s.assign_coords(time=pd.to_datetime(s.time.values).normalize())
    parts.append(s)
erddap = xr.concat(parts, "time")
erddap = erddap.isel(time=np.unique(erddap.time.values, return_index=True)[1])
erddap = erddap.sel(lat=slice(51.0, 62.5), lon=slice(-5.2, 13.2))  # ERDDAP snapped 62.5 to the 62.625 row
fill_files = sorted(glob.glob("/work/bg1446/u241379/data/oisst/ncei_fill/*.nc"))
fill = xr.concat([xr.open_dataset(f).sst.load() for f in fill_files], "time")
fill = fill.assign_coords(time=pd.to_datetime(fill.time.values).normalize())
assert np.allclose(fill.lat, erddap.lat) and np.allclose(fill.lon, erddap.lon), "grids differ"
fill = fill.assign_coords(lat=erddap.lat, lon=erddap.lon)
common = np.intersect1d(fill.time.values, erddap.time.values)
dmax = float(np.nanmax(np.abs(fill.sel(time=common) - erddap.sel(time=common))))
print(f"ERDDAP days {erddap.sizes['time']}, NCEI fill days {fill.sizes['time']}, "
      f"common {len(common)}: max |NCEI - ERDDAP| = {dmax:.2e} degC")
only_fill = fill.sel(time=np.setdiff1d(fill.time.values, erddap.time.values))
sst = xr.concat([erddap, only_fill], "time").sortby("time")
sst = sst.where(sst > -5)  # land / fill -> NaN
t = pd.to_datetime(sst.time.values)
sst = sst.isel(time=np.flatnonzero(~((t.month == 2) & (t.day == 29))))
sdates = pd.to_datetime(sst.time.values)
assert len(sdates) == 14600 and (sdates == mdates.normalize()).all(), "dates differ from model"
print(f"OISST box: {sst.sizes['lat']} lat x {sst.sizes['lon']} lon, "
      f"lat {float(sst.lat.min()):.3f}-{float(sst.lat.max()):.3f}, "
      f"lon {float(sst.lon.min()):.3f}-{float(sst.lon.max()):.3f}, "
      f"ocean cells {int(sst.isel(time=0).notnull().sum())}")
box_unw = sst.mean(["lat", "lon"], skipna=True).values.astype(np.float64)
w = np.cos(np.deg2rad(sst.lat))
box_cos = sst.weighted(w).mean(["lat", "lon"], skipna=True).values.astype(np.float64)


def clim_mean(x, smooth):
    """Windowed (+-HW by index) doy mean over REF, optionally 31-day periodic smooth."""
    n = len(x)
    in_ref = np.flatnonzero((years >= REF[0]) & (years <= REF[1]))
    off = np.arange(-HW, HW + 1)
    c = np.empty(365)
    for d in range(1, 366):
        idx = (in_ref[doys[in_ref] == d][:, None] + off[None, :]).ravel()
        c[d - 1] = np.nanmean(x[idx[(idx >= 0) & (idx < n)]])
    return uniform_filter1d(c, size=SW, mode="wrap") if smooth else c


r = lambda a, b: np.corrcoef(a, b)[0, 1]
print("\n## Weighting: daily r of the anomaly (unsmoothed clim) vs model target")
for name, box in (("unweighted", box_unw), ("cos(lat)", box_cos)):
    an = box - clim_mean(box, False)[doys - 1]
    print(f"  {name:10s}: r vs raw target {r(an, raw_target):.4f}   r vs target+delta {r(an, model['A'][0]):.4f}")
r_unw = r(box_unw - clim_mean(box_unw, False)[doys - 1], raw_target)
r_cos = r(box_cos - clim_mean(box_cos, False)[doys - 1], raw_target)
box = box_cos if r_cos > r_unw else box_unw
print(f"  -> using {'cos(lat)' if r_cos > r_unw else 'unweighted'}")

sat = {
    "B": box,
    "A (smoothed clim)": box - clim_mean(box, True)[doys - 1],
    "A (unsmoothed clim)": box - clim_mean(box, False)[doys - 1],
}
sat_mhw = {k: run(v)[1] for k, v in sat.items()}

print("\n## MHW days / events")
for k, (x, m) in model.items():
    print(f"  model {k}: {m.sum()} days, {label(m)[1]} events")
for k, m in sat_mhw.items():
    print(f"  OISST {k}: {m.sum()} days, {label(m)[1]} events")

print("\n## Daily r model vs OISST")
print(f"  A: model anomaly vs OISST anomaly (smoothed clim)  {r(model['A'][0], sat['A (smoothed clim)']):.4f}")
print(f"  B: model abs SST vs OISST abs SST                  {r(model['B'][0], sat['B']):.4f}")
print(f"     bias model-OISST abs: mean {np.mean(model['B'][0] - sat['B']):+.3f} degC")


def events(m):
    lab, n = label(m)
    return lab, n


def compare(name, m_mod, m_sat):
    lm, nm = events(m_mod)
    ls, ns = events(m_sat)
    paired_mod = sum(1 for k in range(1, nm + 1) if m_sat[lm == k].any())
    paired_sat = sum(1 for k in range(1, ns + 1) if m_mod[ls == k].any())
    print(f"\n## {name}")
    print(f"  events: model {nm} (paired {paired_mod}, unpaired {nm - paired_mod}); "
          f"OISST {ns} (paired {paired_sat}, unpaired {ns - paired_sat})")
    seasons = {"DEF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
    print(f"  {'season':6s} {'hits':>5s} {'FA':>5s} {'miss':>5s} {'POD':>6s} {'FAR':>6s} {'F1':>6s}")
    for s, months in [("all", list(range(1, 13)))] + list(seasons.items()):
        sel = np.isin(mdates.month, months)
        h = int((m_mod & m_sat & sel).sum()); fa = int((m_mod & ~m_sat & sel).sum()); mi = int((~m_mod & m_sat & sel).sum())
        pod = h / (h + mi) if h + mi else np.nan
        far = fa / (h + fa) if h + fa else np.nan
        f1 = 2 * h / (2 * h + fa + mi) if (2 * h + fa + mi) else np.nan
        print(f"  {s:6s} {h:5d} {fa:5d} {mi:5d} {pod:6.2f} {far:6.2f} {f1:6.2f}")


compare("A: model A vs OISST A (smoothed clim)", model["A"][1], sat_mhw["A (smoothed clim)"])
compare("A': model A vs OISST A (unsmoothed clim)", model["A"][1], sat_mhw["A (unsmoothed clim)"])
compare("B: model B vs OISST B", model["B"][1], sat_mhw["B"])
compare("cross: model A vs OISST B", model["A"][1], sat_mhw["B"])
compare("cross: model B vs OISST A (smoothed clim)", model["B"][1], sat_mhw["A (smoothed clim)"])
