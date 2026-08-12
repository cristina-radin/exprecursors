"""
MHW statistics using Hobday et al. 2016 definition:
  - to_anom > p90_thresh(DOY)  — 90th pct of reference-period anomalies, ±5-day window
  - Duration >= 5 consecutive days
  - Gap closure: events separated by <= 2 days are merged

Produces:
  1. mhw_monthly_timeseries.png   — monthly % of domain in MHW conditions (1985-2024)
  2. mhw_spatial_frequency.png    — spatial map of mean MHW days per year
  3. mhw_monthly_climatology.png  — seasonal cycle of MHW frequency
"""

import matplotlib
import numpy as np
import xarray as xr

from src.utils.paths import (
    CLIM_FILE as CLIM_FILE_ENV,
)
from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pandas import Series
from scipy.ndimage import uniform_filter1d

DATA_FILE = DATA_FILE_ENV
CLIM_FILE = CLIM_FILE_ENV
OUT_DIR = Path(".")
MIN_DURATION = 5
GAP_CLOSURE = 2


def apply_hobday_1d(x):
    """Hobday criterion on a 1D binary array (numpy, no Python loop over T)."""
    x = x.astype(np.int8).copy()

    # --- Gap closure ---
    ones_idx = np.where(x == 1)[0]
    if len(ones_idx) >= 2:
        diffs = np.diff(ones_idx)
        for i in np.where((diffs > 1) & (diffs <= GAP_CLOSURE + 1))[0]:
            x[ones_idx[i] + 1 : ones_idx[i + 1]] = 1

    # --- Minimum duration via run-length encoding ---
    result = np.zeros_like(x)
    d = np.diff(np.concatenate([[0], x, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= MIN_DURATION:
            result[s:e] = 1
    return result


# -------------------------------------------------------------------------
print("Loading data...")
ds = xr.open_dataset(DATA_FILE)
to_anom = ds["to_anom"].values  # (T, nlat, nlon)
land_mask = ds["land_mask"].values
time = ds["time"].values
years = ds["time"].dt.year.values
months = ds["time"].dt.month.values
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

T, nlat, nlon = to_anom.shape
doys = ds["time"].dt.dayofyear.values
print(f"  Domain: {nlat}×{nlon}, {T} days  ({years[0]}–{years[-1]})")

ta_std = to_anom.std(axis=0)
ocean = (land_mask == 1) & (ta_std > 0.001)
print(f"  Ocean pixels: {ocean.sum()}")

# Load P90 threshold, interpolate to 0.5° grid, apply 31-day circular smooth
print("Loading P90 threshold...")
ds_clim = xr.open_dataset(CLIM_FILE)
p90_regrid = (
    ds_clim["p90_thresh"]
    .interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": "extrapolate"})
    .values
)  # (365, nlat, nlon)
ds_clim.close()
p90_smooth = uniform_filter1d(p90_regrid, size=31, axis=0, mode="wrap")

# Per-DOY threshold comparison
print("Building binary MHW array...")
binary = np.zeros((T, nlat, nlon), dtype=np.int8)
for doy in range(1, 366):
    idx = np.where(doys == doy)[0]
    if len(idx):
        binary[idx] = (to_anom[idx] > p90_smooth[doy - 1]).astype(np.int8)
binary[:, ~ocean] = 0

# -------------------------------------------------------------------------
# Apply Hobday pixel by pixel (inner ops are numpy, not Python loops over T)
# -------------------------------------------------------------------------
print("Applying Hobday criterion...")
hobday = np.zeros_like(binary)
ocean_idx = np.argwhere(ocean)  # (n_ocean, 2)
n = len(ocean_idx)
for k, (j, i) in enumerate(ocean_idx):
    hobday[:, j, i] = apply_hobday_1d(binary[:, j, i])
    if (k + 1) % 2000 == 0:
        print(f"  {k+1}/{n} pixels", end="\r")
print("\n  Done.")

# -------------------------------------------------------------------------
# 1. Monthly time series: % of ocean area in MHW conditions
# -------------------------------------------------------------------------
unique_months = sorted(set(zip(years.tolist(), months.tolist())))
dates_m, frac_m = [], []
for yr, mo in unique_months:
    mask_t = (years == yr) & (months == mo)
    frac_pixel = hobday[mask_t].mean(axis=0)  # mean over days this month
    frac_domain = frac_pixel[ocean].mean() * 100  # % of ocean pixels
    dates_m.append(np.datetime64(f"{yr:04d}-{mo:02d}-01"))
    frac_m.append(float(frac_domain))

dates_m = np.array(dates_m)
frac_m = np.array(frac_m)
smooth = Series(frac_m).rolling(12, center=True, min_periods=6).mean().values

fig, ax = plt.subplots(figsize=(16, 4))
ax.fill_between(dates_m, frac_m, alpha=0.35, color="tomato")
ax.plot(dates_m, frac_m, color="tomato", lw=0.7, alpha=0.8)
ax.plot(dates_m, smooth, color="darkred", lw=2.0, label="12-month rolling mean")
ax.set_ylabel("% ocean area in MHW (%)", fontsize=10)
ax.set_title(
    "Monthly fraction of North Atlantic in Marine Heatwave conditions\n"
    f"Hobday 2016: to_anom > p90_thresh(DOY), ≥{MIN_DURATION} consecutive days, "
    f"gap closure ≤{GAP_CLOSURE} d",
    fontsize=10,
)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT_DIR / "mhw_monthly_timeseries.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: mhw_monthly_timeseries.png")

# -------------------------------------------------------------------------
# 2. Spatial map: mean MHW days per year
# -------------------------------------------------------------------------
n_years = len(set(years.tolist()))
mhw_total = hobday.sum(axis=0).astype(np.float32)
mhw_total[~ocean] = np.nan
mhw_per_yr = mhw_total / n_years

extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(
    mhw_per_yr, origin="lower", extent=extent, cmap="YlOrRd", aspect="auto", vmin=0
)
plt.colorbar(im, ax=ax, label="MHW days per year", fraction=0.03)
ax.set_title(
    f"Mean Marine Heatwave days per year ({years[0]}–{years[-1]})\n"
    "Hobday 2016 definition",
    fontsize=11,
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
fig.savefig(OUT_DIR / "mhw_spatial_frequency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: mhw_spatial_frequency.png")

# -------------------------------------------------------------------------
# 3. Seasonal cycle: mean % MHW days per calendar month
# -------------------------------------------------------------------------
clim_mo = np.zeros(12)
for mo in range(1, 13):
    vals = []
    for yr in set(years.tolist()):
        mask_t = (years == yr) & (months == mo)
        if mask_t.sum() == 0:
            continue
        vals.append(float(hobday[mask_t][:, ocean].mean()) * 100)
    clim_mo[mo - 1] = np.mean(vals)

month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(month_names, clim_mo, color="tomato", alpha=0.8)
ax.set_ylabel("% ocean area in MHW (%)")
ax.set_title(
    f"Seasonal cycle of MHW frequency — North Atlantic\n"
    f"Mean {years[0]}–{years[-1]} (Hobday 2016)",
    fontsize=10,
)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(OUT_DIR / "mhw_monthly_climatology.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: mhw_monthly_climatology.png")

print("\nAll done.")
