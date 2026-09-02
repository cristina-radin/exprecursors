#!/usr/bin/env python
"""
calibrate_mhw_area_threshold.py — Step 2 of the mean_clim/kfold/ground-truth
plan (known_issues.md #41): quantify the per-pixel-Hobday "area in MHW"
fraction for every day in the NS box, using data already on Raven (no new
transfers), and report its distribution so a threshold can be CHOSEN
(a conceptual decision, not made here) rather than assumed.

Does NOT pick a final cutoff — prints candidate percentiles/day-count
tables for several reasonable choices and saves the calibration figure, so
the actual number can be agreed on before it's used anywhere downstream
(recall recomputation, confusion matrix vs. the basin-mean definition).

CPU only. Reuses src/utils/hobday.py's apply_hobday() — does not
reimplement the Hobday persistence/gap-closure logic.

Usage:
  python scripts/analysis/calibrate_mhw_area_threshold.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.hobday import apply_hobday  # noqa: E402
from src.utils.paths import CLIM_FILE, DATA_FILE  # noqa: E402

NS_LAT_SLICE = slice(51.0, 62.5)
NS_LON_SLICE = slice(-5.2, 13.2)
SMOOTH_DAYS = 31


def main():
    ds = xr.open_dataset(DATA_FILE)
    clim = xr.open_dataset(CLIM_FILE)

    ns_to_anom = ds.to_anom.sel(lat=NS_LAT_SLICE, lon=NS_LON_SLICE)
    ns_land = ds.land_mask.sel(lat=NS_LAT_SLICE, lon=NS_LON_SLICE)
    ocean_mask = ns_land.values == 1
    n_ocean = int(ocean_mask.sum())
    print(f"NS box ocean pixels: {n_ocean}")

    # Bug found Aug 21 2026 (user's methodological review): this script
    # originally used clim.p90_thresh / ds.to_anom RAW, i.e. WITHOUT the
    # 31-day Hobday et al. 2016 smooth -- but the v2 models train with
    # hobday_smooth_target=True, whose target/prediction space and
    # load_ns_p90() threshold ARE smoothed (see src/utils/hobday.py).
    # quantile_head_recall_v2_all5.py's def2 (area_frac-based) then mixed
    # this unsmoothed area_frac with the smoothed thresh1 for def1/recall
    # in the same table -- two different reference climatologies. Fixed
    # here by smoothing BOTH p90_thresh and mean_clim per-pixel along the
    # doy axis (uniform_filter1d, same convention as load_ns_p90 /
    # load_ns_mean_clim_smooth_delta, just applied per grid cell instead
    # of pre-averaged over the NS box, since this script needs a spatial
    # field, not a box-mean scalar).
    p90_native = uniform_filter1d(
        clim.p90_thresh.values, size=SMOOTH_DAYS, axis=0, mode="wrap"
    )
    mean_clim_raw = clim.mean_clim.values
    mean_clim_smooth = uniform_filter1d(
        mean_clim_raw, size=SMOOTH_DAYS, axis=0, mode="wrap"
    )
    mean_clim_delta = (
        mean_clim_raw - mean_clim_smooth
    )  # to_anom correction, per pixel/doy

    p90_native_da = xr.DataArray(
        p90_native, dims=clim.p90_thresh.dims, coords=clim.p90_thresh.coords
    )
    delta_da = xr.DataArray(
        mean_clim_delta, dims=clim.mean_clim.dims, coords=clim.mean_clim.coords
    )
    p90_regrid = p90_native_da.interp(
        lat=ns_to_anom.lat, lon=ns_to_anom.lon, method="linear"
    )
    delta_regrid = delta_da.interp(
        lat=ns_to_anom.lat, lon=ns_to_anom.lon, method="linear"
    )

    years = ds.time.dt.year.values
    doys = ds.time.dt.dayofyear.values
    doys_clamped = doys.copy()
    doys_clamped[doys_clamped >= 365] = 365

    # to_anom_smoothed = to_anom_unsmoothed + (mean_clim_unsmoothed -
    # mean_clim_smoothed), same identity load_ns_mean_clim_smooth_delta's
    # docstring derives, applied per-pixel here.
    to_anom_vals = ns_to_anom.values + delta_regrid.values[doys_clamped - 1]
    p90_vals = p90_regrid.values
    thresh_per_time = p90_vals[doys_clamped - 1]
    exceed = to_anom_vals > thresh_per_time

    nlat, nlon = to_anom_vals.shape[1], to_anom_vals.shape[2]
    mhw_per_pixel = np.zeros_like(to_anom_vals, dtype=bool)
    for i in range(nlat):
        for j in range(nlon):
            if ocean_mask[i, j]:
                mhw_per_pixel[:, i, j] = apply_hobday(exceed[:, i, j])

    area_frac = mhw_per_pixel[:, ocean_mask].mean(axis=1)  # (time,)
    np.save(
        REPO_ROOT / "experiments" / "figures" / "area_frac_timeseries.npy", area_frac
    )

    print("\narea_frac(t) distribution over all 14600 days:")
    for p in [50, 75, 90, 95, 97.5, 99]:
        v = np.percentile(area_frac, p)
        n_days = int((area_frac >= v).sum())
        print(
            f"  p{p:5.1f} = {v:.4f}  (>= this value on {n_days} days, {n_days/len(area_frac)*100:.1f}%)"
        )

    print("\nCandidate thresholds and resulting day counts for known years:")
    candidates = {
        "top1% (p99)": np.percentile(area_frac, 99),
        "top5% (p95)": np.percentile(area_frac, 95),
        "top10% (p90)": np.percentile(area_frac, 90),
        "top20% (p80)": np.percentile(area_frac, 80),
        "50% area (exploratory, no lit. basis)": 0.5,
    }
    for label, thr in candidates.items():
        regional_mhw_day = area_frac >= thr
        print(f"\n  {label} (threshold={thr:.4f}):")
        for y in [2022, 2014, 2007]:
            m = years == y
            print(f"    {y}: {int(regional_mhw_day[m].sum())} days")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(area_frac, bins=100, color="tab:blue", alpha=0.7)
    for label, thr in candidates.items():
        if "50%" in label:
            continue
        ax.axvline(thr, ls="--", label=f"{label} = {thr:.3f}")
    ax.set_xlabel("area_frac(t) — fraction of NS-box ocean pixels in MHW")
    ax.set_ylabel("count (days, 1985-2024)")
    ax.set_title(
        "Distribution of daily NS-box MHW area fraction — candidate thresholds"
    )
    ax.legend()
    plt.tight_layout()
    out_path = (
        REPO_ROOT
        / "experiments"
        / "figures"
        / "step2_mhw_definition"
        / "mhw_area_threshold_calibration.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    print("\nNo threshold has been chosen — this is a conceptual decision, see plan.")


if __name__ == "__main__":
    main()
