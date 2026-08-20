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

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.hobday import apply_hobday  # noqa: E402
from src.utils.paths import CLIM_FILE, DATA_FILE  # noqa: E402

NS_LAT_SLICE = slice(51.0, 62.5)
NS_LON_SLICE = slice(-5.2, 13.2)


def main():
    ds = xr.open_dataset(DATA_FILE)
    clim = xr.open_dataset(CLIM_FILE)

    ns_to_anom = ds.to_anom.sel(lat=NS_LAT_SLICE, lon=NS_LON_SLICE)
    ns_land = ds.land_mask.sel(lat=NS_LAT_SLICE, lon=NS_LON_SLICE)
    ocean_mask = ns_land.values == 1
    n_ocean = int(ocean_mask.sum())
    print(f"NS box ocean pixels: {n_ocean}")

    p90_native = clim.p90_thresh
    p90_regrid = p90_native.interp(
        lat=ns_to_anom.lat, lon=ns_to_anom.lon, method="linear"
    )

    years = ds.time.dt.year.values
    doys = ds.time.dt.dayofyear.values
    doys_clamped = doys.copy()
    doys_clamped[doys_clamped >= 365] = 365

    to_anom_vals = ns_to_anom.values
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
        REPO_ROOT / "experiments" / "figures" / "mhw_area_threshold_calibration.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")
    print("\nNo threshold has been chosen — this is a conceptual decision, see plan.")


if __name__ == "__main__":
    main()
