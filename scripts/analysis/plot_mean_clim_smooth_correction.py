#!/usr/bin/env python
"""
plot_mean_clim_smooth_correction.py — verification figure for known_issues.md
#40 (mean_clim.nc missing the 31-day Hobday smooth).

Panel 1: NS-box mean_clim, unsmoothed vs. 31-day-smoothed, by day-of-year.
Panel 2: delta[doy] = unsmoothed - smoothed, with +-RMS/+-max marked.
Panel 3: target, original vs. hobday_smooth_target-corrected, over one real
         year with known MHW activity (2014, see known_issues.md #40/#41).

CPU only, no GPU needed. Reuses src/utils/hobday.py — does not
reimplement the NS-box slicing/smoothing logic.

Usage:
  python scripts/analysis/plot_mean_clim_smooth_correction.py
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

from src.utils.hobday import (  # noqa: E402
    _NS_LAT_SLICE,
    _NS_LON_SLICE,
    load_ns_mean_clim_smooth_delta,
)
from src.utils.paths import CLIM_FILE, DATA_FILE  # noqa: E402


def main():
    ds_clim = xr.open_dataset(CLIM_FILE)
    mean_clim_ns = (
        ds_clim.mean_clim.sel(lat=_NS_LAT_SLICE, lon=_NS_LON_SLICE)
        .mean(dim=["lat", "lon"], skipna=True)
        .values
    )
    ds_clim.close()
    smoothed = uniform_filter1d(mean_clim_ns, size=31, mode="wrap")
    delta = load_ns_mean_clim_smooth_delta()
    assert np.allclose(delta, mean_clim_ns - smoothed), "delta mismatch vs hobday.py"

    rms = float(np.sqrt((delta**2).mean()))
    dmax = float(np.abs(delta).max())
    print(f"delta: RMS={rms:.4f} degC  max|delta|={dmax:.4f} degC")

    ds_daily = xr.open_dataset(DATA_FILE)
    target = ds_daily["target"]
    doys = ds_daily.time.dt.dayofyear.values
    doys_clamped = doys.copy()
    doys_clamped[doys_clamped >= 365] = 365
    target_corrected = target.values + delta[doys_clamped - 1]

    years = ds_daily.time.dt.year.values
    m2014 = years == 2014

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    doy_axis = np.arange(1, 366)
    axes[0].plot(doy_axis, mean_clim_ns, label="mean_clim (unsmoothed)", alpha=0.8)
    axes[0].plot(doy_axis, smoothed, label="mean_clim (31-day smoothed)", lw=2)
    axes[0].set_xlabel("Day of year")
    axes[0].set_ylabel("NS-box mean_clim (degC)")
    axes[0].set_title("Panel 1: mean_clim unsmoothed vs. 31-day smoothed")
    axes[0].legend()

    axes[1].plot(doy_axis, delta, color="tab:red")
    axes[1].axhline(rms, ls="--", color="gray", label=f"+RMS={rms:.4f}")
    axes[1].axhline(-rms, ls="--", color="gray")
    axes[1].axhline(dmax, ls=":", color="black", label=f"+max={dmax:.4f}")
    axes[1].axhline(-dmax, ls=":", color="black")
    axes[1].set_xlabel("Day of year")
    axes[1].set_ylabel("delta[doy] (degC)")
    axes[1].set_title("Panel 2: delta = unsmoothed - smoothed")
    axes[1].legend()

    axes[2].plot(
        ds_daily.time.values[m2014],
        target.values[m2014],
        label="target (original)",
        alpha=0.8,
    )
    axes[2].plot(
        ds_daily.time.values[m2014],
        target_corrected[m2014],
        label="target (hobday_smooth_target corrected)",
        alpha=0.8,
    )
    axes[2].set_xlabel("2014")
    axes[2].set_ylabel("target (degC)")
    axes[2].set_title("Panel 3: target original vs. corrected, 2014 (known MHW year)")
    axes[2].legend()

    plt.tight_layout()
    out_dir = REPO_ROOT / "experiments" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mean_clim_smooth_correction.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
