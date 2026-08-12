"""
Hobday et al. 2016 MHW classification utilities.

  - to_anom > p90_thresh(DOY)  (90th pct of ref-period anomalies, ±15-day window)
  - min duration: 5 consecutive days
  - gap closure: ≤2 days merged into same event

p90_thresh is stored WITHOUT 31-day smooth in sst_climatology_doy.nc.
Smooth is applied at runtime via uniform_filter1d(size=31, mode='wrap').
"""

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d

CLIM_FILE = "/p/project1/hai_1127/inputs/sst_anomaly/sst_climatology_doy.nc"

_NS_LAT_SLICE = slice(50.0, 63.0)
_NS_LON_SLICE = slice(-5.0, 13.0)

MIN_DURATION = 5
MAX_GAP = 2


def load_ns_p90(smooth_days: int = 31, clim_file: str = CLIM_FILE) -> np.ndarray:
    """Load and return NS-mean P90 threshold, shape (365,)."""
    ds = xr.open_dataset(clim_file)
    p90 = (
        ds.p90_thresh
        .sel(lat=_NS_LAT_SLICE, lon=_NS_LON_SLICE)
        .mean(dim=["lat", "lon"], skipna=True)
        .values
    )
    ds.close()
    return uniform_filter1d(p90, size=smooth_days, mode="wrap")


def apply_hobday(exceedance: np.ndarray,
                 min_dur: int = MIN_DURATION,
                 max_gap: int = MAX_GAP) -> np.ndarray:
    """Apply Hobday gap-closure + minimum-duration criteria to a 1D bool array."""
    mhw = exceedance.copy().astype(bool)
    n = len(mhw)

    # Gap closure
    i = 0
    while i < n:
        if mhw[i]:
            j = i
            while j < n and mhw[j]:
                j += 1
            k = j
            while k < n and not mhw[k]:
                k += 1
            if 0 < (k - j) <= max_gap and k < n:
                mhw[j:k] = True
                i = k
            else:
                i = j
        else:
            i += 1

    # Minimum duration
    i = 0
    while i < n:
        if mhw[i]:
            j = i
            while j < n and mhw[j]:
                j += 1
            if j - i < min_dur:
                mhw[i:j] = False
            i = j
        else:
            i += 1

    return mhw


def mhw_phase_labels(trues: np.ndarray, doys: np.ndarray,
                     p90_ns: np.ndarray) -> np.ndarray:
    """
    Classify each timestep into:  0=no_mhw, 1=onset, 2=mid_event.

    Args:
        trues:   NS-mean to_anom time series
        doys:    day-of-year for each timestep (1–365)
        p90_ns:  smoothed NS-mean P90 threshold, shape (365,)
    """
    thr = p90_ns[doys - 1]
    mhw = apply_hobday(trues > thr)

    onset = np.zeros(len(mhw), dtype=bool)
    for i in range(len(mhw)):
        if mhw[i] and (i == 0 or not mhw[i - 1]):
            onset[i] = True

    labels = np.zeros(len(mhw), dtype=int)
    labels[mhw] = 2
    labels[onset] = 1
    return labels
