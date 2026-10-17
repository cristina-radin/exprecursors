"""
Marine heatwave (MHW) detection after Hobday et al. (2016).

  1. Threshold: for each day of the year, the PERCENTILE-th percentile
     (np.nanpercentile, linear interpolation) of the series itself, pooled
     over the REF_YEARS days with that doy and the WINDOW_HALF_WIDTH days on
     each side. The window is taken by index on the full series, so it crosses
     Dec 31 / Jan 1 and can reach just outside REF_YEARS; only indices outside
     the series are dropped.
  2. The 365-day threshold curve is smoothed with a periodic running mean of
     SMOOTH_WIDTH days.
  3. A day exceeds when the series is strictly above the threshold.
  4. Runs of exceeding days shorter than MIN_DURATION are discarded FIRST;
     the remaining events are THEN joined when separated by at most MAX_GAP
     days, and the gap days count as MHW days.

Calendar: 365-day years without Feb 29 (merged_daily.nc has none), so doy
comes from month and day (noleap_doy), not from dayofyear.

This module also holds load_ns_mean_clim_smooth_delta, which corrects the
regression target for the missing 31-day smooth of mean_clim (open_issues #40).
"""

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d

from src.utils.paths import CLIM_FILE

_NS_LAT_SLICE = slice(50.0, 63.0)
_NS_LON_SLICE = slice(-5.0, 13.0)

# Published definition (Hobday et al. 2016), passed explicitly by every caller.
REF_YEARS = (1985, 2014)
WINDOW_HALF_WIDTH = 5
PERCENTILE = 90
SMOOTH_WIDTH = 31
MIN_DURATION = 5
MAX_GAP = 2

_DAYS_BEFORE_MONTH = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def noleap_doy(months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Day of year (1-365) in a 365-day calendar. Raises on Feb 29."""
    months = np.asarray(months)
    days = np.asarray(days)
    if np.any((months == 2) & (days == 29)):
        raise ValueError(
            "Feb 29 found: the MHW definition here assumes a 365-day calendar."
        )
    return _DAYS_BEFORE_MONTH[months - 1] + days


def periodic_running_mean(curve: np.ndarray, width: int) -> np.ndarray:
    """Centred running mean of `width` (odd) days, wrapping Dec 31 to Jan 1."""
    if width % 2 != 1:
        raise ValueError(f"width must be odd, got {width}")
    return uniform_filter1d(curve, size=width, mode="wrap")


def threshold(
    series: np.ndarray,
    years: np.ndarray,
    doys: np.ndarray,
    ref_years: tuple,
    window_half_width: int,
    percentile: float,
    smooth_width: int,
) -> np.ndarray:
    """Smoothed percentile threshold of `series`, shape (365,), index doy - 1.

    series, years, doys: 1D arrays of the same length, one value per
    consecutive day (doys from noleap_doy).
    """
    n = len(series)
    if not (len(years) == len(doys) == n):
        raise ValueError("series, years and doys must have the same length")
    in_ref = np.flatnonzero((years >= ref_years[0]) & (years <= ref_years[1]))
    present = set(np.unique(years[in_ref]).tolist())
    missing = set(range(ref_years[0], ref_years[1] + 1)) - present
    if missing:
        raise ValueError(f"reference years missing from the series: {sorted(missing)}")

    offsets = np.arange(-window_half_width, window_half_width + 1)
    curve = np.empty(365)
    for d in range(1, 366):
        centres = in_ref[doys[in_ref] == d]
        if len(centres) == 0:
            raise ValueError(f"doy {d} never occurs in the reference years")
        idx = (centres[:, None] + offsets[None, :]).ravel()
        idx = idx[(idx >= 0) & (idx < n)]
        curve[d - 1] = np.nanpercentile(series[idx], percentile, method="linear")
    return periodic_running_mean(curve, smooth_width)


def detect(exceedance: np.ndarray, min_duration: int, max_gap: int) -> np.ndarray:
    """MHW days from a 1D bool array of daily exceedances.

    Runs shorter than min_duration are dropped first; the surviving events
    are then joined across gaps of at most max_gap days (gap days become MHW
    days). E.g. with (5, 2): XXX.XX -> no MHW; XXXXX.XX -> one 5-day event
    (the 2-day run is dropped before joining).
    """
    exc = np.asarray(exceedance, dtype=bool)
    edges = np.diff(np.concatenate([[0], exc.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    keep = (ends - starts) >= min_duration
    starts, ends = starts[keep], ends[keep]

    mhw = np.zeros(len(exc), dtype=bool)
    if len(starts) == 0:
        return mhw
    event_start, event_end = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s - event_end <= max_gap:
            event_end = e
        else:
            mhw[event_start:event_end] = True
            event_start, event_end = s, e
    mhw[event_start:event_end] = True
    return mhw


def load_ns_mean_clim_smooth_delta(
    smooth_days: int = 31, clim_file=CLIM_FILE
) -> np.ndarray:
    """Return delta[doy] = unsmoothed - smoothed NS-mean mean_clim, shape (365,).

    Hobday et al. 2016 prescribes a 31-day smooth for BOTH the threshold and
    the climatological mean, but the mean_clim in sst_climatology_doy.nc (the
    climatology to_anom/target are built from) is only ever the raw CDO
    ydrunmean,11 output (+-5 days), never smoothed. Add this delta to a
    target/anomaly series built from the unsmoothed mean_clim to get what it
    would be under the smoothed one:
        to_anom_smoothed = SST - mean_clim_smoothed
                          = to_anom_unsmoothed + (mean_clim_unsmoothed - mean_clim_smoothed)
                          = to_anom_unsmoothed + delta
    See docs/open_issues.md #40 for the RMS/max magnitude of this correction.
    """
    ds = xr.open_dataset(clim_file)
    mean_clim = (
        ds.mean_clim.sel(lat=_NS_LAT_SLICE, lon=_NS_LON_SLICE)
        .mean(dim=["lat", "lon"], skipna=True)
        .values
    )
    ds.close()
    smoothed = uniform_filter1d(mean_clim, size=smooth_days, mode="wrap")
    return mean_clim - smoothed
