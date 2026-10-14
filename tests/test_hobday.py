"""
MHW detection (src/utils/hobday.py), Hobday et al. 2016 as implemented by
Oliver's marineHeatWaves.detect:
  * detect: minimum duration first, then gap joining (gap days count)
  * periodic_running_mean wraps Dec 31 / Jan 1
  * threshold: hand-checked window by index, ~10 % exceedances on a
    synthetic series
  * noleap_doy and the input checks
"""

import numpy as np
import pytest

from src.utils.hobday import (
    MAX_GAP,
    MIN_DURATION,
    PERCENTILE,
    REF_YEARS,
    SMOOTH_WIDTH,
    WINDOW_HALF_WIDTH,
    detect,
    noleap_doy,
    periodic_running_mean,
    threshold,
)


def _bools(pattern):
    return np.array([c == "X" for c in pattern])


def _calendar(first_year, last_year):
    """years and noleap doys of consecutive 365-day years."""
    n_years = last_year - first_year + 1
    years = np.repeat(np.arange(first_year, last_year + 1), 365)
    doys = np.tile(np.arange(1, 366), n_years)
    return years, doys


# ── detect ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("XXX.XX", "......"),  # both runs < 5 days: gaps are not filled first
        ("XXXXX.XX", "XXXXX..."),  # 2-day run dropped before joining: 5-day event
        ("XXXXX.XXXXX", "XXXXXXXXXXX"),  # both runs >= 5: one 11-day event
        ("XXXXX..XXXXX", "XXXXXXXXXXXX"),  # gap of 2 joined, gap days count
        ("XXXXX...XXXXX", "XXXXX...XXXXX"),  # gap of 3 not joined
        ("XXXXX.XXXXX.XXXXX", "XXXXXXXXXXXXXXXXX"),  # chained joins
        ("..XXXXX", "..XXXXX"),  # event touching the end of the series
        ("XXXX", "...."),
    ],
)
def test_detect(pattern, expected):
    out = detect(_bools(pattern), min_duration=MIN_DURATION, max_gap=MAX_GAP)
    assert out.tolist() == _bools(expected).tolist()


# ── periodic_running_mean ─────────────────────────────────────────────────────


def test_periodic_running_mean_wraps_year_end():
    curve = np.zeros(365)
    curve[0] = 31.0
    out = periodic_running_mean(curve, 31)
    ones = list(range(0, 16)) + list(range(350, 365))  # doy 1 +- 15, wrapped
    assert np.allclose(out[ones], 1.0)
    assert np.allclose(np.delete(out, ones), 0.0)


def test_periodic_running_mean_rejects_even_width():
    with pytest.raises(ValueError, match="odd"):
        periodic_running_mean(np.zeros(365), 30)


# ── threshold ─────────────────────────────────────────────────────────────────


def test_threshold_by_hand_percentile_of_three_years():
    # Value = year index (0, 1, 2), no window, no smoothing:
    # p90 of [0, 1, 2] with linear interpolation is 1.8 for every doy.
    years, doys = _calendar(2000, 2002)
    series = (years - 2000).astype(float)
    thr = threshold(series, years, doys, (2000, 2002), 0, 90, 1)
    assert np.allclose(thr, 1.8)


def test_threshold_window_is_by_index_and_crosses_year_end():
    # series[i] = i, window +-1, no smoothing. The max for Dec 31 of the last
    # year has no Jan 1 after it (dropped); the min for Jan 1 of the first
    # year has no Dec 31 before it.
    years, doys = _calendar(2000, 2002)
    series = np.arange(len(years), dtype=float)
    thr_max = threshold(series, years, doys, (2000, 2002), 1, 100, 1)
    thr_min = threshold(series, years, doys, (2000, 2002), 1, 0, 1)
    assert thr_max[364] == len(years) - 1
    assert thr_min[0] == 0.0
    # With a 2001-only reference period, the Dec 31 window reaches Jan 1 2002
    # (index 730), outside the reference years but inside the series.
    thr_ref = threshold(series, years, doys, (2001, 2001), 1, 100, 1)
    assert thr_ref[364] == 730.0


def test_threshold_gives_about_10_percent_exceedances():
    years, doys = _calendar(REF_YEARS[0], REF_YEARS[1])
    rng = np.random.default_rng(0)
    series = np.sin(2 * np.pi * doys / 365) + rng.normal(size=len(years))
    thr = threshold(
        series,
        years,
        doys,
        ref_years=REF_YEARS,
        window_half_width=WINDOW_HALF_WIDTH,
        percentile=PERCENTILE,
        smooth_width=SMOOTH_WIDTH,
    )
    frac = (series > thr[doys - 1]).mean()
    assert 0.09 < frac < 0.11


def test_threshold_raises_on_missing_reference_years():
    years, doys = _calendar(2000, 2002)
    with pytest.raises(ValueError, match="reference years missing"):
        threshold(np.zeros(len(years)), years, doys, (1999, 2002), 0, 90, 1)


# ── noleap_doy ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "month, day, doy", [(1, 1, 1), (2, 28, 59), (3, 1, 60), (12, 31, 365)]
)
def test_noleap_doy(month, day, doy):
    assert noleap_doy(np.array([month]), np.array([day]))[0] == doy


def test_noleap_doy_raises_on_feb_29():
    with pytest.raises(ValueError, match="Feb 29"):
        noleap_doy(np.array([2]), np.array([29]))
