"""MHW detection on the box-mean SST anomaly (A) vs reconstructed absolute SST (B), 1985-2024.
Produces docs/open_issues.md "MHW detection on anomalies": 80 vs 64 events, 61 overlapping, threshold diff max 0.150 °C.
Needs: `source .env` (MHW_DATA_FILE, MHW_CLIM_FILE); run from the repo root with the project venv. Not imported by the pipeline."""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import label, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.hobday import (MAX_GAP, MIN_DURATION, PERCENTILE, REF_YEARS, SMOOTH_WIDTH,
                              WINDOW_HALF_WIDTH, detect, load_ns_mean_clim_smooth_delta,
                              noleap_doy, threshold)
from src.utils.paths import CLIM_FILE, DATA_FILE

ds = xr.open_dataset(DATA_FILE)
time = ds.time
years = time.dt.year.values
doy_cl = time.dt.dayofyear.values.copy(); doy_cl[doy_cl >= 365] = 365  # as dataset.py
doys = noleap_doy(time.dt.month.values, time.dt.day.values)
raw = ds["target"].values.astype(np.float32)
anom = raw + load_ns_mean_clim_smooth_delta()[doy_cl - 1].astype(np.float32)  # as dataset.py

clim = xr.open_dataset(CLIM_FILE)
c_box = clim.mean_clim.sel(lat=slice(51.0, 62.5), lon=slice(-5.2, 13.2)).mean(["lat", "lon"], skipna=True).values
c_s = uniform_filter1d(c_box.astype(np.float64), size=SMOOTH_WIDTH, mode="wrap")
sst = anom.astype(np.float64) + c_s[doy_cl - 1]


def run(x):
    thr = threshold(x, years, doys, ref_years=REF_YEARS, window_half_width=WINDOW_HALF_WIDTH,
                    percentile=PERCENTILE, smooth_width=SMOOTH_WIDTH)
    return thr, detect(x > thr[doys - 1], min_duration=MIN_DURATION, max_gap=MAX_GAP)


thr_a, m_a = run(anom.astype(np.float64))
thr_s, m_s = run(sst)
la, na = label(m_a); ls, ns = label(m_s)
print(f"anomaly : {m_a.sum()} MHW days, {na} events")
print(f"abs SST : {m_s.sum()} MHW days, {ns} events")
both, only_a, only_s = m_a & m_s, m_a & ~m_s, m_s & ~m_a
print(f"days: both {both.sum()}, only anomaly {only_a.sum()}, only abs {only_s.sum()}")
d = time.dt.strftime("%Y-%m-%d").values
print("only anomaly:", list(d[only_a]))
print("only abs    :", list(d[only_s]))
ov_a = sum(1 for k in range(1, na + 1) if m_s[la == k].any())
ov_s = sum(1 for k in range(1, ns + 1) if m_a[ls == k].any())
print(f"anomaly events overlapping an abs event: {ov_a}/{na}; abs events overlapping an anomaly event: {ov_s}/{ns}")
diff_day = (thr_s[doys - 1] - c_s[doy_cl - 1]) - thr_a[doys - 1]
i = np.abs(diff_day).argmax()
print(f"threshold diff on anomaly scale, per day: max|.| {abs(diff_day[i]):.4f} degC on {d[i]}, RMS {np.sqrt((diff_day**2).mean()):.4f}")
leap = years % 4 == 0
print(f"  non-leap years only: max|.| {np.abs(diff_day[~leap]).max():.4f}, RMS {np.sqrt((diff_day[~leap]**2).mean()):.4f}")
