"""Runs marineHeatWaves.detect (ecjoliver, commit d7292bf, GPL-3) on the absolute SST of abs_vs_anom.py and compares with hobday.py.
Produces docs/open_issues.md: 1617 days / 64 events in both, 0 differing days, threshold diff <= 0.0037 °C near Feb 29.
Needs: a TEMPORARY venv outside the repo with marineHeatWaves installed (never a repo dependency), `source .env`; run with python -I."""
import contextlib
import io
import runpy
from datetime import date

from pathlib import Path

import numpy as np

np.NaN = np.nan  # alias removed in NumPy 2; d7292bf still uses it
import marineHeatWaves as mhw_ref  # noqa: E402
from scipy.ndimage import label

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("abs_vs_anom.py")))
sst, m_ours, thr_ours, doys = g["sst"], g["m_s"], g["thr_s"], g["doys"]
time = g["time"]
# Real calendar dates as ordinals; the series simply has no Feb 29 entries.
t = np.array([date(int(y), int(m), int(d)).toordinal() for y, m, d in
              zip(time.dt.year.values, time.dt.month.values, time.dt.day.values)])

mhws, clim = mhw_ref.detect(t, sst.copy(), climatologyPeriod=[1985, 2014], pctile=90,
                            windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31,
                            minDuration=5, joinAcrossGaps=True, maxGap=2)
m_ref = np.zeros(len(t), bool)
for s, e in zip(mhws["index_start"], mhws["index_end"]):
    m_ref[s:e + 1] = True
print(f"ours (hobday.py): {m_ours.sum()} days, {label(m_ours)[1]} events")
print(f"Oliver detect   : {m_ref.sum()} days (by index), {mhws['n_events']} events, "
      f"sum(duration)={int(np.sum(mhws['duration']))}")
diff = m_ours != m_ref
dates = time.dt.strftime("%Y-%m-%d").values
print("days that differ:", list(dates[diff]))
thr_ref = clim["thresh"]
td = thr_ref - thr_ours[doys - 1]
i = np.abs(td).argmax()
print(f"threshold diff (Oliver - ours): max|.| {abs(td[i]):.2e} on {dates[i]}")
nonleap = time.dt.year.values % 4 != 0
print(f"  non-leap years: max|.| {np.abs(td[nonleap]).max():.2e}")
mo = time.dt.month.values; dy = time.dt.day.values
far = ~(((mo == 1) & (dy >= 30)) | (mo == 2) | ((mo == 3) & (dy <= 31)) | ((mo == 4) & (dy <= 2)))
print(f"  away from Feb 29 +-31 days: max|.| {np.abs(td[far]).max():.2e}")
