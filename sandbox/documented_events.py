"""Matches 7 documented North Sea MHWs (2007, 2009, 2014, 2018, 2022, Jun 2023, Sep 2023) with our detections A and B.
Produces docs/open_issues.md "MHW detection on anomalies": all 7 appear in A and B, several fragmented (dates, duration, max intensity, overlap).
Needs: `source .env`, abs_vs_anom.py next to it; run from the repo root with the project venv. Not imported by the pipeline."""
import contextlib
import io
import runpy

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label, uniform_filter1d

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(str(Path(__file__).with_name("abs_vs_anom.py")))
years, doys = g["years"], g["doys"]
dates = pd.to_datetime(g["time"].values)
REF, HW, SW = g["REF_YEARS"], g["WINDOW_HALF_WIDTH"], g["SMOOTH_WIDTH"]


def seas(x):
    """Oliver's seasonal climatology: windowed mean (same window as threshold), 31-day smoothed."""
    n = len(x)
    in_ref = np.flatnonzero((years >= REF[0]) & (years <= REF[1]))
    off = np.arange(-HW, HW + 1)
    c = np.empty(365)
    for d in range(1, 366):
        idx = (in_ref[doys[in_ref] == d][:, None] + off[None, :]).ravel()
        c[d - 1] = np.nanmean(x[idx[(idx >= 0) & (idx < n)]])
    return uniform_filter1d(c, size=SW, mode="wrap")[doys - 1]


series = {"A": (g["anom"].astype(np.float64), g["m_a"]), "B": (g["sst"], g["m_s"])}
inten = {k: x - seas(x) for k, (x, _) in series.items()}

documented = [
    ("Jan-May 2007 (S NS + Channel, weak)", "2007-01-01", "2007-05-31"),
    ("Jul 2009 (N NS, ~3 wk)", "2009-07-01", "2009-07-31"),
    ("31 Mar-9 May 2014 (S NS, ~40 d)", "2014-03-31", "2014-05-09"),
    ("Jun-Aug 2018", "2018-06-01", "2018-08-31"),
    ("Aug-Oct 2022 (peak Sep)", "2022-08-01", "2022-10-31"),
    ("13 Jun-4 Jul 2023 (NW NS)", "2023-06-13", "2023-07-04"),
    ("Sep 2023", "2023-09-01", "2023-09-30"),
]
for name, s, e in documented:
    win = (dates >= s) & (dates <= e)
    print(f"\n## {name}  [{s} .. {e}, {win.sum()} d]")
    for k, (x, m) in series.items():
        lab, _ = label(m)
        ids = sorted(set(lab[win & m].tolist()))
        if not ids:
            print(f"  {k}: none")
            continue
        for i in ids:
            ev = lab == i
            j = np.flatnonzero(ev)
            print(f"  {k}: {dates[j[0]]:%Y-%m-%d}..{dates[j[-1]]:%Y-%m-%d}  {len(j)} d  "
                  f"imax {inten[k][ev].max():.2f} degC  overlap {int((ev & win).sum())} d")
