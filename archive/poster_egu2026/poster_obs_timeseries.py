#!/usr/bin/env python
"""
Observational MHW timeseries figure (2-panel), full North Atlantic domain.
Panel (a): Basin-mean Sea Temperature Anomaly (8m), monthly.
Panel (b): Mean MHW days per ocean pixel per month (Hobday 2016 per pixel).

Usage:
  python eval/poster_obs_timeseries.py \
    --output poster_figures/fig_obs_timeseries.png
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

FONTSIZE = 16
TICKSIZE = 13
REF_START, REF_END = 1985, 2014
MIN_DURATION = 5
GAP_CLOSURE = 2


def apply_hobday_1d(x):
    x = x.astype(np.int8).copy()
    ones_idx = np.where(x == 1)[0]
    if len(ones_idx) >= 2:
        diffs = np.diff(ones_idx)
        for i in np.where((diffs > 1) & (diffs <= GAP_CLOSURE + 1))[0]:
            x[ones_idx[i] + 1 : ones_idx[i + 1]] = 1
    result = np.zeros_like(x)
    d = np.diff(np.concatenate([[0], x, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        if e - s >= MIN_DURATION:
            result[s:e] = 1
    return result


def rolling_monthly(series, months=60):
    return series.rolling(months, center=True, min_periods=12).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nc",
        default="/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc",
    )
    parser.add_argument("--output", default="poster_figures/fig_obs_timeseries.png")
    args = parser.parse_args()

    print("Loading data...")
    ds = xr.open_dataset(args.nc)
    to_anom = ds["to_anom"].values  # (T, nlat, nlon)
    land_mask = ds["land_mask"].values
    times = pd.DatetimeIndex(ds["time"].values)
    ds.close()

    ocean = land_mask == 1
    ref_mask = (times.year >= REF_START) & (times.year <= REF_END)
    n_ocn = int(ocean.sum())

    # --- Panel (a): basin-mean temperature anomaly ---
    basin = np.nanmean(to_anom[:, ocean], axis=1)  # (T,)
    basin -= basin[ref_mask].mean()
    print(f"  Basin-mean anom range: {basin.min():.3f} – {basin.max():.3f} °C")

    monthly_anom = pd.Series(basin, index=times).resample("MS").mean()
    trend_anom = rolling_monthly(monthly_anom)

    # --- Panel (b): Hobday pixel-by-pixel ---
    print("Applying Hobday per pixel...")
    ocean_idx = np.argwhere(ocean)
    hobday_sum = np.zeros(len(times), dtype=np.float32)

    for k, (j, i) in enumerate(ocean_idx):
        ts = to_anom[:, j, i]
        thr = np.percentile(ts[ref_mask], 90)
        hobday_sum += apply_hobday_1d((ts > thr).astype(np.int8)).astype(np.float32)
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{n_ocn}", end="\r")
    print()

    hobday_mean = hobday_sum / n_ocn  # fraction per day
    monthly_mhw = pd.Series(hobday_mean, index=times).resample("MS").sum()
    trend_mhw = rolling_monthly(monthly_mhw)
    print(
        f"  Monthly MHW days range: {monthly_mhw.min():.1f} – {monthly_mhw.max():.1f}"
    )

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    fig.subplots_adjust(hspace=0.45, top=0.93, bottom=0.08)

    t = monthly_anom.index
    ref_start = pd.Timestamp(f"{REF_START}-01-01")
    ref_end = pd.Timestamp(f"{REF_END}-12-31")

    for ax in (ax1, ax2):
        ax.axvspan(
            ref_start,
            ref_end,
            color="lightgrey",
            alpha=0.4,
            lw=0,
            zorder=0,
            label=f"Reference {REF_START}–{REF_END}",
        )

    colors = ["#c0392b" if v >= 0 else "#2980b9" for v in monthly_anom]
    ax1.bar(t, monthly_anom.values, width=28, color=colors, alpha=0.75, zorder=2)
    ax1.plot(
        t,
        trend_anom.values,
        color="#1a1a6e",
        lw=2.0,
        label="5-yr smoothed trend",
        zorder=3,
    )
    ax1.axhline(0, color="k", lw=0.6, ls="--")
    ax1.set_ylabel("Temperature anomaly (°C)", fontsize=FONTSIZE)
    ax1.tick_params(labelsize=TICKSIZE)
    ax1.legend(fontsize=TICKSIZE - 1, framealpha=0.9, loc="upper left")
    ax1.set_title(
        "Basin-mean Sea Temperature Anomaly (8 m)",
        fontsize=FONTSIZE + 1,
        fontweight="bold",
    )
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", labelsize=TICKSIZE)
    ax1.set_xlim(t[0], t[-1])

    ax2.bar(t, monthly_mhw.values, width=28, color="#c0392b", alpha=0.65, zorder=2)
    ax2.plot(
        t,
        trend_mhw.values,
        color="#1a1a6e",
        lw=2.0,
        label="5-yr smoothed trend",
        zorder=3,
    )
    ax2.set_ylabel("MHW days per month", fontsize=FONTSIZE)
    ax2.tick_params(axis="y", labelsize=TICKSIZE)
    ax2.tick_params(axis="x", labelsize=TICKSIZE)
    ax2.legend(fontsize=TICKSIZE - 1, framealpha=0.9, loc="upper left")
    ax2.set_title(
        "Basin-mean MHW Monthly Frequency (8 m)",
        fontsize=FONTSIZE + 1,
        fontweight="bold",
        pad=8,
    )

    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlabel("Year", fontsize=FONTSIZE)
    ax2.set_xlim(t[0], t[-1])

    fig.align_ylabels([ax1, ax2])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
