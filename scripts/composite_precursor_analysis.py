"""
composite_precursor_analysis.py — item 4 from the user's Aug 21 2026 idea
list ("Análisis de precursores (composite)" + Granger causality + spatial
map). Explicitly NOT a sellable ML-improvement result for the paper (the
user's own framing) -- this is a descriptive, model-free look at whether
a spatial/temporal precursor pattern exists in the raw data before real
MHW onsets, kept for understanding/motivation, not a performance claim.

Framing decisions (confirmed with the user before building this, Aug 21
2026): control group = season-matched non-onset days (same DOY band,
excluding any day near an MHW event) rather than all non-MHW days
(avoids conflating a precursor signal with plain seasonality); lag
window k=1-14 days before onset (same range as the persistence lag
sweep already done today, for direct comparability).

merged_daily.nc's ptho_bot/u10/v10/msl/ssr/to_anom are ALL already
anomaly fields (DOY climatology, +-5d window, ref 1985-2014 already
removed at the source -- see the file's own `description` attribute) --
no additional climatology computation needed here, composite directly
on the stored fields.

Two analyses:
  1. Spatial composite: mean(anomaly field, k days before onset) minus
     mean(anomaly field, k days before season-matched control), per
     variable, for k in {1,3,7,14} plus a k=1-14-averaged summary map.
  2. Granger causality: NS-box-mean anomaly series per variable vs.
     target, lags 1-14 (statsmodels, already in requirements.txt).

Onset events are found on the FULL CONTIGUOUS 40-year target series
(not per-fold) via load_ns_p90()/apply_hobday() -- deliberately avoids
known_issues.md #53/#54's per-fold year-splitting concerns entirely,
since this analysis has no train/test split to respect.

CPU only.

Usage:
  python scripts/composite_precursor_analysis.py \
      --output experiments/figures/step7_persistence/composite_precursor
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.tsa.stattools import grangercausalitytests

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402

_NS_LAT_SLICE = slice(50.0, 63.0)
_NS_LON_SLICE = slice(-5.0, 13.0)
LAGS = list(range(1, 15))
SNAPSHOT_LAGS = [1, 3, 7, 14]
SEASON_WINDOW_DAYS = 10
EXCLUSION_BUFFER_DAYS = 14
VARIABLES = ["ptho_bot", "u10", "v10", "msl", "ssr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged_daily.nc...", flush=True)
    ds = xr.open_dataset(DATA_FILE)
    time = pd.to_datetime(ds.time.values)
    doys = time.dayofyear.values
    lats = ds.lat.values
    lons = ds.lon.values
    land_mask = ds["land_mask"].values

    target = ds["target"].values.astype(np.float64)
    valid = ~np.isnan(target)
    assert valid.all(), "expected target to be fully valid over the record"
    n = len(target)
    print(f"n={n} days, {time[0].date()} to {time[-1].date()}", flush=True)

    p90 = load_ns_p90()
    doy_idx = np.clip(doys, 1, 365) - 1
    exceed = target > p90[doy_idx]
    mhw = apply_hobday(exceed)
    onset_idx = np.where(mhw & ~np.roll(mhw, 1))[0]
    if mhw[0]:
        onset_idx = onset_idx[onset_idx != 0]
    onset_idx = onset_idx[onset_idx >= max(LAGS)]  # need full lookback window
    print(f"Real onsets on full contiguous series: n={len(onset_idx)}", flush=True)
    print(f"  onset years: {sorted(set(time[onset_idx].year))}", flush=True)

    # Exclusion buffer for control-pool eligibility: any day within
    # EXCLUSION_BUFFER_DAYS of an MHW day is not a "typical" day.
    near_mhw = mhw.copy()
    for shift in range(1, EXCLUSION_BUFFER_DAYS + 1):
        near_mhw |= np.roll(mhw, shift)
        near_mhw |= np.roll(mhw, -shift)
    eligible_control_day = ~near_mhw
    print(
        f"Eligible control days (>{EXCLUSION_BUFFER_DAYS}d from any MHW day): "
        f"{eligible_control_day.sum()} / {n}",
        flush=True,
    )

    print(
        "\n=== Granger causality (NS-box-mean series, target vs each variable) ===",
        flush=True,
    )
    ns_series = {}
    for var in VARIABLES:
        field = ds[var].sel(lat=_NS_LAT_SLICE, lon=_NS_LON_SLICE)
        ns_series[var] = field.mean(dim=["lat", "lon"], skipna=True).values.astype(
            np.float64
        )

    granger_results = {}
    for var in VARIABLES:
        x = ns_series[var]
        m = ~(np.isnan(target) | np.isnan(x))
        data = np.column_stack([target[m], x[m]])
        print(f"\n-- {var} (n={m.sum()}) --", flush=True)
        try:
            res = grangercausalitytests(data, maxlag=max(LAGS), verbose=False)
        except Exception as e:
            print(f"  Granger test failed for {var}: {e}", flush=True)
            continue
        pvals = []
        for lag in LAGS:
            p = res[lag][0]["ssr_ftest"][1]
            pvals.append(p)
            marker = " *" if p < 0.05 else ""
            print(f"  lag={lag:2d}d  F-test p={p:.4f}{marker}", flush=True)
        granger_results[var] = pvals

    fig, ax = plt.subplots(figsize=(9, 5))
    for var in VARIABLES:
        if var in granger_results:
            ax.plot(LAGS, granger_results[var], "o-", label=var)
    ax.axhline(0.05, color="red", ls="--", lw=1, label="p=0.05")
    ax.set_xlabel("Lag (days, predictor leads target)")
    ax.set_ylabel("Granger F-test p-value")
    ax.set_title(
        "Granger causality: does variable(t-lag) help predict target(t)?\n(NS-box-mean series, full 40yr record — descriptive only, not an ML result)"
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "granger_causality_pvalues.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_dir / 'granger_causality_pvalues.png'}", flush=True)
    np.savez(
        out_dir / "granger_causality_pvalues.npz",
        lags=np.array(LAGS),
        **granger_results,
    )

    print(
        "\n=== Spatial composite: onset-preceding vs season-matched control ===",
        flush=True,
    )
    fields = {var: ds[var].values.astype(np.float32) for var in VARIABLES}

    ns_curve = {var: [] for var in VARIABLES}
    ns_curve_control = {var: [] for var in VARIABLES}
    snapshot_maps = {var: {} for var in VARIABLES}

    for k in LAGS:
        onset_lag_idx = onset_idx - k

        control_days = []
        for oi in onset_idx:
            target_doy = doys[oi]
            circ_dist = np.minimum(
                np.abs(doys - target_doy), 365 - np.abs(doys - target_doy)
            )
            season_mask = circ_dist <= SEASON_WINDOW_DAYS
            pool = np.where(season_mask & eligible_control_day)[0]
            pool = pool[pool >= k]
            control_days.append(pool - k)
        control_lag_idx = np.concatenate(control_days)

        print(
            f"  k={k:2d}d  onset window n={len(onset_lag_idx)}  "
            f"control pool n={len(control_lag_idx)}",
            flush=True,
        )

        for var in VARIABLES:
            f = fields[var]
            onset_vals = f[onset_lag_idx]
            control_vals = f[control_lag_idx]
            onset_mean_map = np.nanmean(onset_vals, axis=0)
            control_mean_map = np.nanmean(control_vals, axis=0)
            diff_map = onset_mean_map - control_mean_map

            ns_lat_i = (lats >= 50.0) & (lats <= 63.0)
            ns_lon_i = (lons >= -5.0) & (lons <= 13.0)
            ns_diff = np.nanmean(diff_map[np.ix_(ns_lat_i, ns_lon_i)])
            ns_curve[var].append(ns_diff)
            ns_curve_control[var].append(
                np.nanmean(control_mean_map[np.ix_(ns_lat_i, ns_lon_i)])
            )

            if k in SNAPSHOT_LAGS:
                snapshot_maps[var][k] = diff_map

    np.savez(
        out_dir / "composite_ns_box_curve.npz",
        lags=np.array(LAGS),
        **{f"{var}_diff": np.array(ns_curve[var]) for var in VARIABLES},
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for var in VARIABLES:
        ax.plot(LAGS, ns_curve[var], "o-", label=var)
    ax.axhline(0, color="gray", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("Days before onset (k)")
    ax.set_ylabel("NS-box mean anomaly: onset-preceding minus season-matched control")
    ax.set_title("Composite precursor curve (NS-box mean, descriptive only)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "composite_ns_box_curve.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'composite_ns_box_curve.png'}", flush=True)

    for var in VARIABLES:
        for k in SNAPSHOT_LAGS:
            data = snapshot_maps[var][k].copy()
            data[land_mask == 0] = np.nan
            fig, ax = plt.subplots(figsize=(10, 5))
            vmax = np.nanpercentile(np.abs(data), 98)
            im = ax.pcolormesh(
                lons, lats, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
            )
            ax.contour(lons, lats, land_mask, levels=[0.5], colors="k", linewidths=0.6)
            ax.set_facecolor("lightgray")
            ax.set_title(
                f"{var}: composite anomaly diff (onset-k={k}d) — season-matched control\n"
                f"n_onsets={len(onset_idx)} (descriptive, not an ML result)",
                fontsize=10,
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, shrink=0.7)
            plt.tight_layout()
            fname = out_dir / f"composite_{var}_k{k}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
    print(
        f"\nSaved composite snapshot maps for k={SNAPSHOT_LAGS} to {out_dir}",
        flush=True,
    )

    ds.close()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
