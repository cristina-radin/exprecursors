"""
composite_bootstrap_ci.py — bootstrap CI for composite_precursor_analysis.py's
NS-box curve (Aug 21 2026, user request after asking what a bootstrap CI even
is). Same descriptive/non-ML framing as the parent script: not a paper result,
for understanding whether the composite curves (n=52 real onset events) are
distinguishable from noise before citing them anywhere.

Reuses the exact same onset-event and season-matched-control-pool definitions
as composite_precursor_analysis.py (same P90/Hobday onset detection, same
SEASON_WINDOW_DAYS=10, EXCLUSION_BUFFER_DAYS=14) so the point estimate here
reproduces that script's composite_ns_box_curve.npz exactly -- verified as a
consistency check before trusting the CI.

Two-sample bootstrap per (variable, k): resample the n=52 onset events with
replacement, resample the season-matched control pool with replacement
(independently), take diff of means, repeat B=10000 times, report the
2.5/97.5 percentile CI. CPU only, no GPU, no model.
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

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402

_NS_LAT_SLICE = slice(50.0, 63.0)
_NS_LON_SLICE = slice(-5.0, 13.0)
LAGS = list(range(1, 15))
SEASON_WINDOW_DAYS = 10
EXCLUSION_BUFFER_DAYS = 14
VARIABLES = ["ptho_bot", "u10", "v10", "msl", "ssr"]
N_BOOT = 10000
RNG_SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-npz",
        default="experiments/figures/step7_persistence/composite_precursor/composite_ns_box_curve.npz",
        help="Prior run's point-estimate npz, used only as a consistency check.",
    )
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    print("Loading merged_daily.nc...", flush=True)
    ds = xr.open_dataset(DATA_FILE)
    time = pd.to_datetime(ds.time.values)
    doys = time.dayofyear.values
    lats = ds.lat.values
    lons = ds.lon.values

    target = ds["target"].values.astype(np.float64)

    p90 = load_ns_p90()
    doy_idx = np.clip(doys, 1, 365) - 1
    exceed = target > p90[doy_idx]
    mhw = apply_hobday(exceed)
    onset_idx = np.where(mhw & ~np.roll(mhw, 1))[0]
    if mhw[0]:
        onset_idx = onset_idx[onset_idx != 0]
    onset_idx = onset_idx[onset_idx >= max(LAGS)]
    print(f"Real onsets: n={len(onset_idx)}", flush=True)

    near_mhw = mhw.copy()
    for shift in range(1, EXCLUSION_BUFFER_DAYS + 1):
        near_mhw |= np.roll(mhw, shift)
        near_mhw |= np.roll(mhw, -shift)
    eligible_control_day = ~near_mhw

    ns_lat_i = (lats >= 50.0) & (lats <= 63.0)
    ns_lon_i = (lons >= -5.0) & (lons <= 13.0)

    fields = {var: ds[var].values.astype(np.float32) for var in VARIABLES}

    ref = None
    ref_path = REPO_ROOT / args.reference_npz
    if ref_path.exists():
        ref = np.load(ref_path)

    point_curve = {var: [] for var in VARIABLES}
    ci_lo = {var: [] for var in VARIABLES}
    ci_hi = {var: [] for var in VARIABLES}

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
            f"  k={k:2d}d  n_onset={len(onset_lag_idx)}  n_control_pool={len(control_lag_idx)}",
            flush=True,
        )

        for var in VARIABLES:
            f = fields[var]
            onset_vals_ns = np.nanmean(
                f[onset_lag_idx][:, ns_lat_i][:, :, ns_lon_i], axis=(1, 2)
            )
            control_vals_ns = np.nanmean(
                f[control_lag_idx][:, ns_lat_i][:, :, ns_lon_i], axis=(1, 2)
            )

            point_diff = onset_vals_ns.mean() - control_vals_ns.mean()
            point_curve[var].append(point_diff)

            n_onset, n_control = len(onset_vals_ns), len(control_vals_ns)
            onset_boot_idx = rng.integers(0, n_onset, size=(N_BOOT, n_onset))
            control_boot_idx = rng.integers(0, n_control, size=(N_BOOT, n_control))
            boot_diffs = onset_vals_ns[onset_boot_idx].mean(axis=1) - control_vals_ns[
                control_boot_idx
            ].mean(axis=1)
            lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
            ci_lo[var].append(lo)
            ci_hi[var].append(hi)

            if ref is not None and f"{var}_diff" in ref.files:
                ref_val = ref[f"{var}_diff"][LAGS.index(k)]
                if abs(ref_val - point_diff) > 1e-4:
                    print(
                        f"  WARNING: {var} k={k} point estimate mismatch vs "
                        f"reference npz: {point_diff:.6f} vs {ref_val:.6f}",
                        flush=True,
                    )

        # incremental save every lag, per repo convention for jobs >1h
        # (this job is short, but cheap to do anyway)
        np.savez(
            out_dir / "composite_ns_box_curve_bootstrap.npz",
            lags=np.array(LAGS[: LAGS.index(k) + 1]),
            **{f"{var}_diff": np.array(point_curve[var]) for var in VARIABLES},
            **{f"{var}_ci_lo": np.array(ci_lo[var]) for var in VARIABLES},
            **{f"{var}_ci_hi": np.array(ci_hi[var]) for var in VARIABLES},
        )

    print("\n=== Point estimate ± 95% bootstrap CI (n_boot=10000) ===", flush=True)
    for var in VARIABLES:
        print(f"-- {var} --", flush=True)
        for i, k in enumerate(LAGS):
            sig = (
                "significant"
                if (ci_lo[var][i] > 0 or ci_hi[var][i] < 0)
                else "NOT significant"
            )
            print(
                f"  k={k:2d}d  diff={point_curve[var][i]:+.4f}  "
                f"95% CI=[{ci_lo[var][i]:+.4f}, {ci_hi[var][i]:+.4f}]  {sig}",
                flush=True,
            )

    fig, axes = plt.subplots(
        len(VARIABLES), 1, figsize=(9, 4 * len(VARIABLES)), sharex=True
    )
    for ax, var in zip(axes, VARIABLES):
        lo = np.array(ci_lo[var])
        hi = np.array(ci_hi[var])
        mid = np.array(point_curve[var])
        ax.fill_between(LAGS, lo, hi, alpha=0.25, label="95% bootstrap CI")
        ax.plot(LAGS, mid, "o-", label=var)
        ax.axhline(0, color="gray", lw=0.8)
        ax.invert_xaxis()
        ax.set_ylabel(f"{var}\nNS-box diff")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Days before onset (k)")
    fig.suptitle(
        "Composite precursor curve with 95% bootstrap CI (n=52 real onsets, descriptive only)"
    )
    plt.tight_layout()
    plt.savefig(
        out_dir / "composite_ns_box_curve_bootstrap.png", dpi=150, bbox_inches="tight"
    )
    print(f"Saved {out_dir / 'composite_ns_box_curve_bootstrap.png'}", flush=True)


if __name__ == "__main__":
    main()
