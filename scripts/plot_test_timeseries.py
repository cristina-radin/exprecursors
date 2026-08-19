#!/usr/bin/env python
"""
plot_test_timeseries.py — plot per-fold + combined test predictions vs truth.

Reads test_predictions.npz from each fold and produces:
  1. A combined timeseries (all folds, predictions in one color, truth in another)
  2. A per-fold 5-panel subplot

Usage:
  python scripts/plot_test_timeseries.py --partition full --model_tag mse_v2
  python scripts/plot_test_timeseries.py --partition full --model_tag gnll --smooth_days 30
  python scripts/plot_test_timeseries.py --partition full --model_tag gnll --mhw
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter1d

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"
FIGURES_DIR = REPO_ROOT / "figures"

CLIM_FILE = os.environ.get("MHW_CLIM_FILE", "")
_NS_LAT = slice(50.0, 63.0)
_NS_LON = slice(-5.0, 13.0)


def _load_ns_p90(smooth_days: int = 31) -> np.ndarray:
    """Load NS-mean P90 threshold, shape (365,), with circular smooth."""
    ds = xr.open_dataset(CLIM_FILE)
    p90 = (
        ds.p90_thresh.sel(lat=_NS_LAT, lon=_NS_LON)
        .mean(dim=["lat", "lon"], skipna=True)
        .values
    )
    ds.close()
    return uniform_filter1d(p90, size=smooth_days, mode="wrap")


def _apply_hobday(
    exceedance: np.ndarray, min_dur: int = 5, max_gap: int = 2
) -> np.ndarray:
    """Hobday gap-closure + minimum-duration on a 1D bool array."""
    mhw = exceedance.copy().astype(bool)
    n = len(mhw)
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


def _classify_mhw(dates: np.ndarray, values: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """Classify to_anom timeseries as MHW (bool) using Hobday criteria."""
    import datetime as _dt

    doy_int = np.array(
        [
            _dt.date(int(str(d)[:4]), int(str(d)[5:7]), int(str(d)[8:10]))
            .timetuple()
            .tm_yday
            for d in dates.astype("datetime64[D]").astype(str)
        ]
    )
    thr = p90[np.clip(doy_int - 1, 0, 364)]
    exceedance = values > thr
    return _apply_hobday(exceedance)


def _quantile_map(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Map pred to match truth's empirical CDF (quantile mapping)."""
    sorted_truth = np.sort(truth)
    n = len(sorted_truth)
    pred_rank = np.searchsorted(np.sort(pred), pred) / len(pred)
    pred_rank = np.clip(pred_rank, 0, 1)
    idx = (pred_rank * (n - 1)).astype(int)
    return sorted_truth[idx]


def _count_events(mhw: np.ndarray) -> int:
    """Count number of MHW events (contiguous True blocks) in a bool array."""
    count = 0
    i = 0
    n = len(mhw)
    while i < n:
        if mhw[i]:
            count += 1
            while i < n and mhw[i]:
                i += 1
        else:
            i += 1
    return count


def _run_name(partition: str, model_tag: str, fold: int) -> str:
    tag = f"_{model_tag}" if model_tag else ""
    return f"TbotAtm_{partition}{tag}_seed42_fold{fold}"


def _with_gaps(dates, values, gap_days=10):
    dates = dates.astype("datetime64[D]")
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int) > gap_days
    gap_idx = np.where(gaps)[0]
    values = values.astype(float).copy()
    dates_out = dates.copy()
    if len(gap_idx) == 0:
        return dates_out, values
    dates_out = np.insert(dates_out, gap_idx + 1, dates_out[gap_idx])
    values = np.insert(values, gap_idx + 1, np.nan)
    return dates_out, values


def _rolling_smooth(dates, values, window_days=30):
    """Rolling mean that respects gap boundaries (NaN separators)."""
    s = pd.Series(values, index=pd.DatetimeIndex(dates))
    return s.rolling(f"{window_days}D", min_periods=1, center=True).mean().values


def load_fold(fold: int, partition: str, model_tag: str):
    path = (
        EXPERIMENTS_DIR / _run_name(partition, model_tag, fold) / "test_predictions.npz"
    )
    d = np.load(path, allow_pickle=True)
    dates = d["dates"].astype("datetime64[D]")
    order = np.argsort(dates)
    # Model's own quantile-head output, if this fold's checkpoint had one —
    # key carries its tau (quantile_pred_tau0.90_degC), not read/plotted by
    # this script yet, just made available to callers.
    quantile_key = next((k for k in d.files if k.startswith("quantile_pred_tau")), None)
    result = {
        "dates": dates[order],
        "trues": d["trues_degC"][order],
        "preds": d["preds_degC"][order],
        "r": float(d["r"]),
        "mae": float(d["mae_degC"]),
        "std": d["std_degC"][order] if "std_degC" in d else None,
        "quantile_pred": d[quantile_key][order] if quantile_key else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="full")
    parser.add_argument("--model_tag", default="mse_v2")
    parser.add_argument(
        "--smooth_days",
        type=int,
        default=0,
        help="Rolling-mean window in days for combined plot (0 = no smoothing)",
    )
    parser.add_argument(
        "--mhw",
        action="store_true",
        help="Plot MHW binary classification (0/1) instead of continuous to_anom",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply quantile mapping before MHW classification",
    )
    args = parser.parse_args()

    if args.mhw and not CLIM_FILE:
        parser.error(
            "--mhw requires MHW_CLIM_FILE env var pointing to sst_climatology_doy.nc"
        )

    tag_label = f" ({args.model_tag})" if args.model_tag else ""
    partition_label = {"full": "Full", "remote": "Remote", "local": "Local"}[
        args.partition
    ]
    smooth_suffix = f"_smooth{args.smooth_days}d" if args.smooth_days else ""
    smooth_label = (
        f"  ({args.smooth_days}-day rolling mean)" if args.smooth_days else ""
    )

    folds = [load_fold(f, args.partition, args.model_tag) for f in range(5)]

    # ── 1. Per-fold 5-panel subplot ────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
    fig.suptitle(
        f"TbotAtm {partition_label}{tag_label} — test-set predictions vs truth (per fold)",
        fontsize=14,
        y=0.995,
    )

    for fold, (ax, fd) in enumerate(zip(axes, folds)):
        d_t, v_t = _with_gaps(fd["dates"], fd["trues"])
        d_p, v_p = _with_gaps(fd["dates"], fd["preds"])
        ax.plot(d_t, v_t, color="steelblue", lw=0.9, label="Truth")
        ax.plot(d_p, v_p, color="tomato", lw=0.9, ls="--", label="Prediction")
        if fd["std"] is not None:
            d_s, v_s = _with_gaps(fd["dates"], fd["std"])
            ax.fill_between(
                d_p, v_p - v_s, v_p + v_s, color="tomato", alpha=0.2, label="±1σ"
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("to_anom (°C)", fontsize=9)
        ax.set_title(
            f"Fold {fold}   n={len(fd['dates'])}   r={fd['r']:.3f}   MAE={fd['mae']:.3f}°C",
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, fontsize=8)
    plt.tight_layout()

    out1 = (
        FIGURES_DIR
        / f"TbotAtm_{args.partition}_{args.model_tag}_perfolds_timeseries.png"
    )
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── 2. Combined: all folds on one axis ─────────────────────────────────
    all_dates = np.concatenate([fd["dates"] for fd in folds])
    all_trues = np.concatenate([fd["trues"] for fd in folds])
    all_preds = np.concatenate([fd["preds"] for fd in folds])
    order = np.argsort(all_dates)
    all_dates, all_trues, all_preds = (
        all_dates[order],
        all_trues[order],
        all_preds[order],
    )

    overall_r = float(np.corrcoef(all_preds, all_trues)[0, 1])
    overall_mae = float(np.mean(np.abs(all_preds - all_trues)))

    if args.mhw:
        # ── MHW binary classification plot ─────────────────────────────────
        p90 = _load_ns_p90()
        mhw_truth = _classify_mhw(all_dates, all_trues, p90)

        if args.calibrate:
            preds_for_mhw = _quantile_map(all_preds, all_trues)
            cal_label = "  (quantile-mapped)"
        else:
            preds_for_mhw = all_preds
            cal_label = ""

        mhw_pred = _classify_mhw(all_dates, preds_for_mhw, p90)

        # Metrics
        tp = int((mhw_pred & mhw_truth).sum())
        fp = int((mhw_pred & ~mhw_truth).sum())
        fn = int((~mhw_pred & mhw_truth).sum())
        tn = int((~mhw_pred & ~mhw_truth).sum())
        pod = tp / max(tp + fn, 1)
        false_alarm = fp / max(tp + fp, 1)
        pody = tn / max(tn + fp, 1)

        # Count events
        n_true_events = _count_events(mhw_truth)
        n_pred_events = _count_events(mhw_pred)

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(18, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        # Top panel: binary MHW bars
        d_t, v_t = _with_gaps(all_dates, mhw_truth.astype(float))
        d_p, v_p = _with_gaps(all_dates, mhw_pred.astype(float))
        ax1.fill_between(
            d_t,
            0,
            v_t,
            color="steelblue",
            alpha=0.5,
            label=f"Truth MHW ({n_true_events} events)",
            step="post",
        )
        ax1.fill_between(
            d_p,
            0,
            v_p,
            color="tomato",
            alpha=0.5,
            label=f"Predicted MHW ({n_pred_events} events){cal_label}",
            step="post",
        )
        ax1.set_ylabel("MHW", fontsize=10)
        ax1.set_ylim(-0.05, 1.3)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["No MHW", "MHW"])
        ax1.set_title(
            f"TbotAtm {partition_label}{tag_label} — MHW prediction (Hobday P90){cal_label}   "
            f"POD={pod:.3f}  FAR={false_alarm:.3f}  PODY={pody:.3f}",
            fontsize=11,
        )
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9, loc="upper right")

        # Bottom panel: agreement
        agree = mhw_pred == mhw_truth
        agree_f = _with_gaps(all_dates, agree.astype(float))
        disagree = ~agree
        disagree_f = _with_gaps(all_dates, disagree.astype(float))
        ax2.fill_between(
            agree_f[0],
            0,
            agree_f[1],
            color="seagreen",
            alpha=0.6,
            label="Agree",
            step="post",
        )
        ax2.fill_between(
            disagree_f[0],
            0,
            disagree_f[1],
            color="crimson",
            alpha=0.6,
            label="Disagree",
            step="post",
        )
        ax2.set_ylabel("Match", fontsize=10)
        ax2.set_xlabel("Year", fontsize=10)
        ax2.set_ylim(-0.05, 1.3)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Disagree", "Agree"])
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9, loc="upper right")

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        plt.tight_layout()

        suffix = "_calibrated" if args.calibrate else ""
        out = (
            FIGURES_DIR
            / f"TbotAtm_{args.partition}_{args.model_tag}_combined_mhw{suffix}.png"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    else:
        # ── Continuous to_anom plot ────────────────────────────────────────
        has_any_std = any(fd["std"] is not None for fd in folds)
        if has_any_std:
            all_std = np.concatenate(
                [
                    fd["std"] if fd["std"] is not None else np.zeros_like(fd["preds"])
                    for fd in folds
                ]
            )
            all_std = all_std[order]
        else:
            all_std = None

        do_smooth = args.smooth_days > 0
        if do_smooth:
            all_trues_s = _rolling_smooth(all_dates, all_trues, args.smooth_days)
            all_preds_s = _rolling_smooth(all_dates, all_preds, args.smooth_days)
            all_std_s = (
                _rolling_smooth(all_dates, all_std, args.smooth_days)
                if all_std is not None
                else None
            )
        else:
            all_trues_s, all_preds_s, all_std_s = all_trues, all_preds, all_std

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(18, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        d_t, v_t = _with_gaps(all_dates, all_trues_s)
        d_p, v_p = _with_gaps(all_dates, all_preds_s)
        ax1.plot(d_t, v_t, color="steelblue", lw=1.0, label="Truth", zorder=2)
        if all_std_s is not None:
            d_s, v_s = _with_gaps(all_dates, all_std_s)
            ax1.fill_between(
                d_p,
                v_p - v_s,
                v_p + v_s,
                color="tomato",
                alpha=0.2,
                label="±1σ",
                zorder=1,
            )
        ax1.plot(
            d_p,
            v_p,
            color="tomato",
            lw=0.7,
            ls="--",
            alpha=0.7,
            label="Prediction",
            zorder=3,
        )
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_ylabel("to_anom (°C)", fontsize=10)
        ax1.set_title(
            f"TbotAtm {partition_label}{tag_label} — combined test predictions (5-fold){smooth_label}   "
            f"r={overall_r:.3f}   MAE={overall_mae:.3f}°C",
            fontsize=12,
        )
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9, loc="upper right")

        residuals = all_preds_s - all_trues_s
        d_r, v_r = _with_gaps(all_dates, residuals)
        ax2.fill_between(d_r, v_r, 0, where=v_r >= 0, color="tomato", alpha=0.4)
        ax2.fill_between(d_r, v_r, 0, where=v_r < 0, color="steelblue", alpha=0.4)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel("Residual (°C)", fontsize=10)
        ax2.set_xlabel("Year", fontsize=10)
        ax2.grid(alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
        plt.tight_layout()

        out = (
            FIGURES_DIR
            / f"TbotAtm_{args.partition}_{args.model_tag}_combined_timeseries{smooth_suffix}.png"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
