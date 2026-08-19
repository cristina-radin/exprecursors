#!/usr/bin/env python
"""
plot_test_timeseries.py — plot per-fold + combined test predictions vs truth.

Reads test_predictions.npz from each fold and produces:
  1. A combined timeseries (all folds, predictions in one color, truth in another)
  2. A per-fold 5-panel subplot

Usage:
  python scripts/plot_test_timeseries.py --partition full --model_tag mse_v2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"
FIGURES_DIR = REPO_ROOT / "figures"


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


def load_fold(fold: int, partition: str, model_tag: str):
    path = (
        EXPERIMENTS_DIR / _run_name(partition, model_tag, fold) / "test_predictions.npz"
    )
    d = np.load(path, allow_pickle=True)
    dates = d["dates"].astype("datetime64[D]")
    order = np.argsort(dates)
    result = {
        "dates": dates[order],
        "trues": d["trues_degC"][order],
        "preds": d["preds_degC"][order],
        "r": float(d["r"]),
        "mae": float(d["mae_degC"]),
        "std": d["std_degC"][order] if "std_degC" in d else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", default="full")
    parser.add_argument("--model_tag", default="mse_v2")
    args = parser.parse_args()

    tag_label = f" ({args.model_tag})" if args.model_tag else ""
    partition_label = {"full": "Full", "remote": "Remote", "local": "Local"}[
        args.partition
    ]

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

    # ── 2. Combined: all folds on one axis (no overlap → truth all blue, preds all red) ──
    all_dates = np.concatenate([fd["dates"] for fd in folds])
    all_trues = np.concatenate([fd["trues"] for fd in folds])
    all_preds = np.concatenate([fd["preds"] for fd in folds])
    has_any_std = any(fd["std"] is not None for fd in folds)
    if has_any_std:
        all_std = np.concatenate(
            [
                fd["std"] if fd["std"] is not None else np.zeros_like(fd["preds"])
                for fd in folds
            ]
        )
    else:
        all_std = None
    order = np.argsort(all_dates)
    all_dates, all_trues, all_preds = (
        all_dates[order],
        all_trues[order],
        all_preds[order],
    )
    if all_std is not None:
        all_std = all_std[order]

    overall_r = float(np.corrcoef(all_preds, all_trues)[0, 1])
    overall_mae = float(np.mean(np.abs(all_preds - all_trues)))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Top panel: truth + prediction overlay
    d_t, v_t = _with_gaps(all_dates, all_trues)
    d_p, v_p = _with_gaps(all_dates, all_preds)
    ax1.plot(d_t, v_t, color="steelblue", lw=1.0, label="Truth", zorder=2)
    if all_std is not None:
        d_s, v_s = _with_gaps(all_dates, all_std)
        ax1.fill_between(
            d_p, v_p - v_s, v_p + v_s, color="tomato", alpha=0.2, label="±1σ", zorder=1
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
        f"TbotAtm {partition_label}{tag_label} — combined test predictions (5-fold)   "
        f"r={overall_r:.3f}   MAE={overall_mae:.3f}°C",
        fontsize=12,
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, loc="upper right")

    # Bottom panel: residuals (pred - truth)
    residuals = all_preds - all_trues
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

    out2 = (
        FIGURES_DIR
        / f"TbotAtm_{args.partition}_{args.model_tag}_combined_timeseries.png"
    )

    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
