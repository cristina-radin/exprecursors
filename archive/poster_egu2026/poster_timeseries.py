#!/usr/bin/env python
"""
Poster-ready ensemble prediction timeseries.
Reads from ensemble eval_results/predictions.npz.

Usage:
  python eval/poster_timeseries.py \
    --npz split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output poster_figures/fig_timeseries_poster.png
"""

import argparse

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(str(Path(__file__).parent.parent))
from xai.utils import load_config

FONTSIZE = 19
TICKSIZE = 17


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument(
        "--config", default="split_blockyear/TbotAtm_seed100/config.yaml"
    )
    parser.add_argument("--output", default="poster_figures/fig_timeseries_poster.png")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ens = d["ens_full"]
    trues = d["trues"]
    years = d["years"]
    seeds_preds = d["all_preds"]  # (n_seeds, N)
    seed_std = seeds_preds.std(axis=0)
    n_seeds = seeds_preds.shape[0]

    config = load_config(args.config)
    ds_tmp = xr.open_dataset(config["data_dir"])
    times = ds_tmp.time.values
    ds_tmp.close()

    from datamodule import LazyDataModule

    dm = LazyDataModule(config_path=args.config)
    dm.setup()
    full_ds = dm.train_dataset.dataset
    N = len(ens)
    date_arr = np.array(
        [times[i + full_ds.window_size - 1 + full_ds.lead_time] for i in range(N)]
    )

    thr = args.threshold
    # Hobday definition: 5+ consecutive days above threshold
    raw_mask = trues > thr
    mhw_mask = np.zeros_like(raw_mask, dtype=int)
    consecutive = 0
    for idx in range(len(raw_mask)):
        if raw_mask[idx]:
            consecutive += 1
        else:
            if consecutive >= 5:
                mhw_mask[idx - consecutive : idx] = 1
            consecutive = 0
    if consecutive >= 5:
        mhw_mask[len(raw_mask) - consecutive :] = 1

    decades = [(1985, 1994), (1995, 2004), (2005, 2014), (2015, 2024)]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(20, 10),
        sharex=False,
        sharey=True,
    )
    fig.subplots_adjust(hspace=0.25, bottom=0.13, top=0.93)

    for i, (ax, (yr_start, yr_end)) in enumerate(zip(axes, decades)):
        mask = (years >= yr_start) & (years <= yr_end)
        if not mask.any():
            ax.set_visible(False)
            continue

        ax.fill_between(
            date_arr[mask],
            ens[mask] - seed_std[mask],
            ens[mask] + seed_std[mask],
            color="#e08080",
            alpha=0.30,
            lw=0,
        )
        ax.fill_between(
            date_arr[mask],
            -4,
            4,
            where=(mhw_mask[mask] == 1),
            transform=ax.get_xaxis_transform(),
            alpha=0.12,
            color="gold",
            lw=0,
        )
        (l1,) = ax.plot(
            date_arr[mask],
            trues[mask],
            color="#2c7bb6",
            lw=1.2,
            alpha=0.9,
            label="Observation",
        )
        (l2,) = ax.plot(
            date_arr[mask],
            ens[mask],
            color="#d7191c",
            lw=1.2,
            alpha=0.85,
            ls="--",
            label="Ensemble prediction",
        )

        ax.axhline(thr, color="k", lw=0.8, ls=":")
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlim(date_arr[mask][0], date_arr[mask][-1])
        ax.grid(alpha=0.25)

        # Decade label inside the panel — top LEFT
        ax.text(
            0.01,
            0.93,
            f"{yr_start}–{yr_end}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTSIZE,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
        )

        # Y label — skip per-axis, use fig.text for true centering
        pass

        # X ticks: all panels show years
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=TICKSIZE + 1, rotation=0)
        if i == 3:
            ax.set_xlabel("Year", fontsize=FONTSIZE, labelpad=4)

        ax.tick_params(axis="y", labelsize=TICKSIZE)

    # Legend at bottom of last panel — won't overlap lines
    handles = [
        l1,
        l2,
        Patch(color="gold", alpha=0.4, label="MHW events"),
        Patch(color="#e08080", alpha=0.5, label=f"±1σ ({n_seeds} seeds)"),
    ]
    # Centered Y label across all panels
    fig.text(
        0.08,
        0.53,
        "SST anomaly (norm.)",
        fontsize=FONTSIZE,
        ha="center",
        va="center",
        rotation="vertical",
    )

    axes[-1].legend(
        handles=handles,
        fontsize=FONTSIZE - 2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.80),
        ncol=4,
        framealpha=0.9,
    )

    # Tight suptitle
    fig.suptitle(
        "Ensemble prediction vs ground truth  ·  7-day lead",
        fontsize=FONTSIZE + 1,
        fontweight="bold",
        y=0.97,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
