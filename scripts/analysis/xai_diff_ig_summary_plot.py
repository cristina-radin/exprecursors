"""
xai_diff_ig_summary_plot.py -- synthesis figure for the corrected
differential IG (q_pred - y_hat_mean) result, `nearest` model, folds 0-2,
stratified sampling (Aug 24 2026 rerun, supersedes the biased/committed
Aug 21 result -- see docs/narrative.md).

Top panel: variable importance (share of total |diff-IG|), mean+-std over
3 folds, plus sign consistency (u10 is the only variable with a stable
sign across folds).
Bottom panel: u10's diff map for each fold side by side, to show the
dipole pattern (broad negative band + North Sea/Baltic positive patch)
is consistent, not a single-fold fluke.

Recomputes directly from the saved .npy (no GPU) -- avoids hand-copying
numbers that were already computed once in conversation.
"""

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

VARIABLES = ["ptho_bot", "u10", "v10", "msl", "ssr"]
FOLDERS = {
    0: "experiments/figures/xai_integrated_gradients/ig_diff_nearest_fold0/ig_diff_head.npy",
    1: "experiments/figures/xai_integrated_gradients/ig_diff_nearest_fold1/ig_diff_head.npy",
    2: "experiments/figures/xai_integrated_gradients/ig_diff_nearest_fold2/ig_diff_head.npy",
}

COLOR_STABLE = "#2a78d6"  # categorical slot 1 -- u10, the one stable/citable variable
COLOR_OTHER = "#c3c2b7"  # muted neutral -- other 4 variables, not individually reliable
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#52514e"


def main():
    arrs = {f: np.load(p) for f, p in FOLDERS.items()}

    shares = {v: [] for v in VARIABLES}
    pos_fracs = {v: [] for v in VARIABLES}
    for f, arr in arrs.items():
        tot = np.array([np.abs(arr[i]).sum() for i in range(5)])
        for i, v in enumerate(VARIABLES):
            shares[v].append(tot[i] / tot.sum())
            d = arr[i]
            pos, neg = d[d > 0].sum(), -d[d < 0].sum()
            pos_fracs[v].append(pos / (pos + neg))

    order = sorted(VARIABLES, key=lambda v: -np.mean(shares[v]))

    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.6, wspace=0.15)
    ax_bar = fig.add_subplot(gs[0, :])
    ax_maps = [fig.add_subplot(gs[1, i]) for i in range(3)]

    x = np.arange(5)
    means = [np.mean(shares[v]) for v in order]
    stds = [np.std(shares[v]) for v in order]
    colors = [COLOR_STABLE if v == "u10" else COLOR_OTHER for v in order]
    bars = ax_bar.bar(
        x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.5
    )
    for xi, v, m, s in zip(x, order, means, stds):
        sign = np.mean(pos_fracs[v])
        sign_std = np.std(pos_fracs[v])
        label = f"{m:.1%}\n(sign: {sign:.0%}±{sign_std:.0%} pos.)"
        ax_bar.annotate(
            label,
            xy=(xi, m + s),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=COLOR_TEXT,
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(order, fontsize=11)
    ax_bar.set_ylabel("share of total |diff IG|", color=COLOR_TEXT)
    ax_bar.set_ylim(0, max(means) + max(stds) + 0.13)
    ax_bar.set_title(
        "Differential IG (q_pred − mean) variable importance — nearest model, "
        "mean±std over folds 0-2",
        fontsize=12.5,
        loc="left",
        color=COLOR_TEXT,
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="y", color="#e3e2dc", linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.text(
        0.99,
        0.95,
        "u10 = only variable both high-magnitude AND fold-stable\n(sign 96%+ negative in every fold)",
        transform=ax_bar.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=COLOR_STABLE,
        fontweight="bold",
    )

    nc = xr.open_dataset(os.environ["MHW_DATA_FILE"])
    lats, lons = nc.lat.values, nc.lon.values
    land_mask = nc["land_mask"].values
    nc.close()
    vi = VARIABLES.index("u10")

    vmax = max(np.percentile(np.abs(arrs[f][vi]), 98) for f in [0, 1, 2])
    for f, ax in zip([0, 1, 2], ax_maps):
        data = arrs[f][vi]
        im = ax.pcolormesh(
            lons, lats, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
        )
        ax.contour(lons, lats, land_mask, levels=[0.5], colors="k", linewidths=0.4)
        ax.text(
            0.03,
            0.96,
            f"fold{f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=COLOR_TEXT,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
        )
        ax.set_xticks([])
        ax.set_yticks([])
    cbar = fig.colorbar(
        im, ax=ax_maps, shrink=0.85, pad=0.02, label="mean signed diff IG"
    )
    fig.text(
        0.5,
        0.485,
        "u10 differential map, all 3 folds — same dipole: "
        "broad negative band + North Sea/Baltic positive patch",
        fontsize=11.5,
        color=COLOR_TEXT,
        ha="center",
    )

    fig.text(
        0.02,
        0.005,
        "Source: docs/narrative.md, differential-IG rerun (Aug 24 2026), jobs "
        "29565673/74/75. Corrects the Aug 21 biased-sampling/committed-model result.",
        fontsize=8,
        color=COLOR_MUTED,
    )

    out = "experiments/figures/xai_triangulation_summary/differential_ig_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
