"""
xai_triangulation_summary_plot.py -- synthesis figure for the 4-method
XAI triangulation (IG, GradCAM, GradientSHAP, Occlusion) vs. the raw-data
ground truth, Aug 22 2026 cross-fold check (docs/narrative.md, "Cross-fold
check on the XAI triangulation").

Numbers are copied directly from narrative.md's already-verified 3-fold
mean+-std tables (not recomputed here) -- those numbers went through two
rounds of bug-driven correction (sampling bias #57, normalization #55)
before landing, so this script intentionally does not re-derive them from
the raw .npy, only visualizes the settled result. If narrative.md's
numbers are ever revised, update DECAY/ENRICHMENT below to match and
re-run -- no GPU, CPU-only, seconds to run.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["IG", "GradCAM", "GradientSHAP", "Occlusion"]

# (committed_mean, committed_std, nearest_mean, nearest_std)
DECAY = {
    "IG": (14.80, 0.45, 7.82, 2.59),
    "GradCAM": (1.74, 0.45, 1.35, 0.30),
    "GradientSHAP": (24.54, 8.26, 14.41, 3.43),
    "Occlusion": (5.33, 0.88, 1.62, 0.17),
}
DECAY_RAW_TRUTH = 0.87

ENRICHMENT = {
    "IG": (9.47, 5.18, 6.39, 3.40),
    "GradCAM": (5.16, 1.34, 1.91, 0.81),
    "GradientSHAP": (23.48, 2.40, 15.29, 2.63),
    "Occlusion": (0.71, 0.16, 0.38, 0.06),
}
ENRICHMENT_RAW_TRUTH = 1.39

COLOR_COMMITTED = (
    "#2a78d6"  # categorical slot 1 (blue) -- dataviz skill default palette
)
COLOR_NEAREST = "#eb6834"  # categorical slot 2 (orange)
COLOR_TRUTH = "#52514e"  # muted gray -- reference line, not a data series
COLOR_TEXT = "#0b0b0b"


def _panel(ax, data, raw_truth, title, ylabel):
    x = np.arange(len(METHODS))
    width = 0.36

    committed_mean = [data[m][0] for m in METHODS]
    committed_std = [data[m][1] for m in METHODS]
    nearest_mean = [data[m][2] for m in METHODS]
    nearest_std = [data[m][3] for m in METHODS]

    b1 = ax.bar(
        x - width / 2,
        committed_mean,
        width,
        yerr=committed_std,
        capsize=3,
        color=COLOR_COMMITTED,
        label="committed (land_fill=zero)",
        edgecolor="white",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x + width / 2,
        nearest_mean,
        width,
        yerr=nearest_std,
        capsize=3,
        color=COLOR_NEAREST,
        label="nearest (land_fill=nearest, adopted)",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.axhline(raw_truth, color=COLOR_TRUTH, linestyle="--", linewidth=1.3, zorder=1)
    ax.text(
        len(METHODS) - 0.5,
        raw_truth,
        f"  raw-data ground truth: {raw_truth}x",
        color=COLOR_TRUTH,
        fontsize=9,
        va="bottom",
        ha="right",
    )

    for bars, means, stds in [
        (b1, committed_mean, committed_std),
        (b2, nearest_mean, nearest_std),
    ]:
        for rect, val, std in zip(bars, means, stds):
            ax.annotate(
                f"{val:.1f}x",
                xy=(rect.get_x() + rect.get_width() / 2, val + std),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLOR_TEXT,
            )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel(ylabel, color=COLOR_TEXT)
    ax.set_title(title, fontsize=12, color=COLOR_TEXT, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e3e2dc", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    _panel(
        axes[0],
        DECAY,
        DECAY_RAW_TRUTH,
        "Coastal decay ratio (|IG| at 1-2px from coast / |IG| in open ocean)",
        "decay ratio (log scale)",
    )
    _panel(
        axes[1],
        ENRICHMENT,
        ENRICHMENT_RAW_TRUTH,
        "North Sea box open-water enrichment (>5px from coast)",
        "enrichment ratio (log scale)",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.86])

    fig.suptitle(
        "XAI coastal-artifact triangulation, ptho_bot, full_gnll_quantile_v2 "
        "(mean±std over folds 0-2, mean head)",
        fontsize=13,
        y=0.99,
        color=COLOR_TEXT,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        'Source: docs/narrative.md, "Cross-fold check on the XAI triangulation" '
        "(Aug 22 2026). All 4 methods overstate the raw-data effect; nearest "
        "reduces the overstatement on every method/metric.",
        ha="center",
        fontsize=8.5,
        color=COLOR_TRUTH,
    )

    out = "experiments/figures/xai_triangulation_summary/coastal_artifact_triangulation.png"
    plt.savefig(out, dpi=150, bbox_inches=None)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
