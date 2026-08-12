#!/usr/bin/env python
"""
Per-year IG heatmap + trend lines.
Reads ig_peryear.npz and plots two panels:
  Top: heatmap (year × variable)
  Bottom: line plot per variable with linear trend

Usage:
  python eval/poster_ig_peryear_trend.py \
    --npz    poster_figures/ig_peryear.npz \
    --output poster_figures/fig_ig_peryear_trend.png
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import linregress

FONTSIZE = 18
TITLESIZE = 20
TICKSIZE = 13

VAR_LABELS = {
    "ptho_bot": "T$_{bottom}$",
    "to_anom": "SST anom.",
    "u10": "U-wind",
    "v10": "V-wind",
    "msl": "SLP",
    "ssr": "Solar rad.",
}

VAR_COLORS = {
    "ptho_bot": "#d7191c",
    "to_anom": "#e85d04",
    "u10": "#2c7bb6",
    "v10": "#4dac26",
    "msl": "#7b3294",
    "ssr": "#f4a11d",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="poster_figures/ig_peryear.npz")
    parser.add_argument("--output", default="poster_figures/fig_ig_peryear_trend.png")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    years = d["years"].astype(int)
    mat = d["importance"]  # (n_years, n_vars)  — % values
    variables = list(d["variables"])
    n_vars = len(variables)
    var_labels = [VAR_LABELS.get(v, v) for v in variables]

    fig, (ax_heat, ax_line) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    fig.subplots_adjust(hspace=0.35)

    # --- Top: heatmap ---
    im = ax_heat.imshow(mat.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())
    ax_heat.set_xticks(range(len(years)))
    ax_heat.set_xticklabels(years, rotation=90, fontsize=TICKSIZE - 1)
    ax_heat.set_yticks(range(n_vars))
    ax_heat.set_yticklabels(var_labels, fontsize=TICKSIZE)
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
    cbar.set_label("% of total |IG|", fontsize=FONTSIZE - 4)
    cbar.ax.tick_params(labelsize=TICKSIZE - 2)
    ax_heat.set_title(
        "Variable importance per year — Integrated Gradients",
        fontsize=TITLESIZE,
        fontweight="bold",
    )

    # --- Bottom: line plot + trend ---
    x = np.arange(len(years))
    for vi, var in enumerate(variables):
        col = VAR_COLORS.get(var, "#888888")
        label = var_labels[vi]
        vals = mat[:, vi]

        ax_line.plot(years, vals, color=col, lw=1.5, alpha=0.6)
        ax_line.scatter(years, vals, color=col, s=20, alpha=0.7)

        # Linear trend
        slope, intercept, r, p, _ = linregress(x, vals)
        trend = slope * x + intercept
        ls = "-" if p < 0.05 else "--"
        trend_label = f"{label}  (slope={slope*10:+.2f}%/decade, p={p:.2f})"
        ax_line.plot(years, trend, color=col, lw=2.5, ls=ls, label=trend_label)

    ax_line.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax_line.set_ylabel("% of total |IG attribution|", fontsize=FONTSIZE)
    ax_line.tick_params(labelsize=TICKSIZE)
    ax_line.grid(alpha=0.25)
    ax_line.legend(
        fontsize=TICKSIZE - 1,
        framealpha=0.9,
        ncol=1,
        loc="upper left",
        title="— significant  -- not significant (p>0.05)",
        title_fontsize=TICKSIZE - 3,
    )
    ax_line.set_title(
        "Linear trend in variable importance (1985–2024)",
        fontsize=TITLESIZE,
        fontweight="bold",
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # Print trend summary
    print("\nTrend summary:")
    for vi, var in enumerate(variables):
        slope, _, r, p, _ = linregress(x, mat[:, vi])
        sig = (
            "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        )
        print(f"  {var:12}  {slope*10:+.3f}%/decade  r={r:.2f}  p={p:.3f}  {sig}")


if __name__ == "__main__":
    main()
