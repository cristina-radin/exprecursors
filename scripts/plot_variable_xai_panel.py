#!/usr/bin/env python
"""
Compose existing IG spatial maps for noSST / SST / deepSST into one panel.
Shows the dominant variable per model (v10, to_anom, ptho_bot) for 1985-2024.

Usage:
  python eval/plot_variable_xai_panel.py --output_dir eval_results/variable_xai_panel
"""

import argparse

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

BASE = Path(__file__).parents[1] / "split_blockyear"

# For each model: show the most physically meaningful variable
PANELS = [
    {
        "model": "noSST_layers2",
        "var": "v10",
        "title": "noSST  —  dominant var: v10\n(atmosphere only)",
        "subtitle": "r = 0.683  |  ETS = 0.337",
        "period": "2015-2024",
    },
    {
        "model": "SST_layers2",
        "var": "to_anom",
        "title": "SST  —  dominant var: to_anom\n(+ surface SST anomaly)",
        "subtitle": "r = 0.864  |  ETS = 0.493",
        "period": "2015-2024",
    },
    {
        "model": "deepSST_layers2",
        "var": "ptho_bot",
        "title": "deepSST  —  dominant var: ptho_bot\n(+ bottom ocean temperature)",
        "subtitle": "r = 0.717  |  ETS = TBD",
        "period": "2015-2024",
    },
]

# Also show v10 for SST and deepSST to show how atmosphere changes
ATMOS_PANELS = [
    {"model": "noSST_layers2", "var": "v10", "label": "noSST"},
    {"model": "SST_layers2", "var": "v10", "label": "SST"},
    {"model": "deepSST_layers2", "var": "v10", "label": "deepSST"},
]


def load_img(model, var, period):
    path = BASE / model / "xai_results" / period / f"ig_spatial_{var}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return mpimg.imread(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="eval_results/variable_xai_panel")
    parser.add_argument(
        "--period",
        default="2015-2024",
        help="XAI period to show (1985-2004, 2005-2014, 2015-2024)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Figure A: dominant variable per model (3 panels)                    #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Integrated Gradients — dominant variable per model  ({args.period})\n"
        "Block-year split  |  2 LSTM layers  |  lead = 7 days",
        fontsize=13,
        y=1.02,
    )

    for ax, p in zip(axes, PANELS):
        try:
            img = load_img(p["model"], p["var"], args.period)
            ax.imshow(img)
        except FileNotFoundError as e:
            ax.text(
                0.5,
                0.5,
                str(e),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="red",
            )
        ax.set_title(p["title"], fontsize=11, pad=6)
        ax.set_xlabel(p["subtitle"], fontsize=10, labelpad=4)
        ax.axis("off")

    plt.tight_layout()
    path_a = out / f"dominant_var_panel_{args.period}.png"
    fig.savefig(path_a, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_a}")
    plt.close()

    # ------------------------------------------------------------------ #
    # Figure B: same variable (v10) across models — how attribution       #
    # changes when ocean info is added                                     #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"IG attribution for v10 across variable sets  ({args.period})\n"
        "Same variable — does adding ocean info change where the model looks?",
        fontsize=13,
        y=1.02,
    )

    subtitles = {
        "noSST_layers2": "r=0.683",
        "SST_layers2": "r=0.864",
        "deepSST_layers2": "r=0.717",
    }
    for ax, p in zip(axes, ATMOS_PANELS):
        try:
            img = load_img(p["model"], p["var"], args.period)
            ax.imshow(img)
        except FileNotFoundError as e:
            ax.text(
                0.5,
                0.5,
                str(e),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="red",
            )
        ax.set_title(f"{p['label']}\n{subtitles[p['model']]}", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    path_b = out / f"v10_across_models_{args.period}.png"
    fig.savefig(path_b, dpi=150, bbox_inches="tight")
    print(f"Saved: {path_b}")
    plt.close()


if __name__ == "__main__":
    main()
