#!/usr/bin/env python
"""
Perturbation-based temporal importance analysis.
Avoids LSTM vanishing-gradient artifacts of IG by directly masking
each temporal bin and measuring prediction degradation.

For each top MHW event and each seed:
  1. Get full prediction pred_full
  2. For each bin (early/mid/recent) × each variable: zero out that
     (bin, variable) slice and get pred_masked
  3. Degradation = pred_full - pred_masked  (positive = that bin/var helped)

Outputs:
  fig_perturb_temporal.png        — grouped bar chart: degradation per variable per bin
  fig_perturb_temporal_tbot.png   — line plot: T_bottom degradation per individual window day

Usage:
  python eval/poster_perturb_temporal.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_perturb_temporal.png \
    --n_events 50
"""

import argparse
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel

from xai.utils import load_config

FONTSIZE = 18
TITLESIZE = 20
TICKSIZE = 14

VAR_LABELS = {
    "ptho_bot": "T$_{bottom}$",
    "to_anom": "SST anom.",
    "u10": "U-wind",
    "v10": "V-wind",
    "msl": "SLP",
    "ssr": "Solar rad.",
}

BINS = [
    ("Early\n(day −60–41)", slice(0, 20)),
    ("Mid\n(day −40–21)", slice(20, 40)),
    ("Recent\n(day −20–1)", slice(40, 60)),
]
BIN_COLORS = ["#2166ac", "#92c5de", "#d7191c"]


def best_checkpoint(exp_dir):
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))

    def val_loss(p):
        try:
            return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
        except:
            return float("inf")

    return min(ckpts, key=val_loss)


def load_model(ckpt_path, config, device):
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden", 256),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 3),
        dropout=config.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)
    return lm


def sep_indices(preds, n, sep=60):
    order = np.argsort(preds)[::-1]
    sel = []
    for i in order:
        if all(abs(i - j) >= sep for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return np.array(sel)


@torch.no_grad()
def predict(lm, xs, xt, device):
    pred, _ = lm.model.forward_with_attention(
        xs.unsqueeze(0).float().to(device),
        xt.unsqueeze(0).float().to(device),
    )
    return pred.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", default="poster_figures/fig_perturb_temporal.png")
    parser.add_argument("--n_events", type=int, default=50)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0 = load_config(str(exp_dirs[0] / "config.yaml"))
    variables = config0["variables"]
    n_vars = len(variables)
    window = config0.get("window_size", 60)

    d = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    idx_events = sep_indices(ens_pred, args.n_events, sep=60)
    print(f"Selected {len(idx_events)} top events")

    # degradation_bins[b, v] = mean(pred_full - pred_masked) over events × seeds
    degradation_bins = np.zeros((len(BINS), n_vars))
    # degradation_days[t, v] = mean degradation when day t of variable v is zeroed
    degradation_days = np.zeros((window, n_vars))
    counts = 0

    for exp_dir in exp_dirs:
        config = load_config(str(exp_dir / "config.yaml"))
        lm = load_model(best_checkpoint(exp_dir), config, device)
        dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed = config.get("seed", "?")
        print(f"  Seed={seed}...")

        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[int(idx)]
            pred_full = predict(lm, xs, xt, device)

            # --- Bin perturbation ---
            for b, (_, sl) in enumerate(BINS):
                for v in range(n_vars):
                    xs_masked = xs.clone()
                    xs_masked[sl, v, :, :] = 0.0
                    pred_m = predict(lm, xs_masked, xt, device)
                    degradation_bins[b, v] += pred_full - pred_m

            # --- Per-day perturbation for all variables ---
            for t in range(window):
                for v in range(n_vars):
                    xs_masked = xs.clone()
                    xs_masked[t, v, :, :] = 0.0
                    pred_m = predict(lm, xs_masked, xt, device)
                    degradation_days[t, v] += pred_full - pred_m

            counts += 1
            print(f"    {k+1}/{len(idx_events)}", end="\r")
        print()

    degradation_bins /= counts
    degradation_days /= counts

    # ------------------------------------------------------------------ #
    # Figure 1: grouped bar chart (bins × variables)
    # ------------------------------------------------------------------ #
    x = np.arange(n_vars)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))

    for b, (bin_name, _) in enumerate(BINS):
        offset = (b - 1) * width
        ax.bar(
            x + offset,
            degradation_bins[b],
            width,
            label=bin_name.replace("\n", " "),
            color=BIN_COLORS[b],
            alpha=0.85,
        )

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([VAR_LABELS.get(v, v) for v in variables], fontsize=TICKSIZE)
    ax.set_ylabel("Mean prediction drop when bin is masked", fontsize=FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICKSIZE)
    ax.legend(fontsize=FONTSIZE - 2, title="Window bin", title_fontsize=FONTSIZE - 4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Temporal perturbation importance — prediction degradation\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
        fontsize=TITLESIZE,
        fontweight="bold",
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ------------------------------------------------------------------ #
    # Figure 2: per-day line plot for each variable
    # ------------------------------------------------------------------ #
    days = np.arange(-window, 0)  # -60 … -1 relative to last input day

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    colors_vars = ["#d73027", "#4575b4", "#74add1", "#f46d43", "#fee090"]
    for v, var in enumerate(variables):
        color = colors_vars[v % len(colors_vars)]
        ax2.plot(
            days,
            degradation_days[:, v],
            label=VAR_LABELS.get(var, var),
            color=color,
            linewidth=1.8,
            alpha=0.85,
        )

    # Shade the 3 bins
    bin_edges = [
        (-60, -41, BIN_COLORS[0], 0.08),
        (-40, -21, BIN_COLORS[1], 0.08),
        (-20, -1, BIN_COLORS[2], 0.08),
    ]
    for xmin, xmax, c, a in bin_edges:
        ax2.axvspan(xmin, xmax, color=c, alpha=a)

    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Day relative to last input day", fontsize=FONTSIZE)
    ax2.set_ylabel("Mean prediction drop", fontsize=FONTSIZE)
    ax2.tick_params(labelsize=TICKSIZE)
    ax2.legend(fontsize=FONTSIZE - 4, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_title(
        f"Per-day perturbation importance\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
        fontsize=TITLESIZE,
        fontweight="bold",
    )
    out2 = out.parent / (out.stem + "_perday.png")
    plt.tight_layout()
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # Print summary
    print("\nDegradation per bin per variable (mean pred drop):")
    print(f"{'':12}", end="")
    for name, _ in BINS:
        print(f"  {name.split(chr(10))[0]:>10}", end="")
    print()
    for v, var in enumerate(variables):
        print(f"  {var:12}", end="")
        for b in range(len(BINS)):
            print(f"  {degradation_bins[b, v]:+.5f}", end="")
        print()


if __name__ == "__main__":
    main()
