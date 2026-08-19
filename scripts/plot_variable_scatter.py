#!/usr/bin/env python
"""
Two-panel figure comparing variable sets (noSST / SST / deepSST):
  Panel A: 3 scatter plots (predicted vs observed) — block-year split, layers=2
  Panel B: bar chart of skill metrics (r, ETS, CSI)

Usage:
  python eval/plot_variable_scatter.py --output_dir eval_results/variable_scatter
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
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.xai.utils import load_config

BASE = Path(__file__).parents[1]

MODELS = {
    "noSST\n(atmosphere only)": {
        "config": BASE / "split_blockyear/noSST_layers2/config_blockyear_noSST.yaml",
        "checkpoint": BASE
        / "split_blockyear/noSST_layers2/checkpoints/cnn-lstm-epoch=08-val_loss=0.8687.ckpt",
        "color": "#ff7f0e",
    },
    "SST\n(+ surface SST)": {
        "config": BASE / "split_blockyear/SST_layers2/config_blockyear_baseline.yaml",
        "checkpoint": BASE
        / "split_blockyear/SST_layers2/checkpoints/cnn-lstm-epoch=12-val_loss=0.5751.ckpt",
        "color": "#1f77b4",
    },
    "deepSST\n(+ bottom ocean temp)": {
        "config": BASE
        / "split_blockyear/deepSST_layers2/config_blockyear_deepSST.yaml",
        "checkpoint": BASE
        / "split_blockyear/deepSST_layers2/checkpoints/cnn-lstm-epoch=07-val_loss=0.5455.ckpt",
        "color": "#2ca02c",
    },
}

# Pre-computed skill scores (from eval_results)
SKILL = {
    "noSST": {"r": 0.683, "ETS": 0.337, "CSI": 0.649},
    "SST": {"r": 0.864, "ETS": 0.493, "CSI": 0.745},
    "deepSST": {"r": 0.879, "ETS": 0.552, "CSI": 0.767},
}


def load_model(config, ckpt_path, device):
    cfg = load_config(str(config))
    cnn_lstm = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 128),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        temporal_features=cfg.get("temporal_features", 3),
        dropout=cfg.get("dropout", 0.3),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
        pooling=cfg.get("pooling", "max"),
        quantile_head=cfg.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)
    return lm, cfg


def run_inference(lm, config_path, device):
    dm = LazyDataModule(config_path=str(config_path))
    dm.setup()
    test_indices = list(dm.test_dataset.indices)
    full_ds = dm.train_dataset.dataset

    preds, trues = [], []
    with torch.no_grad():
        for idx in test_indices:
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            trues.append(y.item())
    return np.array(preds), np.array(trues)


def pearson_r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="eval_results/variable_scatter")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # ------------------------------------------------------------------ #
    # Figure layout: 2 rows — top: scatter plots, bottom: bar chart       #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.4, wspace=0.35)

    scatter_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bar_ax = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "Impact of ocean variable choice on MHW prediction skill\n"
        "(block-year split, 2 LSTM layers, lead = 7 days)",
        fontsize=13,
        y=0.98,
    )

    # ---- Scatter plots ----
    for ax, (label, spec) in zip(scatter_axes, MODELS.items()):
        short = label.split("\n")[0]
        print(f"\n--- {short} ---")
        lm, cfg = load_model(spec["config"], spec["checkpoint"], device)
        preds, trues = run_inference(lm, spec["config"], device)
        r = pearson_r(preds, trues)
        print(f"  r={r:.3f}  n={len(preds)}")

        ax.scatter(trues, preds, s=3, alpha=0.3, color=spec["color"], rasterized=True)
        vmin = min(trues.min(), preds.min())
        vmax = max(trues.max(), preds.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)
        ax.set_xlabel("Observed to_anom (°C)", fontsize=10)
        ax.set_ylabel("Predicted to_anom (°C)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.annotate(
            f"r = {r:.3f}\nn = {len(preds):,}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.set_aspect("equal", adjustable="datalim")

    # ---- Bar chart ----
    model_names = list(SKILL.keys())
    metrics = ["r", "ETS", "CSI"]
    colors_bar = ["#ff7f0e", "#1f77b4", "#2ca02c"]
    x = np.arange(len(metrics))
    width = 0.22

    for i, (name, col) in enumerate(zip(model_names, colors_bar)):
        vals = [SKILL[name][m] for m in metrics]
        bars = bar_ax.bar(
            x + (i - 1) * width, vals, width, label=name, color=col, alpha=0.85
        )
        for bar, val in zip(bars, vals):
            bar_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(["Pearson r", "ETS", "CSI"], fontsize=11)
    bar_ax.set_ylim(0, 1.05)
    bar_ax.set_ylabel("Score", fontsize=11)
    bar_ax.set_title("Skill metric comparison (test set)", fontsize=11)
    bar_ax.legend(fontsize=10, loc="upper left")
    bar_ax.axhline(0, color="k", lw=0.5)

    out_path = out / "variable_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
