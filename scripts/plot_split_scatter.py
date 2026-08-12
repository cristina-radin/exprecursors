#!/usr/bin/env python
"""
Three-panel scatter plot: predicted vs observed to_anom for each split strategy.
Highlights how split choice inflates or deflates apparent model skill.

Usage:
  python eval/plot_split_scatter.py --output_dir eval_results/split_scatter
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

SPLITS = {
    "Random\n(r ≈ 0.98, invalid)": {
        "config": BASE / "split_random/SST_layers2/config.yaml",
        "checkpoint": BASE
        / "split_random/SST_layers2/outputs/checkpoints/cnn-lstm-epoch=207-val_loss=0.0492.ckpt",
        "color": "#d62728",
    },
    "Block-year\n(r = 0.864)": {
        "config": BASE / "split_blockyear/SST_layers2/config_blockyear_baseline.yaml",
        "checkpoint": BASE
        / "split_blockyear/SST_layers2/checkpoints/cnn-lstm-epoch=12-val_loss=0.5751.ckpt",
        "color": "#2ca02c",
    },
    "Temporal\n(r ≈ 0.43)": {
        "config": BASE / "split_temporal/SST_layers2/config_temporal_baseline.yaml",
        "checkpoint": BASE
        / "split_temporal/SST_layers2/checkpoints/cnn-lstm-epoch=01-val_loss=0.3394.ckpt",
        "color": "#1f77b4",
    },
}


def load_model(config, ckpt_path, device):
    cfg = load_config(str(config))
    cnn_lstm = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 128),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        temporal_features=3,
        dropout=cfg.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)
    return lm, cfg


def run_inference(lm, config_path, device):
    dm = LazyDataModule(config_path=str(config_path))
    dm.setup()

    # Use only test set indices
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
    parser.add_argument("--output_dir", default="eval_results/split_scatter")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Impact of split strategy on apparent model skill\n(SST, layers=2, lead=7d)",
        fontsize=13,
        y=1.01,
    )

    for ax, (label, spec) in zip(axes, SPLITS.items()):
        print(f"\n--- {label.replace(chr(10), ' ')} ---")
        lm, cfg = load_model(spec["config"], spec["checkpoint"], device)
        preds, trues = run_inference(lm, spec["config"], device)
        r = pearson_r(preds, trues)
        print(f"  r={r:.3f}  n={len(preds)}")

        ax.scatter(trues, preds, s=3, alpha=0.3, color=spec["color"], rasterized=True)

        vmin = min(trues.min(), preds.min())
        vmax = max(trues.max(), preds.max())
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, label="1:1")

        ax.set_xlabel("Observed to_anom (°C)", fontsize=11)
        ax.set_ylabel("Predicted to_anom (°C)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.annotate(
            f"r = {r:.3f}\nn = {len(preds):,}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=11,
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    out_path = out / "split_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
