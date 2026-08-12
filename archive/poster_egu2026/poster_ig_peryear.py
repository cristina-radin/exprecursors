#!/usr/bin/env python
"""
Per-year IG variable importance heatmap.
For each year, selects top-N MHW events and computes mean |IG| per variable.
Output: heatmap (year x variable) showing how variable importance shifts over time.

Usage:
  python eval/poster_ig_peryear.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_ig_peryear.png \
    --n_per_year 5
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

from xai.integrated_gradients import _integrated_gradients
from xai.run_xai import top_indices_for_period
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
        temporal_features=3,
        dropout=config.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)
    return lm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", default="poster_figures/fig_ig_peryear.png")
    parser.add_argument("--n_per_year", type=int, default=5)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0 = load_config(str(exp_dirs[0] / "config.yaml"))
    variables = config0["variables"]
    n_vars = len(variables)

    d = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    years = d["years"]
    all_years = sorted(set(years.astype(int)))

    # Load models + datasets once
    models, datasets = [], []
    for exp_dir in exp_dirs:
        config = load_config(str(exp_dir / "config.yaml"))
        lm = load_model(best_checkpoint(exp_dir), config, device)
        dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        models.append(lm)
        datasets.append(dm.train_dataset.dataset)
        print(f"  Loaded seed={config.get('seed','?')}")

    # Per-year IG importance matrix
    year_importance = {}  # year → (n_vars,) normalized

    for yr in all_years:
        idx_yr = top_indices_for_period(
            ens_pred, years, yr, yr, args.n_per_year, "high", min_separation=30
        )
        if len(idx_yr) == 0:
            print(f"  {yr}: no events, skipping")
            continue

        accum = np.zeros(n_vars)
        count = 0
        for lm, full_ds in zip(models, datasets):
            for idx in idx_yr:
                xs, xt, _ = full_ds[idx]
                attrs = _integrated_gradients(
                    lm, xs.unsqueeze(0).to(device), xt.unsqueeze(0).to(device)
                )
                accum += attrs.abs().mean(dim=(0, 2, 3)).numpy()
                count += 1

        accum /= count
        # Normalize to % within year
        year_importance[yr] = accum / accum.sum() * 100
        print(
            f"  {yr}: {len(idx_yr)} events × {len(exp_dirs)} seeds  "
            + "  ".join(
                f"{v}={year_importance[yr][i]:.1f}%" for i, v in enumerate(variables)
            )
        )

    if not year_importance:
        print("No years with events found.")
        return

    # Build matrix (years × vars)
    yrs_with_data = sorted(year_importance.keys())
    mat = np.array([year_importance[y] for y in yrs_with_data])  # (n_years, n_vars)

    var_labels = [VAR_LABELS.get(v, v) for v in variables]

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(mat.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())

    ax.set_xticks(range(len(yrs_with_data)))
    ax.set_xticklabels(yrs_with_data, rotation=90, fontsize=TICKSIZE - 2)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(var_labels, fontsize=TICKSIZE)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% of total |IG|", fontsize=FONTSIZE - 2)
    cbar.ax.tick_params(labelsize=TICKSIZE - 2)

    ax.set_title(
        f"Variable importance per year — Integrated Gradients\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{args.n_per_year} events/year  ·  7-day lead",
        fontsize=TITLESIZE,
        fontweight="bold",
        pad=10,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # Also save npz for further analysis
    np.savez(
        out.parent / "ig_peryear.npz",
        years=np.array(yrs_with_data),
        importance=mat,
        variables=np.array(variables),
    )
    print(f"Saved: {out.parent / 'ig_peryear.npz'}")


if __name__ == "__main__":
    main()
