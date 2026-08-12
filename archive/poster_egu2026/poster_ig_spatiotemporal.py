#!/usr/bin/env python
"""
Spatial IG maps split by temporal position within the 60-day window.
For each time bin (early / mid / recent), shows WHERE each variable matters.
Primary output: T_bottom 3-panel map to test whether Gulf Stream appears earlier.

Usage:
  python eval/poster_ig_spatiotemporal.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_ig_spatiotemporal.png \
    --n_ig     50
"""

import argparse
import sys

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel

from xai.integrated_gradients import _integrated_gradients
from xai.utils import load_config

FONTSIZE = 18
TITLESIZE = 20

VAR_LABELS = {
    "ptho_bot": "T$_{bottom}$",
    "to_anom": "SST anom.",
    "u10": "U-wind",
    "v10": "V-wind",
    "msl": "SLP",
    "ssr": "Solar rad.",
}

# Bins defined relative to window start (day 0 = oldest, day 59 = t-7)
BINS = [
    ("Early\n(day −60 to −41)", slice(0, 20)),
    ("Mid\n(day −40 to −21)", slice(20, 40)),
    ("Recent\n(day −20 to −1)", slice(40, 60)),
]


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
    """Select top-n predictions separated by at least `sep` samples."""
    order = np.argsort(preds)[::-1]
    sel = []
    for i in order:
        if all(abs(i - j) >= sep for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return np.array(sel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", default="poster_figures/fig_ig_spatiotemporal.png")
    parser.add_argument("--n_ig", type=int, default=50)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0 = load_config(str(exp_dirs[0] / "config.yaml"))
    variables = config0["variables"]
    n_vars = len(variables)

    ds_xr = xr.open_dataset(config0["data_dir"])
    lat = ds_xr.lat.values
    lon = ds_xr.lon.values
    ds_xr.close()

    d = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    idx_events = sep_indices(ens_pred, args.n_ig, sep=60)
    print(f"Selected {len(idx_events)} top events")

    # Accumulate: (n_bins, n_vars, lat, lon)
    n_bins = len(BINS)
    n_lat, n_lon = len(lat), len(lon)
    accum = np.zeros((n_bins, n_vars, n_lat, n_lon))
    count = 0

    for exp_dir in exp_dirs:
        config = load_config(str(exp_dir / "config.yaml"))
        lm = load_model(best_checkpoint(exp_dir), config, device)
        dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed = config.get("seed", "?")
        print(f"  Seed={seed} IG...")

        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[int(idx)]
            attrs = _integrated_gradients(
                lm,
                xs.unsqueeze(0).to(device),
                xt.unsqueeze(0).to(device),
            )  # (window, n_vars, lat, lon)
            abs_attrs = attrs.abs().numpy()  # (60, n_vars, lat, lon)

            for b, (_, sl) in enumerate(BINS):
                accum[b] += abs_attrs[sl].mean(axis=0)  # (n_vars, lat, lon)

            count += 1
            print(f"    {k+1}/{len(idx_events)}", end="\r")
        print()

    accum /= count  # (n_bins, n_vars, lat, lon)

    # --- Figure: rows = variables, cols = time bins ---
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    fig, axes = plt.subplots(
        n_vars, n_bins, figsize=(5 * n_bins, 4 * n_vars), squeeze=False
    )

    for i, var in enumerate(variables):
        # Normalize each variable globally across all bins so bins are comparable
        vmax = accum[:, i].max()
        vmax = vmax if vmax > 0 else 1.0

        for b, (bin_name, _) in enumerate(BINS):
            ax = axes[i][b]
            data = accum[b, i].copy()
            im = ax.imshow(
                data / vmax,
                origin="lower",
                extent=extent,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            if i == 0:
                ax.set_title(bin_name, fontsize=FONTSIZE, fontweight="bold", pad=8)
            if b == 0:
                ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=FONTSIZE)
                ax.tick_params(labelsize=FONTSIZE - 4)
            else:
                ax.set_yticks([])
            if i == n_vars - 1:
                ax.set_xlabel("Longitude", fontsize=FONTSIZE - 2)
                ax.tick_params(axis="x", labelsize=FONTSIZE - 4)
            else:
                ax.set_xticks([])
            if b == n_bins - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Norm. |IG|", fontsize=FONTSIZE - 4)
                cbar.ax.tick_params(labelsize=FONTSIZE - 6)

    n_seeds = len(exp_dirs)
    fig.suptitle(
        f"Spatial IG attribution by window position — Integrated Gradients\n"
        f"Ensemble of {n_seeds} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
        fontsize=TITLESIZE,
        fontweight="bold",
        y=1.01,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # --- Also save T_bottom-only figure ---
    if "ptho_bot" in variables:
        vi = variables.index("ptho_bot")
        vmax = accum[:, vi].max()
        vmax = vmax if vmax > 0 else 1.0

        fig2, axes2 = plt.subplots(1, n_bins, figsize=(6 * n_bins, 5))
        for b, (bin_name, _) in enumerate(BINS):
            ax = axes2[b]
            data = accum[b, vi].copy()
            im = ax.imshow(
                data / vmax,
                origin="lower",
                extent=extent,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            ax.set_title(bin_name, fontsize=FONTSIZE, fontweight="bold", pad=8)
            ax.tick_params(labelsize=FONTSIZE - 4)
            if b > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel("Latitude", fontsize=FONTSIZE - 2)
            ax.set_xlabel("Longitude", fontsize=FONTSIZE - 2)
            if b == n_bins - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Norm. |IG|", fontsize=FONTSIZE - 4)
                cbar.ax.tick_params(labelsize=FONTSIZE - 6)

        fig2.suptitle(
            f"T$_{{bottom}}$ spatial attribution by window position\n"
            f"Ensemble of {n_seeds} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
            fontsize=TITLESIZE,
            fontweight="bold",
        )
        out2 = out.parent / (out.stem + "_pthobot.png")
        plt.tight_layout()
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
