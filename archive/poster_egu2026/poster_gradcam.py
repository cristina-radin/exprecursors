#!/usr/bin/env python
"""
Poster-ready ensemble Grad-CAM spatial map.
Averages attention-weighted Grad-CAM over top-N MHW events across all seeds.
Produces a single clean spatial map (one panel).

Usage:
  python eval/poster_gradcam.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_gradcam_poster.png \
    --n_events 50
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
from scipy.ndimage import binary_dilation

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel

from xai.grad_cam import AttentionGradCAM
from xai.utils import load_config

FONTSIZE = 20
TITLESIZE = 22


def best_checkpoint(exp_dir: Path) -> Path:
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {exp_dir}/checkpoints/")

    def val_loss(p):
        try:
            return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
        except Exception:
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
        str(ckpt_path),
        model=cnn_lstm,
        map_location=device,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", default="poster_figures/fig_gradcam_poster.png")
    parser.add_argument("--n_events", type=int, default=50)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    # Load lat/lon and land mask
    config0 = load_config(str(exp_dirs[0] / "config.yaml"))
    ds_xr = xr.open_dataset(config0["data_dir"])
    lat, lon = ds_xr.lat.values, ds_xr.lon.values
    ds_xr.close()

    # Select top events by ensemble prediction
    d = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    idx_events = sep_indices(ens_pred, args.n_events, sep=60)
    print(f"Selected {len(idx_events)} top events")

    # Load models + datasets, run Grad-CAM
    cam_stack = []
    for exp_dir in exp_dirs:
        config = load_config(str(exp_dir / "config.yaml"))
        ckpt = best_checkpoint(exp_dir)
        lm = load_model(ckpt, config, device)
        dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        seed = config.get("seed", "?")
        print(f"  Grad-CAM seed={seed}...")

        engine = AttentionGradCAM(lm)
        cam_accum = np.zeros((len(lat), len(lon)))

        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[idx]
            cam, _ = engine.compute(
                xs.unsqueeze(0).to(device),
                xt.unsqueeze(0).to(device),
            )
            cam_accum += cam
            print(f"    {k+1}/{len(idx_events)}", end="\r")
        print()

        cam_accum /= len(idx_events)
        cam_stack.append(cam_accum)

    cam_mean = np.stack(cam_stack).mean(axis=0)  # (lat, lon)
    cam_std = np.stack(cam_stack).std(axis=0)

    # Land mask
    dm0 = LazyDataModule(config_path=str(exp_dirs[0] / "config.yaml"))
    dm0.setup()
    land_mask = dm0.train_dataset.dataset.tierra_mask.numpy()
    coast_mask = binary_dilation(land_mask, iterations=2)
    cam_mean = np.where(coast_mask, np.nan, cam_mean)

    # --- Plot ---
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        cam_mean,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        vmin=0,
        vmax=np.nanpercentile(cam_mean, 98),
        aspect="auto",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Grad-CAM activation (norm.)", fontsize=FONTSIZE - 2)
    cbar.ax.tick_params(labelsize=FONTSIZE - 4)

    ax.set_xlabel("Longitude", fontsize=FONTSIZE)
    ax.set_ylabel("Latitude", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE - 4)

    ax.set_title(
        f"Spatial attribution — Grad-CAM\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{len(idx_events)} MHW events  ·  7-day lead",
        fontsize=TITLESIZE,
        fontweight="bold",
        pad=10,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
