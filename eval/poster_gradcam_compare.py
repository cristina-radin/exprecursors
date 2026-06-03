#!/usr/bin/env python
"""
Grad-CAM comparison: MHW vs non-MHW  +  per period.
Produces a 2-row figure:
  Row 1: MHW events | non-MHW events | difference
  Row 2: 1985-2004  | 2005-2014      | 2015-2024

Usage:
  python eval/poster_gradcam_compare.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_gradcam_compare.png
"""

import argparse
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_dilation
import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel
from xai.utils import load_config
from xai.grad_cam import AttentionGradCAM

FONTSIZE  = 14
TITLESIZE = 15
N_EVENTS  = 30


def best_checkpoint(exp_dir):
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))
    def val_loss(p):
        try: return float(str(p).split("val_loss=")[1].replace(".ckpt",""))
        except: return float("inf")
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
        str(ckpt_path), model=cnn_lstm, map_location=device,
    )
    lm.eval().to(device)
    return lm


def sep_indices(preds, n, sep=60, mode="high"):
    order = np.argsort(preds)[::-1] if mode == "high" else np.argsort(preds)
    sel = []
    for i in order:
        if all(abs(i - j) >= sep for j in sel):
            sel.append(i)
        if len(sel) == n:
            break
    return np.array(sel)


def compute_cam(models_datasets, idx_events, device):
    """Average Grad-CAM over given event indices across all seeds."""
    stack = []
    for lm, full_ds, lat, lon in models_datasets:
        engine    = AttentionGradCAM(lm)
        cam_accum = np.zeros((len(lat), len(lon)))
        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[idx]
            cam, _    = engine.compute(xs.unsqueeze(0).to(device),
                                       xt.unsqueeze(0).to(device))
            cam_accum += cam
        cam_accum /= max(len(idx_events), 1)
        stack.append(cam_accum)
    return np.stack(stack).mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--npz",      required=True)
    parser.add_argument("--output",   default="poster_figures/fig_gradcam_compare.png")
    parser.add_argument("--n_events", type=int, default=N_EVENTS)
    parser.add_argument("--no_cuda",  action="store_true")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0  = load_config(str(exp_dirs[0] / "config.yaml"))
    ds_xr    = xr.open_dataset(config0["data_dir"])
    lat, lon = ds_xr.lat.values, ds_xr.lon.values
    ds_xr.close()

    d        = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    years    = d["years"].astype(int)

    # Load models + datasets
    models_datasets = []
    for exp_dir in exp_dirs:
        config  = load_config(str(exp_dir / "config.yaml"))
        ckpt    = best_checkpoint(exp_dir)
        lm      = load_model(ckpt, config, device)
        dm      = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset
        models_datasets.append((lm, full_ds, lat, lon))
        print(f"  Loaded seed={config.get('seed','?')}")

    dm0       = LazyDataModule(config_path=str(exp_dirs[0] / "config.yaml"))
    dm0.setup()
    land_mask  = dm0.train_dataset.dataset.tierra_mask.numpy()
    coast_mask = binary_dilation(land_mask, iterations=2)

    def mask(cam): return np.where(coast_mask, np.nan, cam)

    # --- Event sets ---
    print("Computing Grad-CAM for MHW events...")
    idx_mhw    = sep_indices(ens_pred, args.n_events, mode="high")
    cam_mhw    = mask(compute_cam(models_datasets, idx_mhw, device))

    print("Computing Grad-CAM for non-MHW events...")
    idx_nomhw  = sep_indices(ens_pred, args.n_events, mode="low")
    cam_nomhw  = mask(compute_cam(models_datasets, idx_nomhw, device))

    periods = [(1985, 2004), (2005, 2014), (2015, 2024)]
    cam_periods = []
    for yr_start, yr_end in periods:
        print(f"Computing Grad-CAM {yr_start}–{yr_end}...")
        mask_yr = (years >= yr_start) & (years <= yr_end)
        preds_yr = np.where(mask_yr, ens_pred, -np.inf)
        idx_p    = sep_indices(preds_yr, args.n_events, mode="high")
        cam_periods.append(mask(compute_cam(models_datasets, idx_p, device)))

    # Normalize all maps together to [0,1] for shared colorscale
    all_maps  = [cam_mhw, cam_nomhw] + cam_periods
    global_max = np.nanmax([np.nanmax(m) for m in all_maps])
    all_maps  = [m / global_max for m in all_maps]
    cam_mhw, cam_nomhw = all_maps[0], all_maps[1]
    cam_periods = all_maps[2:]
    cam_diff    = cam_mhw - cam_nomhw   # signed difference

    # --- Plot ---
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    titles_row1 = [f"MHW events (top-{args.n_events})",
                   f"non-MHW events (bottom-{args.n_events})",
                   "Difference (MHW − non-MHW)"]
    titles_row2 = [f"{s}–{e}" for s, e in periods]

    # Row 1: MHW, non-MHW, difference
    for col, (cam, title) in enumerate(zip([cam_mhw, cam_nomhw, cam_diff], titles_row1)):
        ax = axes[0, col]
        if col < 2:
            im = ax.imshow(cam, origin="lower", extent=extent,
                           cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        else:
            vabs = np.nanmax(np.abs(cam))
            im   = ax.imshow(cam, origin="lower", extent=extent,
                             cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03).ax.tick_params(labelsize=FONTSIZE-3)
        ax.set_title(title, fontsize=TITLESIZE, fontweight="bold")
        ax.tick_params(labelsize=FONTSIZE - 3)
        if col == 0: ax.set_ylabel("Latitude", fontsize=FONTSIZE)
        else: ax.set_yticks([])

    # Row 2: per period
    for col, (cam, title) in enumerate(zip(cam_periods, titles_row2)):
        ax = axes[1, col]
        im = ax.imshow(cam, origin="lower", extent=extent,
                       cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03).ax.tick_params(labelsize=FONTSIZE-3)
        ax.set_title(title, fontsize=TITLESIZE, fontweight="bold")
        ax.set_xlabel("Longitude", fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE - 3)
        if col == 0: ax.set_ylabel("Latitude", fontsize=FONTSIZE)
        else: ax.set_yticks([])

    fig.suptitle("Grad-CAM spatial attribution — ensemble 4 seeds · 7-day lead",
                 fontsize=TITLESIZE + 1, fontweight="bold", y=0.98)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
