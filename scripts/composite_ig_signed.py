"""
composite_ig_signed.py — Signed Integrated Gradients composite.

Unlike composite_ig.py (which uses |IG|), this keeps the sign:
  positive attribution → input region pushed prediction higher (toward MHW)
  negative attribution → input region suppressed prediction

Separate spatial maps per variable for MHW and non-MHW composites,
plus a MHW-minus-nonMHW difference map to highlight precursor patterns.

Usage:
  python eval/composite_ig_signed.py \
      --checkpoint <ckpt> --config <yaml> --output_dir <dir>
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
from xai.integrated_gradients import _integrated_gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--n_samples",   type=int, default=200)
    parser.add_argument("--threshold",   type=float, default=0.0)
    parser.add_argument("--no_cuda",     action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    config    = load_config(args.config)
    variables = config["variables"]
    thr       = args.threshold

    # Load model
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden",   256),
        lstm_layers=config.get("lstm_layers",     2),
        temporal_features=config.get("temporal_features", 0),
        dropout=config.get("dropout", 0.3),
        arch=config.get("arch", "lstm_attention"),
        gaussian_nll=config.get("gaussian_nll", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        args.checkpoint, model=cnn_lstm, map_location=device,
    )
    lm.eval().to(device)
    print(f"Loaded: {args.checkpoint}")

    dm = LazyDataModule(config_path=args.config)
    dm.setup()
    full_ds = dm.train_dataset.dataset

    ds_nc = xr.open_dataset(config["data_dir"])
    lat, lon = ds_nc.lat.values, ds_nc.lon.values
    ds_nc.close()

    n_vars      = len(variables)
    window_size = config.get("window_size", 60)
    days        = np.arange(-window_size + 1, 1)
    extent      = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Quick inference pass
    print("Collecting truth labels...")
    preds, trues = [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            trues.append(y.item())
            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(full_ds)}", end="\r")
    print()
    preds = np.array(preds)
    trues = np.array(trues)

    mhw_idx   = np.where(trues > thr)[0]
    nomhw_idx = np.where(trues <= thr)[0]
    print(f"MHW: {len(mhw_idx)}   non-MHW: {len(nomhw_idx)}")

    rng = np.random.default_rng(42)

    def select_separated(candidates, n, sep=60):
        selected = []
        for c in candidates:
            if all(abs(c - s) >= sep for s in selected):
                selected.append(c)
            if len(selected) == n:
                break
        return np.array(selected)

    sep = window_size
    mhw_sorted   = mhw_idx[np.argsort(preds[mhw_idx])[::-1]]
    mhw_sample   = select_separated(mhw_sorted, args.n_samples, sep)
    nomhw_sample = select_separated(
        rng.choice(nomhw_idx, size=min(len(nomhw_idx), 5 * args.n_samples), replace=False),
        args.n_samples, sep
    )
    print(f"Selected: MHW={len(mhw_sample)}  non-MHW={len(nomhw_sample)}")

    land_mask_np = full_ds.tierra_mask.numpy()
    coast_mask   = binary_dilation(land_mask_np, iterations=2)

    groups = {
        "MHW":    mhw_sample,
        "nonMHW": nomhw_sample,
    }

    # Accumulators — signed (no abs)
    spatial_signed  = {}
    temporal_signed = {}
    global_signed   = {}

    for gname, idxs in groups.items():
        print(f"\nSigned IG for {gname} (n={len(idxs)})")
        sp_acc  = np.zeros((n_vars, len(lat), len(lon)))
        tmp_acc = np.zeros((window_size, n_vars))
        gl_acc  = np.zeros(n_vars)

        for k, idx in enumerate(idxs):
            xs, xt, _ = full_ds[idx]
            attrs = _integrated_gradients(
                lm,
                xs.unsqueeze(0).to(device),
                xt.unsqueeze(0).to(device),
            )
            # Keep sign — mean over time for spatial, mean over space for temporal
            sp_acc  += attrs.mean(dim=0).numpy()           # (C, H, W)
            tmp_acc += attrs.mean(dim=(2, 3)).numpy()      # (T, C)
            gl_acc  += attrs.mean(dim=(0, 2, 3)).numpy()   # (C,)
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(idxs)}", end="\r")
        print()

        spatial_signed[gname]  = sp_acc  / len(idxs)
        temporal_signed[gname] = tmp_acc / len(idxs)
        global_signed[gname]   = gl_acc  / len(idxs)

    # --- Plot 1: Bar chart signed global attribution ---
    fig, ax = plt.subplots(figsize=(9, 4))
    x     = np.arange(n_vars)
    width = 0.35
    colors = {"MHW": "#d6604d", "nonMHW": "#4393c3"}
    for i, (gname, vals) in enumerate(global_signed.items()):
        ax.bar(x + i * width, vals, width, label=gname,
               color=colors[gname], alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variables, rotation=15)
    ax.set_ylabel("Mean signed IG attribution")
    ax.set_title("Signed variable importance: MHW vs non-MHW")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "signed_ig_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: signed_ig_importance.png")

    # --- Plot 2: Spatial maps per variable ---
    sp_mhw    = spatial_signed["MHW"]
    sp_nomhw  = spatial_signed["nonMHW"]
    sp_diff   = sp_mhw - sp_nomhw

    for i, var in enumerate(variables):
        m0   = np.where(coast_mask, np.nan, sp_mhw[i])
        m1   = np.where(coast_mask, np.nan, sp_nomhw[i])
        diff = np.where(coast_mask, np.nan, sp_diff[i])

        vmax_single = np.nanmax(np.abs([m0, m1]))
        vmax_diff   = np.nanmax(np.abs(diff))

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, data, title, vmax in [
            (axes[0], m0,   f"MHW composite  (n={len(mhw_sample)})",    vmax_single),
            (axes[1], m1,   f"non-MHW composite  (n={len(nomhw_sample)})", vmax_single),
            (axes[2], diff, "Difference  (MHW − non-MHW)",               vmax_diff),
        ]:
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("lon"); ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, fraction=0.03, label="signed attr")

        fig.suptitle(f"Signed IG spatial composite — {var}", fontsize=12)
        plt.tight_layout()
        fig.savefig(output_dir / f"signed_ig_spatial_{var}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: signed_ig_spatial_{var}.png")

    # --- Plot 3: Temporal profiles signed ---
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharey=False)
    if n_vars == 1:
        axes = [axes]

    for j, (ax, var) in enumerate(zip(axes, variables)):
        for gname, tmp in temporal_signed.items():
            ax.plot(days, tmp[:, j], label=gname,
                    lw=1.8, color=colors[gname], alpha=0.9)
        ax.axhline(0, color="k", ls="-", lw=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean signed IG", fontsize=9)
        ax.grid(alpha=0.3)
        if j == n_vars - 1:
            ax.set_xlabel("Day relative to last input day")

    axes[0].legend(fontsize=9)
    fig.suptitle("Signed temporal attribution: MHW vs non-MHW", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "signed_ig_temporal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: signed_ig_temporal.png")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
