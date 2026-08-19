#!/usr/bin/env python
"""
Composite Integrated Gradients: MHW days vs non-MHW days.

Splits samples by truth label (to_anom > 0 = MHW) and computes mean IG
attributions for each group. Produces spatial maps and temporal profiles
showing what spatial/temporal patterns the model associates with MHW
vs non-MHW predictions.

Usage:
  python eval/composite_ig.py --checkpoint <ckpt> --config <yaml> --output_dir <dir>
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
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.xai.integrated_gradients import _integrated_gradients
from src.xai.utils import load_config

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples per group for IG (default=100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="to_anom threshold for MHW label (default=0)",
    )
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    config = load_config(args.config)
    variables = config["variables"]
    thr = args.threshold

    # --- Load model ---
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden", 256),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 3),
        dropout=config.get("dropout", 0.3),
        arch=config.get("arch", "lstm_only"),
        gaussian_nll=config.get("gaussian_nll", False),
        pooling=config.get("pooling", "max"),
        quantile_head=config.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        args.checkpoint,
        model=cnn_lstm,
        map_location=device,
    )
    lm.eval().to(device)
    print(f"Loaded: {args.checkpoint}")

    dm = LazyDataModule(config_path=args.config)
    dm.setup()
    full_ds = dm.train_dataset.dataset

    ds_nc = xr.open_dataset(config["data_dir"])
    lat, lon = ds_nc.lat.values, ds_nc.lon.values
    ds_nc.close()

    n_vars = len(variables)
    window_size = config.get("window_size", 60)
    days = np.arange(-window_size + 1, 1)

    # --- Quick inference pass to get truth values for sampling ---
    print("Quick inference pass to collect truth labels...")
    preds, trues, years = [], [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            trues.append(y.item())
            t = idx + full_ds.window_size - 1 + full_ds.lead_time
            years.append(int(full_ds.years[t]))
            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(full_ds)}", end="\r")
    print()
    preds = np.array(preds)
    trues = np.array(trues)
    years = np.array(years)

    # --- Sample indices for each group ---
    mhw_idx = np.where(trues > thr)[0]
    nomhw_idx = np.where(trues <= thr)[0]
    print(f"MHW samples:    {len(mhw_idx)}")
    print(f"non-MHW samples: {len(nomhw_idx)}")

    rng = np.random.default_rng(42)
    # For MHW: pick the top predicted ones (model most confident)
    # For non-MHW: pick random to be representative
    mhw_sorted = mhw_idx[np.argsort(preds[mhw_idx])[::-1]]

    # Ensure separation of 60 days
    def select_separated(candidates, n, sep=60):
        selected = []
        for c in candidates:
            if all(abs(c - s) >= sep for s in selected):
                selected.append(c)
            if len(selected) == n:
                break
        return np.array(selected)

    sep = config.get("window_size", 60)
    mhw_sample = select_separated(mhw_sorted, args.n_samples, sep)
    nomhw_sample = select_separated(
        rng.choice(
            nomhw_idx, size=min(len(nomhw_idx), 5 * args.n_samples), replace=False
        ),
        args.n_samples,
        sep,
    )
    print(f"Selected for IG: MHW={len(mhw_sample)}  non-MHW={len(nomhw_sample)}")

    # Land mask
    land_mask_np = full_ds.is_land.numpy()
    coast_mask = binary_dilation(land_mask_np, iterations=2)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # --- Run IG for each group ---
    groups = {
        "MHW (truth > 0)": mhw_sample,
        "non-MHW (truth ≤ 0)": nomhw_sample,
    }

    spatial_by_group = {}
    temporal_by_group = {}
    global_by_group = {}

    for group_name, idxs in groups.items():
        print(f"\nRunning IG for group: {group_name}  (n={len(idxs)})")
        spatial_accum = np.zeros((n_vars, len(lat), len(lon)))
        temporal_accum = np.zeros((window_size, n_vars))
        global_accum = np.zeros(n_vars)

        for k, idx in enumerate(idxs):
            xs, xt, _ = full_ds[idx]
            attrs = _integrated_gradients(
                lm,
                xs.unsqueeze(0).to(device),
                xt.unsqueeze(0).to(device),
            )
            abs_a = attrs.abs()
            spatial_accum += abs_a.mean(dim=0).numpy()
            global_accum += abs_a.mean(dim=(0, 2, 3)).numpy()
            temporal_accum += abs_a.mean(dim=(2, 3)).numpy()
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(idxs)}", end="\r")
        print()

        spatial_by_group[group_name] = spatial_accum / len(idxs)
        temporal_by_group[group_name] = temporal_accum / len(idxs)
        global_by_group[group_name] = global_accum / len(idxs)

    # --- Plot 1: Bar chart variable importance by group ---
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(n_vars)
    width = 0.35
    for i, (gname, vals) in enumerate(global_by_group.items()):
        ax.bar(x + i * width, vals, width, label=gname, alpha=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variables, rotation=15)
    ax.set_ylabel("Mean |IG attribution|")
    ax.set_title("Variable importance: MHW vs non-MHW days")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        output_dir / "composite_ig_importance.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved: composite_ig_importance.png")

    # --- Plot 2: Spatial maps per variable (MHW | non-MHW | difference) ---
    group_names = list(groups.keys())
    g0, g1 = group_names[0], group_names[1]
    for i, var in enumerate(variables):
        sp0 = spatial_by_group[g0][i].copy()
        sp1 = spatial_by_group[g1][i].copy()
        diff = sp0 - sp1

        sp0 = np.where(coast_mask, np.nan, sp0)
        sp1 = np.where(coast_mask, np.nan, sp1)
        diff = np.where(coast_mask, np.nan, diff)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        vmax_abs = max(np.nanmax(np.abs(sp0)), np.nanmax(np.abs(sp1)))
        vmax_diff = np.nanmax(np.abs(diff))

        for ax, data, title, cmap, vmin, vmax in [
            (axes[0], sp0, f"MHW  ({len(mhw_sample)} samples)", "YlOrRd", 0, vmax_abs),
            (
                axes[1],
                sp1,
                f"non-MHW  ({len(nomhw_sample)} samples)",
                "YlOrRd",
                0,
                vmax_abs,
            ),
            (
                axes[2],
                diff,
                "Difference (MHW minus non-MHW)",
                "RdBu_r",
                -vmax_diff,
                vmax_diff,
            ),
        ]:
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, fraction=0.03, label="|attr|")

        fig.suptitle(f"IG spatial composite — {var}", fontsize=12)
        plt.tight_layout()
        fig.savefig(
            output_dir / f"composite_ig_spatial_{var}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved: composite_ig_spatial_{var}.png")

    # --- Plot 3: Temporal profiles per variable ---
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharey=False)
    if n_vars == 1:
        axes = [axes]

    colors = {"MHW (truth > 0)": "tomato", "non-MHW (truth ≤ 0)": "steelblue"}
    for j, (ax, var) in enumerate(zip(axes, variables)):
        for gname, temporal in temporal_by_group.items():
            ax.plot(days, temporal[:, j], label=gname, lw=1.8, color=colors.get(gname))
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean |IG|", fontsize=9)
        ax.grid(alpha=0.3)
        if j == n_vars - 1:
            ax.set_xlabel("Day relative to last input day")

    axes[0].legend(fontsize=9)
    fig.suptitle("Temporal attribution: MHW vs non-MHW days", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "composite_ig_temporal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: composite_ig_temporal.png")

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
