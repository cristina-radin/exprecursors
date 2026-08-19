"""
ig_simple.py — Integrated Gradients spatial map, no grouping.

Computes mean signed IG attribution over all test samples.
One map per input variable. Output: ig_simple_<var>.png + ig_simple_all.npy

Usage:
  python eval/ig_simple.py --checkpoint <ckpt> --config <yaml> --output <dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import MHWDataset
from src.models.cnn_lstm import CNNLSTMModel
from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)


def integrated_gradients(model, x_spatial, x_temporal, steps=50):
    baseline_s = torch.zeros_like(x_spatial)
    baseline_t = torch.zeros_like(x_temporal)

    alphas = torch.linspace(0, 1, steps, device=x_spatial.device).view(-1, 1, 1, 1, 1)
    interp_s = baseline_s + alphas * (x_spatial - baseline_s)
    interp_t = baseline_t.unsqueeze(0).expand(steps, -1, -1, -1) + torch.linspace(
        0, 1, steps, device=x_temporal.device
    ).view(-1, 1, 1, 1) * (x_temporal - baseline_t).unsqueeze(0)

    B = x_spatial.shape[0]
    interp_s = interp_s.view(steps * B, *x_spatial.shape[1:]).requires_grad_(True)
    interp_t = interp_t.view(steps * B, *x_temporal.shape[1:])

    out = model(interp_s, interp_t)
    if isinstance(out, tuple):
        out = out[0]
    out = out.mean()
    out.backward()

    grads = interp_s.grad.view(steps, B, *x_spatial.shape[1:])
    ig = ((x_spatial - baseline_s) * grads.mean(0)).detach().cpu()
    return ig  # (B, T, C, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_steps", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    ds = MHWDataset(args.config)
    total = len(ds)
    target_years = np.array(
        [int(ds.years[i + ds.window_size - 1]) for i in range(total)]
    )
    unique_years = np.sort(np.unique(target_years))
    n_folds = cfg.get("n_folds", 5)
    fold = cfg.get("fold", 0)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(unique_years)
    fold_groups = np.array_split(shuffled, n_folds)
    test_years = set(fold_groups[fold].tolist())
    test_idx = [i for i in range(total) if target_years[i] in test_years]
    print(f"Test samples: {len(test_idx)}")

    # Model
    arch = cfg.get("arch", "lstm_attention")
    gnll = cfg.get("gaussian_nll", False)
    tf = cfg.get("temporal_features", 0)
    model = CNNLSTMModel(
        in_channels=len(cfg["variables"]),
        arch=arch,
        gaussian_nll=gnll,
        temporal_features=tf,
        pooling=cfg.get("pooling", "max"),
        padding_mode=cfg.get("padding_mode", "zeros"),
        quantile_head=cfg.get("quantile_head", False),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Accumulate IG over all test samples (batch=1 to save memory)
    variables = cfg["variables"]
    n_vars = len(variables)
    H, W = ds[0][0].shape[-2], ds[0][0].shape[-1]

    sum_ig = torch.zeros(n_vars, H, W)  # mean over time and samples
    count = 0

    for idx in test_idx:
        xs, xt, _ = ds[idx]
        xs = xs.unsqueeze(0).float().to(device)  # (1, T, C, H, W)
        xt = xt.unsqueeze(0).float().to(device)  # (1, T, tf)

        ig = integrated_gradients(model, xs, xt, steps=args.n_steps)  # (1, T, C, H, W)
        # Mean over time dimension → (C, H, W)
        sum_ig += ig[0].mean(0).cpu()
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count}/{len(test_idx)}")

    mean_ig = (sum_ig / count).numpy()  # (C, H, W)
    np.save(out_dir / "ig_simple_all.npy", mean_ig)
    print(f"Saved ig_simple_all.npy  shape={mean_ig.shape}")

    # Load grid for plotting
    import xarray as xr

    nc_file = str(DATA_FILE_ENV)
    nc = xr.open_dataset(nc_file)
    lats = nc.lat.values
    lons = nc.lon.values
    nc.close()

    # Plot one map per variable
    for i, var in enumerate(variables):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        data = mean_ig[i]
        vmax = np.percentile(np.abs(data), 98)
        im = ax.pcolormesh(
            lons,
            lats,
            data,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_title(f"Mean signed IG — {var}  (n={count} test samples)", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Mean signed IG")
        plt.tight_layout()
        fname = out_dir / f"ig_simple_{var}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {fname}")

    print(f"\nDone. All maps in {out_dir}")


if __name__ == "__main__":
    main()
