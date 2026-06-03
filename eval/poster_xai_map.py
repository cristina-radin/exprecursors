#!/usr/bin/env python
"""
Poster-ready XAI spatial map: one panel per variable, averaged over all events
and all seeds. No period split.

Usage:
  python eval/poster_xai_map.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --output   poster_figures/fig_xai_poster.png \
    --n_ig 50
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
from xai.run_xai import collect_predictions, top_indices_for_period

FONTSIZE  = 20
TITLESIZE = 22

VAR_LABELS = {
    "ptho_bot": "T$_{bottom}$",
    "to_anom":  "SST anomaly",
    "u10":      "U-wind (10m)",
    "v10":      "V-wind (10m)",
    "msl":      "Sea-level pressure",
    "ssr":      "Solar radiation",
}


def _plot(spatial_mean, lat, lon, variables, out):
    """Plot normalized IG spatial maps (shared 0-1 colorscale)."""
    n_vars = len(variables)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Normalize each variable to [0, 1]
    spatial_norm = np.copy(spatial_mean)
    for i in range(n_vars):
        vmax = np.nanmax(spatial_norm[i])
        if vmax > 0:
            spatial_norm[i] /= vmax

    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 5))
    if n_vars == 1:
        axes = [axes]

    for i, (ax, var) in enumerate(zip(axes, variables)):
        im = ax.imshow(spatial_norm[i], origin="lower", extent=extent,
                       cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        label = VAR_LABELS.get(var, var)
        ax.set_title(label, fontsize=TITLESIZE, fontweight="bold", pad=8)
        ax.tick_params(labelsize=FONTSIZE - 4)
        if i == 0:
            ax.set_ylabel("Latitude", fontsize=FONTSIZE - 2)
        else:
            ax.set_yticks([])
        ax.set_xlabel("Longitude", fontsize=FONTSIZE - 2)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if i == n_vars - 1:
            cbar.set_label("Norm. |IG| (0–1)", fontsize=FONTSIZE - 4)
        cbar.ax.tick_params(labelsize=FONTSIZE - 6)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


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
        str(ckpt_path), model=cnn_lstm, map_location=device,
    )
    lm.eval().to(device)
    return lm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs",   nargs="+", default=[])
    parser.add_argument("--output",     default="poster_figures/fig_xai_poster.png")
    parser.add_argument("--n_ig",       type=int, default=50)
    parser.add_argument("--year_start", type=int, default=None,
                        help="First year to include (default: all years)")
    parser.add_argument("--year_end",   type=int, default=None,
                        help="Last year to include (default: all years)")
    parser.add_argument("--no_cuda",    action="store_true")
    parser.add_argument("--from_npz",  default=None,
                        help="Skip IG computation, load spatial_mean from this npz")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs] if args.exp_dirs else []

    # --- Fast path: load precomputed spatial maps ---
    if args.from_npz:
        d            = np.load(args.from_npz, allow_pickle=True)
        spatial_mean = d["spatial_mean"]        # (n_vars, lat, lon)
        lat          = d["lat"]
        lon          = d["lon"]
        variables    = list(d["variables"])
        _plot(spatial_mean, lat, lon, variables, Path(args.output))
        return

    config0 = load_config(str(exp_dirs[0] / "config.yaml"))
    ds_xr   = xr.open_dataset(config0["data_dir"])
    lat     = ds_xr.lat.values
    lon     = ds_xr.lon.values
    ds_xr.close()
    variables = config0["variables"]

    # Load models and datasets
    models, datasets = [], []
    for exp_dir in exp_dirs:
        ckpt    = best_checkpoint(exp_dir)
        config  = load_config(str(exp_dir / "config.yaml"))
        lm      = load_model(ckpt, config, device)
        dm      = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        models.append(lm)
        datasets.append(dm.train_dataset.dataset)
        print(f"  Loaded seed={config.get('seed','?')}")

    full_ds0   = datasets[0]
    land_mask  = full_ds0.tierra_mask.numpy()
    coast_mask = binary_dilation(land_mask, iterations=2)

    # Ensemble predictions to rank events
    print("Collecting ensemble predictions...")
    all_preds = []
    for lm, full_ds in zip(models, datasets):
        preds, _, years, months = collect_predictions(lm, full_ds, device)
        all_preds.append(preds)
    ens_preds = np.stack(all_preds).mean(axis=0)

    # Top events — optionally filtered to a year window
    sep   = config0.get("window_size", 60)
    y_min = args.year_start if args.year_start is not None else int(years.min())
    y_max = args.year_end   if args.year_end   is not None else int(years.max())
    idx_events = top_indices_for_period(
        ens_preds, years, y_min, y_max, args.n_ig, "high", sep
    )
    print(f"Selected {len(idx_events)} top events ({y_min}–{y_max})")

    # Run IG on each seed, accumulate
    spatial_stack = []
    for s_idx, (lm, full_ds) in enumerate(zip(models, datasets)):
        seed = load_config(str(exp_dirs[s_idx] / "config.yaml")).get("seed", s_idx)
        print(f"  IG seed={seed}...")
        spatial_accum = np.zeros((len(variables), len(lat), len(lon)))
        for k, idx in enumerate(idx_events):
            xs, xt, _ = full_ds[idx]
            attrs = _integrated_gradients(lm,
                                          xs.unsqueeze(0).to(device),
                                          xt.unsqueeze(0).to(device))
            spatial_accum += attrs.abs().mean(dim=0).numpy()
            print(f"    {k+1}/{len(idx_events)}", end="\r")
        print()
        spatial_accum /= len(idx_events)
        spatial_stack.append(spatial_accum)

    spatial_stack_arr = np.stack(spatial_stack)            # (n_seeds, n_vars, lat, lon)
    spatial_mean = spatial_stack_arr.mean(axis=0)          # (n_vars, lat, lon)

    # Save per-variable means for barplot (tiny file, no recompute needed)
    var_means = spatial_stack_arr.mean(axis=(2, 3))        # (n_seeds, n_vars)
    out       = Path(args.output)
    npz_path  = out.parent / f"ig_var_importance_{out.stem.replace('fig_xai_','')}.npz"
    np.savez(npz_path,
             var_means=var_means,
             variables=np.array(variables))
    print(f"Saved: {npz_path}")

    # Mask land
    for i in range(len(variables)):
        spatial_mean[i] = np.where(coast_mask, np.nan, spatial_mean[i])

    # Save spatial maps so we can replot without recomputing
    out = Path(args.output)
    spatial_npz = out.parent / f"ig_spatial_mean_{out.stem.replace('fig_xai_','')}.npz"
    np.savez(spatial_npz,
             spatial_mean=spatial_mean,
             lat=lat, lon=lon,
             variables=np.array(variables))
    print(f"Saved: {spatial_npz}")

    _plot(spatial_mean, lat, lon, variables, out)


if __name__ == "__main__":
    main()
