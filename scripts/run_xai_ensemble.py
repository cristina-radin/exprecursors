#!/usr/bin/env python
"""
Ensemble XAI: run Integrated Gradients on N seed models, average spatial maps.

For each period/season:
  1. Select top events by ENSEMBLE prediction (so all seeds agree on which events matter)
  2. Run IG on each seed model separately for those events
  3. Average IG maps across seeds → robust attribution

Usage:
  python xai/run_xai_ensemble.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 split_blockyear/TbotAtm_seed200 \
               split_blockyear/TbotAtm_seed300 split_blockyear/TbotAtm_seed400 \
    --output_dir split_blockyear/TbotAtm_ensemble/xai_results \
    --periods 1985-2004,2005-2014,2015-2024

  # Or by season:
  python xai/run_xai_ensemble.py ... --seasons DJF,MAM,JJA,SON
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
from scripts.run_xai import (
    SEASONS,
    collect_predictions,
    plot_ig_comparison,
    plot_ig_temporal,
    top_indices_for_period,
)
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.xai.integrated_gradients import _integrated_gradients
from src.xai.utils import load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def load_model(ckpt_path: Path, config: dict, device: str):
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
        str(ckpt_path),
        model=cnn_lstm,
        map_location=device,
    )
    lm.eval().to(device)
    return lm


def plot_ig_ensemble_spatial(
    spatial_mean,
    spatial_std,
    variables,
    ocean_vars,
    lat,
    lon,
    land_mask,
    label,
    output_dir,
):
    """One figure per variable: mean ± std IG map side by side."""
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    coast = binary_dilation(land_mask, iterations=2)

    for i, var in enumerate(variables):
        mean = spatial_mean[i].copy()
        std = spatial_std[i].copy()
        mean = np.where(coast, np.nan, mean)
        std = np.where(coast, np.nan, std)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        vmax = np.nanpercentile(mean, 98)

        im0 = axes[0].imshow(
            mean,
            origin="lower",
            extent=extent,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        axes[0].set_title(f"Mean |IG| — {var}\n{label}", fontsize=10)
        axes[0].set_xlabel("lon")
        axes[0].set_ylabel("lat")
        plt.colorbar(im0, ax=axes[0], label="|attribution|")

        im1 = axes[1].imshow(
            std, origin="lower", extent=extent, cmap="Blues", vmin=0, aspect="auto"
        )
        axes[1].set_title(f"Std |IG| across seeds — {var}", fontsize=10)
        axes[1].set_xlabel("lon")
        plt.colorbar(im1, ax=axes[1], label="std")

        plt.tight_layout()
        fname = output_dir / f"ig_ensemble_{var}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()


def plot_ig_panel(ig_mean_dict, variables, ocean_vars, lat, lon, land_mask, output_dir):
    """Multi-panel: one row per variable, one column per period/season."""
    periods = list(ig_mean_dict.keys())
    n_vars = len(variables)
    n_per = len(periods)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    coast = binary_dilation(land_mask, iterations=2)

    fig, axes = plt.subplots(n_vars, n_per, figsize=(5 * n_per, 3.5 * n_vars))
    if n_vars == 1:
        axes = [axes]
    if n_per == 1:
        axes = [[ax] for ax in axes]

    for i, var in enumerate(variables):
        maps = [ig_mean_dict[p][i] for p in periods]
        vmax = max(np.nanpercentile(np.where(coast, np.nan, m), 98) for m in maps)
        for j, (period, ax) in enumerate(zip(periods, axes[i])):
            data = maps[j].copy()
            data = np.where(coast, np.nan, data)
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap="YlOrRd",
                vmin=0,
                vmax=vmax,
                aspect="auto",
            )
            if i == 0:
                ax.set_title(period, fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(var, fontsize=9, fontweight="bold")
            else:
                ax.set_yticks([])
            ax.set_xticks([])
            plt.colorbar(im, ax=ax, fraction=0.03)

    fig.suptitle(
        "Ensemble IG spatial maps — mean |attribution| across seeds", fontsize=11
    )
    plt.tight_layout()
    fname = output_dir / "ig_ensemble_panel.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_ig", type=int, default=30)
    parser.add_argument("--periods", type=str, default="1985-2004,2005-2014,2015-2024")
    parser.add_argument("--seasons", type=str, default=None)
    parser.add_argument("--season_years", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    exp_dirs = [Path(d) for d in args.exp_dirs]

    # Parse periods
    if args.seasons is not None:
        sy0, sy1 = (
            args.season_years.split("-") if args.season_years else ("1985", "2024")
        )
        seasons_list = [s.strip() for s in args.seasons.split(",")]
        periods = [
            (int(sy0), int(sy1), s, SEASONS[s]) for s in seasons_list if s in SEASONS
        ]
    else:
        periods = []
        for p in args.periods.split(","):
            y0, y1 = p.strip().split("-")
            periods.append((int(y0), int(y1), f"{y0}–{y1}", None))

    # ---------------------------------------------------------------------------
    # Load all models and datasets
    # ---------------------------------------------------------------------------
    models = []
    datasets = []
    configs = []

    for exp_dir in exp_dirs:
        ckpt = best_checkpoint(exp_dir)
        config = load_config(str(exp_dir / "config.yaml"))
        seed = config.get("seed", "?")
        print(f"\n[seed={seed}] Loading {ckpt.name}")

        lm = load_model(ckpt, config, device)
        dm = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        full_ds = dm.train_dataset.dataset

        models.append(lm)
        datasets.append(full_ds)
        configs.append(config)

    # All seeds share the same data file, use first dataset for coords/mask
    config0 = configs[0]
    full_ds0 = datasets[0]
    ds_xr = xr.open_dataset(config0["data_dir"])
    lat = ds_xr.lat.values
    lon = ds_xr.lon.values
    ds_xr.close()

    variables = config0["variables"]
    ocean_vars = set(config0.get("ocean_variables", variables))
    land_mask = full_ds0.is_land.numpy()

    # ---------------------------------------------------------------------------
    # Collect ensemble predictions (average across all seeds)
    # ---------------------------------------------------------------------------
    print("\nCollecting predictions from all seeds...")
    all_preds = []
    for i, (lm, full_ds) in enumerate(zip(models, datasets)):
        seed = configs[i].get("seed", i)
        print(f"  seed={seed}...")
        preds_i, trues_i, years_i, months_i = collect_predictions(lm, full_ds, device)
        all_preds.append(preds_i)

    ens_preds = np.stack(all_preds).mean(axis=0)  # (N,)
    years = years_i
    months = months_i
    print(f"Ensemble preds ready. N={len(ens_preds)}")

    # ---------------------------------------------------------------------------
    # Per-period ensemble XAI
    # ---------------------------------------------------------------------------
    sep = config0.get("window_size", 60)

    ig_mean_dict = {}  # period_label → (n_vars, lat, lon) mean across seeds
    ig_temporal_dict = {}  # period_label → (window, n_vars) mean across seeds
    ig_import_dict = {}  # period_label → {var: importance}

    for y0, y1, label, months_filter in periods:
        print(f"\n{'='*50}")
        print(f"Period: {label}")
        period_dir = output_dir / label.replace("–", "-")
        period_dir.mkdir(exist_ok=True)

        # Select top events using ENSEMBLE prediction
        idx_events = top_indices_for_period(
            ens_preds,
            years,
            y0,
            y1,
            args.n_ig,
            "high",
            sep,
            months=months,
            months_filter=months_filter,
        )
        if len(idx_events) == 0:
            print(f"  No events in period {label}, skipping.")
            continue
        print(f"  {len(idx_events)} events selected (ensemble top-{args.n_ig})")

        # Run IG on each seed for these events, accumulate
        spatial_per_seed = []  # list of (n_vars, lat, lon) arrays, one per seed
        temporal_per_seed = []  # list of (window, n_vars), one per seed

        for s_idx, (lm, full_ds) in enumerate(zip(models, datasets)):
            seed = configs[s_idx].get("seed", s_idx)
            print(f"  IG seed={seed}...")
            spatial_accum = np.zeros((len(variables), len(lat), len(lon)))
            temporal_accum = np.zeros((config0.get("window_size", 60), len(variables)))

            for k, idx in enumerate(idx_events):
                xs, xt, _ = full_ds[idx]
                attrs = _integrated_gradients(
                    lm,
                    xs.unsqueeze(0).to(device),
                    xt.unsqueeze(0).to(device),
                )
                abs_a = attrs.abs()
                spatial_accum += abs_a.mean(dim=0).numpy()
                temporal_accum += abs_a.mean(dim=(2, 3)).numpy()
                print(f"    event {k+1}/{len(idx_events)}", end="\r")
            print()

            spatial_accum /= len(idx_events)
            temporal_accum /= len(idx_events)
            spatial_per_seed.append(spatial_accum)
            temporal_per_seed.append(temporal_accum)

        # Average and std across seeds
        spatial_stack = np.stack(spatial_per_seed)  # (n_seeds, n_vars, lat, lon)
        temporal_stack = np.stack(temporal_per_seed)  # (n_seeds, window, n_vars)

        spatial_mean = spatial_stack.mean(axis=0)
        spatial_std = spatial_stack.std(axis=0)
        temporal_mean = temporal_stack.mean(axis=0)

        ig_mean_dict[label] = spatial_mean
        ig_temporal_dict[label] = temporal_mean
        ig_import_dict[label] = dict(zip(variables, spatial_mean.mean(axis=(1, 2))))

        # Per-variable maps (mean ± std)
        plot_ig_ensemble_spatial(
            spatial_mean,
            spatial_std,
            variables,
            ocean_vars,
            lat,
            lon,
            land_mask,
            label,
            period_dir,
        )
        print(f"  Saved spatial maps for {label}")

    # ---------------------------------------------------------------------------
    # Comparison figures across periods
    # ---------------------------------------------------------------------------
    if len(ig_mean_dict) > 1:
        plot_ig_panel(
            ig_mean_dict, variables, ocean_vars, lat, lon, land_mask, output_dir
        )
        plot_ig_comparison(
            ig_import_dict,
            variables,
            ocean_vars,
            output_dir / "comparison_ig_importance.png",
        )
        print("Saved: comparison_ig_importance.png")
        plot_ig_temporal(
            ig_temporal_dict, variables, output_dir / "comparison_ig_temporal.png"
        )
        print("Saved: comparison_ig_temporal.png")

    print(f"\nAll ensemble XAI results in: {output_dir}")


if __name__ == "__main__":
    main()
