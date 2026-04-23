#!/usr/bin/env python
"""
XAI analysis for CNN-LSTM + Temporal Attention — period comparison.

For each defined time period, selects the top-N extreme events (by model
prediction) and runs Grad-CAM + Integrated Gradients. The goal is to compare
whether MHW precursor patterns are stable across decades or have shifted.

The full dataset is used (not restricted to train/val/test splits).
Normalisation stats are derived from the training split to match what the
model saw during training.

Usage:
  python xai/run_xai.py --checkpoint outputs/checkpoints/best.ckpt
"""

import argparse
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel
from xai.utils import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_predictions(lightning_module, full_ds, device):
    """
    Run inference on the full dataset.
    Returns arrays of shape (N,): pred, true, year, month
    """
    lightning_module.eval()
    preds, trues, years, months = [], [], [], []

    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lightning_module.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            trues.append(y.item())

            # Date of the predicted target (t + lead_time)
            t = idx + full_ds.window_size - 1 + full_ds.lead_time
            years.append(int(full_ds.years[t]))
            months.append(int(full_ds.months[t]))

            if (idx + 1) % 1000 == 0:
                print(f"  Predictions: {idx+1}/{len(full_ds)}", end="\r")

    print(f"  Predictions done: {len(full_ds)} samples")
    return (np.array(preds), np.array(trues),
            np.array(years), np.array(months))


SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def top_indices_for_period(preds, years, year_start, year_end, n,
                           extreme="high", min_separation=60,
                           months=None, months_filter=None):
    """
    Return up to n extreme samples within [year_start, year_end] with
    at least min_separation days between selected indices (avoids picking
    overlapping windows of the same event).

    min_separation should match window_size in config so that no two
    selected events share input days.

    months: array of month integers (1-12) aligned with preds/years.
    months_filter: list of ints (1-12) to restrict to specific months/season.
    """
    mask = (years >= year_start) & (years <= year_end)
    if months_filter is not None and months is not None:
        mask = mask & np.isin(months, list(months_filter))
    idx_in_period = np.where(mask)[0]
    if len(idx_in_period) == 0:
        return np.array([], dtype=int)

    order = np.argsort(preds[idx_in_period])
    if extreme == "high":
        candidates = idx_in_period[order[::-1]]   # highest first
    else:
        candidates = idx_in_period[order]          # lowest first

    # Greedy selection: keep event only if far enough from already-selected ones
    selected = []
    for idx in candidates:
        if all(abs(idx - s) >= min_separation for s in selected):
            selected.append(idx)
        if len(selected) == n:
            break

    return np.array(selected, dtype=int)


def plot_period_comparison(cam_dict, lat, lon, title, save_path):
    """
    Side-by-side mean Grad-CAM across periods.
    cam_dict: {period_label: mean_cam (lat, lon)}
    """
    periods = list(cam_dict.keys())
    n = len(periods)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    vmax = max(v.max() for v in cam_dict.values())
    for ax, period in zip(axes, periods):
        im = ax.imshow(cam_dict[period], origin="lower", extent=extent,
                       cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(period, fontsize=10)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, fraction=0.03)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ig_temporal(ig_temporal_dict, variables, save_path):
    """
    Temporal attribution profile per variable, one subplot per variable.
    ig_temporal_dict: {period_label: array (window, n_vars)}
    """
    periods  = list(ig_temporal_dict.keys())
    n_vars   = len(variables)
    window   = next(iter(ig_temporal_dict.values())).shape[0]
    days     = np.arange(-window + 1, 1)

    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharey=False)
    if n_vars == 1:
        axes = [axes]

    for j, (ax, var) in enumerate(zip(axes, variables)):
        for period in periods:
            profile = ig_temporal_dict[period][:, j]  # (window,)
            ax.plot(days, profile, label=period, linewidth=1.5)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean |IG|", fontsize=9)
        ax.grid(alpha=0.3)
        if j == n_vars - 1:
            ax.set_xlabel("Day relative to last input day")

    axes[-1].legend(fontsize=9)
    fig.suptitle("Temporal attribution profile per variable — Integrated Gradients", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ig_comparison(ig_dict, variables, ocean_vars, save_path):
    """Bar chart comparing variable importance across periods."""
    periods = list(ig_dict.keys())
    x = np.arange(len(variables))
    width = 0.8 / len(periods)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, period in enumerate(periods):
        vals = [ig_dict[period][v] for v in variables]
        ax.bar(x + i * width, vals, width, label=period, alpha=0.8)

    ax.set_xticks(x + width * (len(periods) - 1) / 2)
    ax.set_xticklabels(variables, rotation=15)
    ax.set_ylabel("Mean |attribution|")
    ax.set_title("Variable importance by period — Integrated Gradients")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attention_comparison(attn_dict, save_path):
    """Mean attention weight profiles per period."""
    window = len(next(iter(attn_dict.values()))[0])  # length of one attn array
    days   = np.arange(-window + 1, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for period, attn_list in attn_dict.items():
        mean_attn = np.stack(attn_list).mean(axis=0)
        std_attn  = np.stack(attn_list).std(axis=0)
        ax.plot(days, mean_attn, label=period)
        ax.fill_between(days, mean_attn - std_attn, mean_attn + std_attn, alpha=0.2)

    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Day relative to last input day")
    ax.set_ylabel("Attention weight")
    ax.set_title("Mean temporal attention by period")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, default="config.yaml")
    parser.add_argument("--output_dir",  type=str, default="xai_results")
    parser.add_argument("--n_events",    type=int, default=10,
                        help="Top events per period for Grad-CAM")
    parser.add_argument("--n_ig",        type=int, default=30,
                        help="Events per period for Integrated Gradients")
    parser.add_argument("--periods",     type=str,
                        default="1985-2004,2005-2014,2015-2024",
                        help="Comma-separated year ranges, e.g. '1985-2004,2005-2024'")
    parser.add_argument("--seasons",     type=str, default=None,
                        help="If set, run XAI by season instead of year ranges. "
                             "E.g. 'DJF,MAM,JJA,SON' or any subset. "
                             "Overrides --periods.")
    parser.add_argument("--season_years", type=str, default=None,
                        help="Year range for seasonal mode, e.g. '1985-2024'. "
                             "Defaults to full dataset range.")
    parser.add_argument("--no_cuda",     action="store_true")
    args = parser.parse_args()

    config     = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Parse periods / seasons
    if args.seasons is not None:
        # Seasonal mode: iterate over seasons, optionally restricted to a year range
        if args.season_years is not None:
            sy0, sy1 = args.season_years.strip().split("-")
            season_y0, season_y1 = int(sy0), int(sy1)
        else:
            season_y0, season_y1 = 1985, 2024  # will be overridden after data load
        seasons_requested = [s.strip() for s in args.seasons.split(",")]
        # periods here are (y0, y1, label, months_filter)
        periods = [
            (season_y0, season_y1, s, SEASONS[s])
            for s in seasons_requested if s in SEASONS
        ]
        seasonal_mode = True
    else:
        periods = []
        for p in args.periods.split(","):
            y0, y1 = p.strip().split("-")
            periods.append((int(y0), int(y1), f"{y0}–{y1}", None))
        seasonal_mode = False

    # --- Load model ---
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 128),
        lstm_hidden=config.get("lstm_hidden",   256),
        lstm_layers=config.get("lstm_layers",     2),
        temporal_features=3,
        dropout=config.get("dropout", 0.3),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        args.checkpoint, model=cnn_lstm, map_location=device,
    )
    lm.eval().to(device)
    print(f"Loaded: {args.checkpoint}")
    print(f"  target_mean={lm.target_mean:.4f}  target_std={lm.target_std:.4f}")

    # --- Load dataset with training normalisation stats ---
    # Use the datamodule to reproduce the exact stats from training
    dm = LazyDataModule(config_path=args.config)
    dm.setup()
    full_ds = dm.train_dataset.dataset   # underlying dataset, shared by all splits

    # Grid coordinates
    ds  = xr.open_dataset(config["data_dir"])
    lat = ds.lat.values
    lon = ds.lon.values
    ds.close()

    variables  = config["variables"]
    ocean_vars = set(config.get("ocean_variables", variables))

    print(f"\nFull dataset: {len(full_ds)} samples  "
          f"({full_ds.years[full_ds.window_size]} – "
          f"{full_ds.years[-full_ds.lead_time - 1]})")

    # --- Collect all predictions ---
    print("\nCollecting predictions over full dataset...")
    preds, trues, years, months = collect_predictions(lm, full_ds, device)

    # --- Per-period XAI ---
    from xai.grad_cam import AttentionGradCAM, _plot_cam, _plot_attention
    from xai.integrated_gradients import (
        _integrated_gradients, analyze_integrated_gradients
    )

    gradcam_engine = AttentionGradCAM(lm)

    # Accumulators for comparison plots
    cam_high_mean    = {}   # period_label → mean cam (lat, lon)
    ig_importance    = {}   # period_label → {var: value}
    ig_temporal      = {}   # period_label → (window, n_vars) temporal attribution
    attn_by_period   = {}   # period_label → list of attn arrays

    # In seasonal mode, extend year range to full dataset if not specified
    if seasonal_mode and args.season_years is None:
        y_min = int(years.min())
        y_max = int(years.max())
        periods = [(y_min, y_max, label, mf) for (_, _, label, mf) in periods]

    for y0, y1, label, months_filter in periods:
        mode_str = f"season={label}" if seasonal_mode else f"{y0}–{y1}"
        print(f"\n{'='*50}")
        print(f"Period: {mode_str}")
        period_dir = output_dir / label.replace("–", "-")
        period_dir.mkdir(exist_ok=True)

        sep = config.get("window_size", 60)
        idx_high = top_indices_for_period(preds, years, y0, y1, args.n_events, "high", sep,
                                          months=months, months_filter=months_filter)
        idx_low  = top_indices_for_period(preds, years, y0, y1, args.n_events, "low",  sep,
                                          months=months, months_filter=months_filter)

        if len(idx_high) == 0:
            print(f"  No samples in period {label}, skipping.")
            continue

        print(f"  High-anomaly events: {len(idx_high)}  "
              f"(pred range {preds[idx_high].min():.2f}–{preds[idx_high].max():.2f})")
        print(f"  Low-anomaly events:  {len(idx_low)}")

        # --- Grad-CAM ---
        print("  Running Grad-CAM...")
        cams_high, attn_high, preds_high = [], [], []
        for rank, idx in enumerate(idx_high):
            xs, xt, y_true = full_ds[idx]
            xs_dev = xs.unsqueeze(0).to(device)
            xt_dev = xt.unsqueeze(0).to(device)
            cam, attn = gradcam_engine.compute(xs_dev, xt_dev)

            yr, mo = years[idx], months[idx]
            label_s = (f"high #{rank+1}  {yr}-{mo:02d}  "
                       f"pred={preds[idx]:.2f}  true={trues[idx]:.2f}")
            cams_high.append(cam)
            attn_high.append(attn)
            preds_high.append(abs(preds[idx]))

            _plot_cam(cam, xs.numpy(), variables, ocean_vars,
                      lat, lon, label_s,
                      period_dir / f"gradcam_high_{rank+1}_{yr}.png")

        # Weighted average: weight each event by its prediction magnitude
        weights = np.array(preds_high)
        weights = weights / weights.sum()
        cam_high_mean[label] = (np.stack(cams_high) * weights[:, None, None]).sum(axis=0)
        attn_by_period[label] = attn_high

        # Low-anomaly Grad-CAM
        for rank, idx in enumerate(idx_low):
            xs, xt, y_true = full_ds[idx]
            xs_dev = xs.unsqueeze(0).to(device)
            xt_dev = xt.unsqueeze(0).to(device)
            cam, attn = gradcam_engine.compute(xs_dev, xt_dev)
            yr, mo = years[idx], months[idx]
            label_s = (f"low #{rank+1}  {yr}-{mo:02d}  "
                       f"pred={preds[idx]:.2f}  true={trues[idx]:.2f}")
            _plot_cam(cam, xs.numpy(), variables, ocean_vars,
                      lat, lon, label_s,
                      period_dir / f"gradcam_low_{rank+1}_{yr}.png")

        # Attention summary for this period
        _plot_attention(attn_high,
                        [f"{years[i]}-{months[i]:02d}" for i in idx_high],
                        period_dir / "attention_weights.png")

        # --- Integrated Gradients ---
        print(f"  Running Integrated Gradients ({args.n_ig} samples)...")
        ig_idx = top_indices_for_period(preds, years, y0, y1,
                                        args.n_ig, "high", sep,
                                        months=months, months_filter=months_filter)
        window_size    = config.get("window_size", 60)
        spatial_accum  = np.zeros((len(variables), len(lat), len(lon)))
        global_accum   = np.zeros(len(variables))
        temporal_accum = np.zeros((window_size, len(variables)))

        for k, idx in enumerate(ig_idx):
            xs, xt, _ = full_ds[idx]
            attrs = _integrated_gradients(
                lm,
                xs.unsqueeze(0).to(device),
                xt.unsqueeze(0).to(device),
            )
            abs_a           = attrs.abs()                              # (window, n_vars, lat, lon)
            spatial_accum  += abs_a.mean(dim=0).numpy()                # (n_vars, lat, lon)
            global_accum   += abs_a.mean(dim=(0, 2, 3)).numpy()        # (n_vars,)
            temporal_accum += abs_a.mean(dim=(2, 3)).numpy()           # (window, n_vars)
            print(f"    IG {k+1}/{len(ig_idx)}", end="\r")
        print()

        if len(ig_idx) > 0:
            spatial_accum  /= len(ig_idx)
            global_accum   /= len(ig_idx)
            temporal_accum /= len(ig_idx)
            ig_importance[label] = dict(zip(variables, global_accum))
            ig_temporal[label]   = temporal_accum

            # Spatial maps per variable
            # Build dilated land mask to suppress coastline gradient artifact
            from scipy.ndimage import binary_dilation
            land_mask_np = full_ds.tierra_mask.numpy()           # True = land
            coast_mask   = binary_dilation(land_mask_np, iterations=2)  # 2-pixel buffer

            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            for i, var in enumerate(variables):
                data = spatial_accum[i].copy()
                data = np.where(coast_mask, np.nan, data)  # mask land + coastline buffer
                fig, ax = plt.subplots(figsize=(8, 4))
                im = ax.imshow(data, origin="lower", extent=extent,
                               cmap="YlOrRd", aspect="auto")
                ax.set_title(f"IG {var} — {label}")
                ax.set_xlabel("lon"); ax.set_ylabel("lat")
                plt.colorbar(im, ax=ax, label="|attribution|")
                plt.tight_layout()
                plt.savefig(period_dir / f"ig_spatial_{var}.png",
                            dpi=150, bbox_inches="tight")
                plt.close()

        print(f"  Period {label} done. Files in {period_dir}/")

    # --- Comparison plots across periods ---
    print("\nGenerating comparison plots...")

    if len(cam_high_mean) > 1:
        plot_period_comparison(
            cam_high_mean, lat, lon,
            "Mean Grad-CAM — high-anomaly events by period",
            output_dir / "comparison_gradcam.png",
        )
        print("  Saved comparison_gradcam.png")

    if len(ig_importance) > 1:
        plot_ig_comparison(
            ig_importance, variables, ocean_vars,
            output_dir / "comparison_ig_importance.png",
        )
        print("  Saved comparison_ig_importance.png")

    if len(attn_by_period) > 1:
        plot_attention_comparison(
            attn_by_period,
            output_dir / "comparison_attention.png",
        )
        print("  Saved comparison_attention.png")

    if len(ig_temporal) > 0:
        plot_ig_temporal(
            ig_temporal, variables,
            output_dir / "comparison_ig_temporal.png",
        )
        print("  Saved comparison_ig_temporal.png")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
