#!/usr/bin/env python
"""
Per-year importance heatmap including temporal features (year_norm, month_sin, month_cos).

Same logic as poster_ig_peryear.py but uses _integrated_gradients_full so that
x_temporal attributions are also computed. Produces a heatmap with spatial
variables + temporal features as rows, allowing comparison of year_norm importance
vs T_bottom importance over time.

Usage:
  python eval/poster_ig_peryear_temporal.py \
    --exp_dirs split_blockyear/TbotAtm_seed100 ... seed400 \
    --npz      split_blockyear/TbotAtm_ensemble/eval_results/predictions.npz \
    --output   poster_figures/fig_ig_peryear_temporal.png \
    --n_per_year 5
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))
from datamodule import LazyDataModule
from model import CNNLightningModule, CNNLSTMModel
from xai.utils import load_config
from xai.integrated_gradients import _integrated_gradients_full
from xai.run_xai import top_indices_for_period

FONTSIZE  = 18
TITLESIZE = 20
TICKSIZE  = 14

VAR_LABELS = {
    "ptho_bot":   "T$_{bottom}$",
    "to_anom":    "SST anom.",
    "u10":        "U-wind",
    "v10":        "V-wind",
    "msl":        "SLP",
    "ssr":        "Solar rad.",
    "year_norm":  "Year",
    "month_sin":  "Month (sin)",
    "month_cos":  "Month (cos)",
}

TEMPORAL_VARS = ["year_norm", "month_sin", "month_cos"]


def best_checkpoint(exp_dir):
    ckpts = list((exp_dir / "checkpoints").glob("*.ckpt"))
    def val_loss(p):
        try: return float(str(p).split("val_loss=")[1].replace(".ckpt", ""))
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
        str(ckpt_path), model=cnn_lstm, map_location=device
    )
    lm.eval().to(device)
    return lm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dirs",   nargs="+", required=True)
    parser.add_argument("--npz",        required=True)
    parser.add_argument("--output",     default="poster_figures/fig_ig_peryear_temporal.png")
    parser.add_argument("--n_per_year", type=int, default=5)
    parser.add_argument("--no_cuda",    action="store_true")
    args = parser.parse_args()

    device   = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exp_dirs = [Path(d) for d in args.exp_dirs]

    config0   = load_config(str(exp_dirs[0] / "config.yaml"))
    variables = config0["variables"]
    n_vars    = len(variables)
    all_row_vars = variables + TEMPORAL_VARS   # spatial vars + year/sin/cos

    d        = np.load(args.npz, allow_pickle=True)
    ens_pred = d["ens_full"]
    years    = d["years"]
    all_years = sorted(set(years.astype(int)))

    models, datasets = [], []
    for exp_dir in exp_dirs:
        config = load_config(str(exp_dir / "config.yaml"))
        lm     = load_model(best_checkpoint(exp_dir), config, device)
        dm     = LazyDataModule(config_path=str(exp_dir / "config.yaml"))
        dm.setup()
        models.append(lm)
        datasets.append(dm.train_dataset.dataset)
        print(f"  Loaded seed={config.get('seed', '?')}")

    year_importance = {}   # year → (n_vars + 3,) normalized

    for yr in all_years:
        idx_yr = top_indices_for_period(
            ens_pred, years, yr, yr, args.n_per_year, "high", min_separation=30
        )
        if len(idx_yr) == 0:
            print(f"  {yr}: no events, skipping")
            continue

        accum = np.zeros(n_vars + 3)
        count = 0
        for lm, full_ds in zip(models, datasets):
            for idx in idx_yr:
                xs, xt, _ = full_ds[idx]
                attr_s, attr_t = _integrated_gradients_full(
                    lm,
                    xs.unsqueeze(0).to(device),
                    xt.unsqueeze(0).to(device),
                )
                # spatial: mean over window + spatial dims → (n_vars,)
                accum[:n_vars] += attr_s.abs().mean(dim=(0, 2, 3)).numpy()
                # temporal: mean over window → (3,)  [year_norm, sin, cos]
                accum[n_vars:] += attr_t.abs().mean(dim=0).numpy()
                count += 1

        accum /= count
        year_importance[yr] = accum / accum.sum() * 100
        print(f"  {yr}: {len(idx_yr)} events × {len(exp_dirs)} seeds  "
              + "  ".join(f"{v}={year_importance[yr][i]:.1f}%"
                          for i, v in enumerate(all_row_vars)))

    if not year_importance:
        print("No years with events found.")
        return

    yrs_with_data = sorted(year_importance.keys())
    mat = np.array([year_importance[y] for y in yrs_with_data])   # (n_years, n_vars+3)

    row_labels = [VAR_LABELS.get(v, v) for v in all_row_vars]
    n_rows = len(all_row_vars)

    # --- Figure: two panels, share colorbar ---
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [n_vars, 3]},
    )

    vmax = mat.max()

    # Top panel — spatial variables
    im0 = axes[0].imshow(mat[:, :n_vars].T, aspect="auto",
                         cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[0].set_xticks([])
    axes[0].set_yticks(range(n_vars))
    axes[0].set_yticklabels(row_labels[:n_vars], fontsize=TICKSIZE)
    axes[0].set_title(
        f"Variable importance per year — Integrated Gradients (full)\n"
        f"Ensemble of {len(exp_dirs)} seeds  ·  top-{args.n_per_year} events/year  ·  7-day lead",
        fontsize=TITLESIZE, fontweight="bold", pad=8,
    )

    # Bottom panel — temporal features
    im1 = axes[1].imshow(mat[:, n_vars:].T, aspect="auto",
                         cmap="YlOrRd", vmin=0, vmax=vmax)
    axes[1].set_xticks(range(len(yrs_with_data)))
    axes[1].set_xticklabels(yrs_with_data, rotation=90, fontsize=TICKSIZE - 2)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(row_labels[n_vars:], fontsize=TICKSIZE)

    # Divider label
    axes[1].set_xlabel("Year", fontsize=FONTSIZE)

    # Shared colorbar
    cbar = fig.colorbar(im0, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("% of total |IG|", fontsize=FONTSIZE - 2)
    cbar.ax.tick_params(labelsize=TICKSIZE - 2)

    # Separator line between panels
    axes[0].spines["bottom"].set_linewidth(2)
    axes[1].spines["top"].set_linewidth(2)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    np.savez(
        out.parent / "ig_peryear_temporal.npz",
        years=np.array(yrs_with_data),
        importance=mat,
        variables=np.array(all_row_vars),
    )
    print(f"Saved: {out.parent / 'ig_peryear_temporal.npz'}")


if __name__ == "__main__":
    main()
