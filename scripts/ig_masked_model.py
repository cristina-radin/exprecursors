"""
ig_masked_model.py — Integrated Gradients on the masked SSTAtm model.

The masked model has North Sea SST zeroed in the input, forcing it to predict
using only remote Atlantic + atmosphere. IG on this model reveals genuine
non-local physical drivers of North Sea MHWs.

Runs across all 25 checkpoints (5 seeds x 5 folds), averages IG maps.

Outputs (saved to --output_dir):
  ig_spatial_<var>.png     — mean |IG| spatial map per variable
  ig_barplot.png           — variable importance bar chart
  ig_peryear.npy           — (n_years, n_vars) per-year importance
  ig_peryear.png           — heatmap of variable importance by year

Usage:
  python eval/ig_masked_model.py
  python eval/ig_masked_model.py --output_dir experiments/figures/xai_masked/
"""

import argparse
import re
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)
from src.utils.paths import (
    EXPERIMENTS_DIR,
)

MASKED_DIR = Path(str(EXPERIMENTS_DIR))
SEEDS = [42, 123, 456, 789, 1337]
N_FOLDS = 5
N_IG_STEPS = 50


# ── helpers ───────────────────────────────────────────────────────────────────


def best_ckpt(run_dir: Path) -> Path:
    ckpts = list((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}/checkpoints/")

    def val_loss(p):
        m = re.search(r"val_loss=(-?[\d]+\.[\d]+)", str(p))
        return float(m.group(1)) if m else 0.0

    return min(ckpts, key=val_loss)


def load_model(ckpt_path: Path, cfg: dict, device: str) -> CNNLightningModule:
    inner = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 256),
        lstm_hidden=cfg.get("lstm_hidden", 512),
        lstm_layers=cfg.get("lstm_layers", 4),
        temporal_features=cfg.get("temporal_features", 0),
        dropout=cfg.get("dropout", 0.3),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path),
        model=inner,
        strict=False,
        map_location=device,
    )
    return lm.eval().to(device)


def compute_ig(lm, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """
    Integrated Gradients w.r.t. x_spatial.
    xs: (1, T, C, H, W) on device
    Returns attributions: (T, C, H, W) CPU
    """
    baseline = torch.zeros_like(xs)
    grads = []
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    for step in range(N_IG_STEPS):
        alpha = step / (N_IG_STEPS - 1)
        interp = (baseline + alpha * (xs - baseline)).requires_grad_(True)
        pred, _ = lm.model.forward_with_attention(interp.float(), xt.float())
        pred[:, 0].sum().backward()  # mu channel only (works for MSE and GNLL)
        grads.append(interp.grad.detach().squeeze(0).clone())
    torch.backends.cudnn.enabled = prev_cudnn
    avg_grads = torch.stack(grads).mean(0)  # (T, C, H, W)
    attributions = (xs.squeeze(0) - baseline.squeeze(0)) * avg_grads
    return attributions.abs().cpu()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experiments/figures/xai_masked")
    parser.add_argument("--n_steps", type=int, default=50)
    args = parser.parse_args()
    global N_IG_STEPS
    N_IG_STEPS = args.n_steps

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Accumulate across all 25 runs
    # We'll collect per-year importance and spatial maps
    spatial_accum = None  # (C, H, W)
    peryear_runs = []  # list of dicts: {year: (C,)}
    variables = None
    total_runs = 0

    for seed in SEEDS:
        for fold in range(N_FOLDS):
            run_name = f"SSTAtm_lstmonly_gnll_masked_seed{seed}_fold{fold}"
            run_dir = MASKED_DIR / run_name

            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                print(f"  SKIP {run_name}: no config")
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)

            try:
                ckpt = best_ckpt(run_dir)
            except FileNotFoundError:
                print(f"  SKIP {run_name}: no checkpoint")
                continue

            print(f"\n[{total_runs+1}/25] {run_name}")
            lm = load_model(ckpt, cfg, device)

            dm = LazyDataModule(config_path=str(cfg_path))
            dm.setup()
            # Use test split only
            test_ds = dm.test_dataloader().dataset

            if variables is None:
                variables = cfg["variables"]
                # get H, W from first sample
                xs0, _, _ = test_ds[0]
                _, C_, H, W = xs0.shape
                spatial_accum = np.zeros((C_, H, W), dtype=np.float64)

            # Get year of each test sample
            base_ds = test_ds.dataset if hasattr(test_ds, "dataset") else test_ds
            indices = (
                test_ds.indices if hasattr(test_ds, "indices") else range(len(test_ds))
            )

            run_peryear: dict = {}  # year -> (n_samples, C)

            for local_i, global_i in enumerate(indices):
                xs, xt, y = base_ds[global_i]
                target_t = global_i + base_ds.window_size - 1 + base_ds.lead_time
                yr = int(base_ds.years[target_t])

                xs_dev = xs.unsqueeze(0).to(device)
                xt_dev = xt.unsqueeze(0).to(device)

                attrs = compute_ig(lm, xs_dev, xt_dev)  # (T, C, H, W)
                spatial_accum += attrs.mean(0).numpy()  # mean over T

                # Per-variable global importance for this sample
                var_imp = attrs.mean(dim=(0, 2, 3)).numpy()  # (C,)
                run_peryear.setdefault(yr, []).append(var_imp)

                if (local_i + 1) % 50 == 0:
                    print(f"  {local_i+1}/{len(indices)}", end="\r")

            print(f"  Done: {len(indices)} samples")

            # Compress to mean per year for this run
            run_yr_means = {yr: np.mean(v, axis=0) for yr, v in run_peryear.items()}
            peryear_runs.append(run_yr_means)
            total_runs += 1

    if total_runs == 0:
        print("No runs completed.")
        return

    spatial_accum /= total_runs
    print(f"\nAveraged over {total_runs} runs.")

    # ── Per-year heatmap ──────────────────────────────────────────────────────
    all_years = sorted(set(yr for run in peryear_runs for yr in run))
    peryear_mat = np.zeros((len(all_years), len(variables)), dtype=np.float64)
    peryear_cnt = np.zeros(len(all_years), dtype=int)
    for run in peryear_runs:
        for yi, yr in enumerate(all_years):
            if yr in run:
                peryear_mat[yi] += run[yr]
                peryear_cnt[yi] += 1
    for yi in range(len(all_years)):
        if peryear_cnt[yi] > 0:
            peryear_mat[yi] /= peryear_cnt[yi]

    # Normalise each year to sum=1 (relative importance)
    row_sums = peryear_mat.sum(axis=1, keepdims=True)
    peryear_pct = np.where(row_sums > 0, peryear_mat / row_sums * 100, 0)

    np.save(str(out_dir / "ig_peryear.npy"), peryear_pct)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        peryear_pct.T,
        aspect="auto",
        cmap="YlOrRd",
        extent=[all_years[0] - 0.5, all_years[-1] + 0.5, len(variables) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_xlabel("Year")
    ax.set_title(
        "Variable importance (% |IG|) — MASKED model (NS zeroed)\nOnly remote Atlantic + atmosphere"
    )
    plt.colorbar(im, ax=ax, label="% of total |IG|")
    plt.tight_layout()
    plt.savefig(str(out_dir / "ig_peryear.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ig_peryear.png")

    # ── Spatial maps ──────────────────────────────────────────────────────────
    ds_meta = __import__("xarray").open_dataset(str(DATA_FILE_ENV))
    lat = ds_meta.lat.values
    lon = ds_meta.lon.values
    ocean = ds_meta["land_mask"].values.astype(bool)
    ds_meta.close()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # Mark NS box
    ns_lat = slice(100, 127)
    ns_lon = slice(150, 187)

    for ci, var in enumerate(variables):
        data = spatial_accum[ci].copy()
        data[~ocean] = np.nan
        fig, ax = plt.subplots(figsize=(10, 5))
        vmax = np.nanpercentile(data, 98)
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            aspect="auto",
        )
        # Mark NS box

        ns_rect = plt.Rectangle(
            (lon[ns_lon.start], lat[ns_lat.start]),
            lon[ns_lon.stop - 1] - lon[ns_lon.start],
            lat[ns_lat.stop - 1] - lat[ns_lat.start],
            fill=False,
            edgecolor="blue",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(ns_rect)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(
            f"Mean |IG| spatial attribution — {var}\nMASKED model (NS zeroed, {total_runs} runs)"
        )
        plt.colorbar(im, ax=ax, label="|attribution|")
        plt.tight_layout()
        plt.savefig(
            str(out_dir / f"ig_spatial_{var}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(f"Saved ig_spatial_{var}.png")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    global_imp = spatial_accum.mean(axis=(1, 2))
    global_imp_pct = global_imp / global_imp.sum() * 100
    colors = ["tomato" if v == "to_anom" else "steelblue" for v in variables]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(variables, global_imp_pct, color=colors)
    ax.set_xlabel("% of total |IG|")
    ax.set_title(
        f"Variable importance — MASKED model ({total_runs} runs)\n(NS SST zeroed: only remote drivers)"
    )
    plt.tight_layout()
    plt.savefig(str(out_dir / "ig_barplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ig_barplot.png")

    print("\n=== Variable importance (global) ===")
    for var, pct in zip(variables, global_imp_pct):
        print(f"  {var:<12} {pct:.1f}%")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
