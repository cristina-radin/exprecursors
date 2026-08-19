"""
ig_masked_casestudy_2023.py — Signed IG case study for North Sea MHW Jun-Jul 2023.

Loads the 5 masked-model checkpoints from fold=2 (2023 is in the test set for that fold),
computes signed Integrated Gradients separately for:
  - Jun-Jul 2023 MHW samples  (to_anom > 0, target month in {6,7}, year=2023)
  - all other test samples    (climatological baseline)

Outputs:
  casestudy_signed_ig_{var}.png     — spatial maps: MHW 2023 | Baseline | Difference
  casestudy_barplot.png             — variable importance comparison bar chart
  casestudy_signed_ig.npy           — dict saved as npz: mhw_mean, base_mean, variables

Usage:
  python eval/ig_masked_casestudy_2023.py
  python eval/ig_masked_casestudy_2023.py --output_dir experiments/figures/xai_casestudy_2023
"""

import argparse
import re
import sys

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.patches as mpatches
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
TARGET_FOLD = 2  # 2023 is in the test set for fold 2
N_IG_STEPS = 50
MHW_YEAR = 2023
MHW_MONTHS = {6, 7}  # June, July


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
        pooling=cfg.get("pooling", "max"),
        quantile_head=cfg.get("quantile_head", False),
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt_path),
        model=inner,
        strict=False,
        map_location=device,
    )
    return lm.eval().to(device)


def compute_ig_signed(lm, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """
    Signed Integrated Gradients w.r.t. x_spatial.
    Positive attribution  → input feature pushed prediction higher (toward MHW).
    Negative attribution  → input feature pushed prediction lower (away from MHW).

    xs: (1, T, C, H, W) on device
    Returns attributions: (T, C, H, W) CPU  — SIGNED, no abs()
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
    return attributions.cpu()  # signed — no .abs()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="experiments/figures/xai_casestudy_2023"
    )
    parser.add_argument("--n_steps", type=int, default=50)
    args = parser.parse_args()
    global N_IG_STEPS
    N_IG_STEPS = args.n_steps

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Using fold={TARGET_FOLD} only ({MHW_YEAR} in test)")

    # Accumulate per group
    mhw_accum = None  # (C, H, W) signed IG — MHW Jun-Jul 2023
    base_accum = None  # (C, H, W) signed IG — all other test samples
    n_mhw = 0
    n_base = 0
    variables = None
    n_runs = 0

    for seed in SEEDS:
        fold = TARGET_FOLD
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

        print(f"\n[seed={seed}] {run_name}")
        lm = load_model(ckpt, cfg, device)

        dm = LazyDataModule(config_path=str(cfg_path))
        dm.setup()
        test_ds = dm.test_dataloader().dataset

        if variables is None:
            variables = cfg["variables"]
            xs0, _, _ = test_ds[0]
            _, C_, H, W = xs0.shape
            mhw_accum = np.zeros((C_, H, W), dtype=np.float64)
            base_accum = np.zeros((C_, H, W), dtype=np.float64)

        base_ds = test_ds.dataset if hasattr(test_ds, "dataset") else test_ds
        indices = (
            test_ds.indices if hasattr(test_ds, "indices") else range(len(test_ds))
        )

        n_mhw_this = 0
        n_base_this = 0

        for local_i, global_i in enumerate(indices):
            xs, xt, y = base_ds[global_i]
            target_t = global_i + base_ds.window_size - 1 + base_ds.lead_time
            yr = int(base_ds.years[target_t])
            mon = int(base_ds.months[target_t])

            xs_dev = xs.unsqueeze(0).to(device)
            xt_dev = xt.unsqueeze(0).to(device)

            attrs = compute_ig_signed(lm, xs_dev, xt_dev)  # (T, C, H, W) signed
            # Average over input time window T → (C, H, W)
            attrs_mean = attrs.mean(0).numpy()

            is_mhw_2023 = (yr == MHW_YEAR) and (mon in MHW_MONTHS)
            if is_mhw_2023:
                mhw_accum += attrs_mean
                n_mhw_this += 1
            else:
                base_accum += attrs_mean
                n_base_this += 1

            if (local_i + 1) % 50 == 0:
                print(
                    f"  {local_i+1}/{len(indices)}  mhw={n_mhw_this}  base={n_base_this}",
                    end="\r",
                )

        n_mhw += n_mhw_this
        n_base += n_base_this
        n_runs += 1
        print(f"\n  Done: mhw={n_mhw_this}  base={n_base_this} samples")

    if n_runs == 0:
        print("No runs completed.")
        return
    if n_mhw == 0:
        print(
            f"WARNING: No MHW {MHW_YEAR} Jun-Jul samples found in fold={TARGET_FOLD} test set!"
        )

    print(
        f"\nTotal across {n_runs} seeds: MHW samples={n_mhw}, baseline samples={n_base}"
    )

    # Normalize by sample count
    mhw_mean = mhw_accum / n_mhw if n_mhw > 0 else mhw_accum
    base_mean = base_accum / n_base if n_base > 0 else base_accum
    diff_mean = mhw_mean - base_mean

    # Save
    np.savez(
        str(out_dir / "casestudy_signed_ig.npz"),
        mhw_mean=mhw_mean,
        base_mean=base_mean,
        diff_mean=diff_mean,
        variables=np.array(variables),
    )
    print("Saved casestudy_signed_ig.npz")

    # Load grid metadata
    import xarray as xr

    ds_meta = xr.open_dataset(str(DATA_FILE_ENV))
    lat = ds_meta.lat.values
    lon = ds_meta.lon.values
    ocean = ds_meta["land_mask"].values.astype(bool)  # True = ocean
    ds_meta.close()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # NS box indices (approx)
    ns_lat_start, ns_lat_stop = 100, 127
    ns_lon_start, ns_lon_stop = 150, 187

    # ── Spatial maps (3 panels per variable) ─────────────────────────────────
    for ci, var in enumerate(variables):
        mhw_d = mhw_mean[ci].copy()
        base_d = base_mean[ci].copy()
        diff_d = diff_mean[ci].copy()
        for arr in (mhw_d, base_d, diff_d):
            arr[~ocean] = np.nan

        # Symmetric colormap for signed IG
        vabs = max(
            np.nanpercentile(np.abs(mhw_d), 98),
            np.nanpercentile(np.abs(base_d), 98),
        )
        dvabs = np.nanpercentile(np.abs(diff_d), 98)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        titles = [
            f"MHW Jun–Jul {MHW_YEAR}\n(n={n_mhw//n_runs} samples/seed)",
            f"Climatological baseline\n(all other test samples, n≈{n_base//n_runs}/seed)",
            "Difference\n(MHW − baseline)",
        ]
        datas = [mhw_d, base_d, diff_d]
        vmaxs = [vabs, vabs, dvabs]

        for ax, data, title, vmax in zip(axes, datas, titles, vmaxs):
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )
            # NS box
            ns_rect = mpatches.Rectangle(
                (lon[ns_lon_start], lat[ns_lat_start]),
                lon[ns_lon_stop - 1] - lon[ns_lon_start],
                lat[ns_lat_stop - 1] - lat[ns_lat_start],
                fill=False,
                edgecolor="black",
                linewidth=1.5,
                linestyle="--",
            )
            ax.add_patch(ns_rect)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, label="signed IG", shrink=0.8)

        fig.suptitle(
            f"Signed IG — {var} — MASKED model (NS zeroed, {n_runs} seeds, fold={TARGET_FOLD})\n"
            f"Positive = pushes prediction toward MHW  |  Negative = away from MHW",
            fontsize=10,
        )
        plt.tight_layout()
        fig.savefig(
            str(out_dir / f"casestudy_signed_ig_{var}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved casestudy_signed_ig_{var}.png")

    # ── Bar chart: mean signed IG per variable ────────────────────────────────
    # Global mean signed IG over ocean pixels
    mhw_var_imp = np.array(
        [np.nanmean(np.abs(mhw_mean[ci][ocean])) for ci in range(len(variables))]
    )
    base_var_imp = np.array(
        [np.nanmean(np.abs(base_mean[ci][ocean])) for ci in range(len(variables))]
    )

    x = np.arange(len(variables))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x - width / 2,
        mhw_var_imp / mhw_var_imp.sum() * 100,
        width,
        label=f"MHW Jun–Jul {MHW_YEAR}",
        color="tomato",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        base_var_imp / base_var_imp.sum() * 100,
        width,
        label="Climatological baseline",
        color="steelblue",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variables, rotation=15)
    ax.set_ylabel("% of total |IG|")
    ax.set_title(
        f"Variable importance comparison — MASKED model (fold={TARGET_FOLD}, {n_runs} seeds)"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / "casestudy_barplot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved casestudy_barplot.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n=== Mean |IG| per variable ===")
    print(f"{'Variable':<12}  {'MHW 2023':>10}  {'Baseline':>10}  {'Ratio':>8}")
    for ci, var in enumerate(variables):
        m = mhw_var_imp[ci]
        b = base_var_imp[ci]
        print(f"  {var:<12}  {m:10.4f}  {b:10.4f}  {m/b if b>0 else 0:8.2f}x")
    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
