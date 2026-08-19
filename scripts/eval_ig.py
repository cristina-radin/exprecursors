"""
composite_ig_hobday.py — Signed IG composite with proper Hobday MHW definition.

MHW classification:
  1. to_anom > p90_thresh(DOY)  — P90 of NS-averaged SST anomalies,
     ±5-day window, ref 1985-2014, smoothed with 31-day moving average
  2. 5-day persistence criterion (at least 5 consecutive days above threshold)
  3. Gap merging: gaps ≤2 days between events are merged

Usage:
  python eval/composite_ig_hobday.py \
      --checkpoint <ckpt> --config <yaml> --output_dir <dir>
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
from scipy.ndimage import binary_dilation, uniform_filter1d

sys.path.append(str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.paths import (
    CLIM_FILE as CLIM_FILE_ENV,
)
from src.xai.integrated_gradients import _integrated_gradients
from src.xai.utils import load_config

# ── Hobday MHW labelling ──────────────────────────────────────────────────────

CLIM_FILE = CLIM_FILE_ENV
NS_LAT = slice(50.0, 63.0)
NS_LON = slice(-5.0, 13.0)


def load_ns_p90(smooth_days: int = 31) -> np.ndarray:
    """Returns p90_thresh averaged over NS ocean pixels, shape (365,). 31-day smooth."""
    ds = xr.open_dataset(CLIM_FILE)
    ns = ds.p90_thresh.sel(lat=NS_LAT, lon=NS_LON)
    p90 = ns.mean(dim=["lat", "lon"], skipna=True).values  # (365,)
    ds.close()
    p90_smooth = uniform_filter1d(p90, size=smooth_days, mode="wrap")
    return p90_smooth


def apply_hobday(
    exceedance: np.ndarray, min_duration: int = 5, max_gap: int = 2
) -> np.ndarray:
    """
    Apply persistence + gap merging to a binary exceedance array.
    Operates on a contiguous time series (consecutive days).
    """
    mhw = exceedance.copy().astype(bool)
    n = len(mhw)

    # Step 1: fill gaps ≤ max_gap between exceedance runs
    i = 0
    while i < n:
        if mhw[i]:
            j = i
            while j < n and mhw[j]:
                j += 1  # j = first non-MHW after run
            k = j
            while k < n and not mhw[k]:
                k += 1  # k = start of next run
            if 0 < (k - j) <= max_gap and k < n:
                mhw[j:k] = True
                i = k
            else:
                i = j
        else:
            i += 1

    # Step 2: remove runs shorter than min_duration
    i = 0
    while i < n:
        if mhw[i]:
            j = i
            while j < n and mhw[j]:
                j += 1
            if j - i < min_duration:
                mhw[i:j] = False
            i = j
        else:
            i += 1

    return mhw


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    config = load_config(args.config)
    variables = config["variables"]
    window_size = config.get("window_size", 60)
    lead_time = config.get("lead_time", 7)

    # Load model
    cnn_lstm = CNNLSTMModel(
        in_channels=config["in_channels"],
        cnn_features=config.get("cnn_features", 256),
        lstm_hidden=config.get("lstm_hidden", 512),
        lstm_layers=config.get("lstm_layers", 2),
        temporal_features=config.get("temporal_features", 0),
        dropout=config.get("dropout", 0.2),
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
    days = np.arange(-window_size + 1, 1)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    # ── Load Hobday threshold ────────────────────────────────────────────────
    print("Loading NS p90 threshold (Hobday)...")
    p90_ns = load_ns_p90()  # (365,) DOY-indexed, 31-day smoothed
    print(
        f"  p90_ns: min={p90_ns.min():.3f}  max={p90_ns.max():.3f}  mean={p90_ns.mean():.3f} °C"
    )

    # ── Inference pass: collect trues, predictions, and target DOYs ──────────
    print("Collecting truth labels...")
    preds, trues, doys_target = [], [], []
    with torch.no_grad():
        for idx in range(len(full_ds)):
            xs, xt, y = full_ds[idx]
            p, _ = lm.model.forward_with_attention(
                xs.unsqueeze(0).float().to(device),
                xt.unsqueeze(0).float().to(device),
            )
            preds.append(p.item())
            trues.append(y.item())
            target_abs = idx + window_size - 1 + lead_time
            doy = int(full_ds.doys[target_abs])
            if doy == 366:
                doy = 365
            doys_target.append(doy)
            if (idx + 1) % 2000 == 0:
                print(f"  {idx+1}/{len(full_ds)}", end="\r")
    print()

    preds = np.array(preds)
    trues = np.array(trues)
    doys_target = np.array(doys_target, dtype=int)

    # ── Hobday classification ────────────────────────────────────────────────
    # Step 1: threshold exceedance per sample
    thr_per_sample = p90_ns[doys_target - 1]  # DOY 1-indexed → 0-indexed
    exceedance = trues > thr_per_sample

    # Step 2+3: persistence + gap merging on the ordered sequence
    mhw_labels = apply_hobday(exceedance)

    mhw_idx = np.where(mhw_labels)[0]
    nomhw_idx = np.where(~mhw_labels)[0]
    print(
        f"Exceedance days: {exceedance.sum()}  |  After Hobday: MHW={len(mhw_idx)}  non-MHW={len(nomhw_idx)}"
    )
    print(f"MHW fraction: {len(mhw_idx)/len(mhw_labels)*100:.1f}%  (expected ~10%)")

    rng = np.random.default_rng(42)
    sep = window_size

    def select_separated(candidates, n, sep):
        selected = []
        for c in candidates:
            if all(abs(c - s) >= sep for s in selected):
                selected.append(c)
            if len(selected) == n:
                break
        return np.array(selected)

    mhw_sorted = mhw_idx[np.argsort(preds[mhw_idx])[::-1]]
    mhw_sample = select_separated(mhw_sorted, args.n_samples, sep)
    nomhw_sample = select_separated(
        rng.choice(
            nomhw_idx, size=min(len(nomhw_idx), 5 * args.n_samples), replace=False
        ),
        args.n_samples,
        sep,
    )
    print(f"Selected: MHW={len(mhw_sample)}  non-MHW={len(nomhw_sample)}")

    land_mask_np = full_ds.is_land.numpy()
    coast_mask = binary_dilation(land_mask_np, iterations=2)

    groups = {"MHW": mhw_sample, "nonMHW": nomhw_sample}

    # ── IG accumulation (signed) ─────────────────────────────────────────────
    spatial_signed = {}
    temporal_signed = {}
    global_signed = {}

    for gname, idxs in groups.items():
        print(f"\nSigned IG for {gname} (n={len(idxs)})")
        sp_acc = np.zeros((n_vars, len(lat), len(lon)))
        tmp_acc = np.zeros((window_size, n_vars))
        gl_acc = np.zeros(n_vars)

        for k, idx in enumerate(idxs):
            xs, xt, _ = full_ds[idx]
            attrs = _integrated_gradients(
                lm,
                xs.unsqueeze(0).to(device),
                xt.unsqueeze(0).to(device),
            )
            sp_acc += attrs.mean(dim=0).numpy()
            tmp_acc += attrs.mean(dim=(2, 3)).numpy()
            gl_acc += attrs.mean(dim=(0, 2, 3)).numpy()
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(idxs)}", end="\r")
        print()

        spatial_signed[gname] = sp_acc / len(idxs)
        temporal_signed[gname] = tmp_acc / len(idxs)
        global_signed[gname] = gl_acc / len(idxs)

    # ── Plots ────────────────────────────────────────────────────────────────
    colors = {"MHW": "#d6604d", "nonMHW": "#4393c3"}

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    x, width = np.arange(n_vars), 0.35
    for i, (gname, vals) in enumerate(global_signed.items()):
        ax.bar(x + i * width, vals, width, label=gname, color=colors[gname], alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variables, rotation=15)
    ax.set_ylabel("Mean signed IG attribution")
    ax.set_title(
        f"Signed variable importance: MHW vs non-MHW (Hobday)\n"
        f"MHW={len(mhw_sample)} samples, non-MHW={len(nomhw_sample)} samples"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "hobday_ig_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Spatial maps
    sp_mhw = spatial_signed["MHW"]
    sp_nomhw = spatial_signed["nonMHW"]
    sp_diff = sp_mhw - sp_nomhw

    for i, var in enumerate(variables):
        m0 = np.where(coast_mask, np.nan, sp_mhw[i])
        m1 = np.where(coast_mask, np.nan, sp_nomhw[i])
        diff = np.where(coast_mask, np.nan, sp_diff[i])
        vmax_s = np.nanmax(np.abs([m0, m1]))
        vmax_d = np.nanmax(np.abs(diff))

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, data, title, vmax in [
            (axes[0], m0, f"MHW (n={len(mhw_sample)})", vmax_s),
            (axes[1], m1, f"non-MHW (n={len(nomhw_sample)})", vmax_s),
            (axes[2], diff, "Difference (MHW − non-MHW)", vmax_d),
        ]:
            im = ax.imshow(
                data,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, fraction=0.03, label="signed attr")

        fig.suptitle(f"Signed IG — {var}  [Hobday MHW definition]", fontsize=12)
        plt.tight_layout()
        fig.savefig(
            output_dir / f"hobday_ig_spatial_{var}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved: hobday_ig_spatial_{var}.png")

    # Temporal profiles
    fig, axes = plt.subplots(n_vars, 1, figsize=(12, 3 * n_vars), sharey=False)
    if n_vars == 1:
        axes = [axes]
    for j, (ax, var) in enumerate(zip(axes, variables)):
        for gname, tmp in temporal_signed.items():
            ax.plot(days, tmp[:, j], label=gname, lw=1.8, color=colors[gname])
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(var, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean signed IG")
        ax.grid(alpha=0.3)
        if j == n_vars - 1:
            ax.set_xlabel("Day relative to last input day")
    axes[0].legend()
    fig.suptitle("Signed temporal IG: MHW vs non-MHW [Hobday]", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "hobday_ig_temporal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
