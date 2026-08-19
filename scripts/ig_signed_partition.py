"""
ig_signed_partition.py — Signed IG on TbotAtm_{partition}_seed42_fold{0-4} (no MHW split).

Runs on ALL test samples. Saves per-fold partial results and mid-fold checkpoints.
After all 5 partials exist, --merge produces final bar chart + spatial maps.

Usage (one fold per SLURM array task):
  python scripts/ig_signed_partition.py --fold 0 --partition full|remote|local [--model_tag mse_v2] [--output_dir ...]

Merge (run after all 5 array tasks complete):
  python scripts/ig_signed_partition.py --merge --partition remote --output_dir experiments/xai_partition_remote

--model_tag selects a trained variant sharing the same partition/fold layout,
e.g. --model_tag mse_v2 uses configs/partition/{partition}_mse_v2/fold{N}.yaml
and experiments/partition/TbotAtm_{partition}_mse_v2_seed42_fold{N}/checkpoints/
instead of the untagged TbotAtm_{partition}_seed42_fold{N} run.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import xarray as xr
import yaml
from scipy.ndimage import binary_dilation, gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.data.masking import _NS_LAT, _NS_LON, mask_local, mask_remote
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.checkpoints import best_ckpt

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"


def _config_subdir(partition: str, model_tag: str) -> str:
    return f"{partition}_{model_tag}" if model_tag else partition


def _run_name(partition: str, model_tag: str, fold: int) -> str:
    tag = f"_{model_tag}" if model_tag else ""
    return f"TbotAtm_{partition}{tag}_seed42_fold{fold}"


def _smooth_ignore_nan(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Gaussian smoothing for display only — does not touch the saved .npz.
    Vanilla IG gradients through the CNN encoder's MaxPool2d layers produce
    high-frequency per-pixel sign noise (well-known for gradient saliency
    maps, see SmoothGrad); this is cosmetic, applied after masking, using
    normalized convolution so NaN (land/partition-box) pixels don't bleed
    into neighbouring ocean pixels.
    """
    nan_mask = np.isnan(data)
    filled = np.where(nan_mask, 0.0, data)
    smoothed_sum = gaussian_filter(filled, sigma=sigma)
    weight = gaussian_filter((~nan_mask).astype(float), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed_sum / weight
    smoothed[nan_mask] = np.nan
    return smoothed


def load_model(fold: int, partition: str, device: str, model_tag: str = ""):
    cfg_path = (
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(partition, model_tag)
        / f"fold{fold}.yaml"
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    inner = CNNLSTMModel(
        in_channels=cfg["in_channels"],
        cnn_features=cfg.get("cnn_features", 128),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        temporal_features=cfg.get("temporal_features", 0),
        dropout=cfg.get("dropout", 0.2),
        arch=cfg.get("arch", "lstm_only"),
        gaussian_nll=cfg.get("gaussian_nll", False),
        pooling=cfg.get("pooling", "max"),
        padding_mode=cfg.get("padding_mode", "zeros"),
        quantile_head=cfg.get("quantile_head", False),
    )
    run_dir = EXPERIMENTS_DIR / _run_name(partition, model_tag, fold)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(
        f"  Loading checkpoint: {ckpt.name}  (config: {cfg_path.relative_to(REPO_ROOT)})"
    )
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=False, map_location=device
    )
    return lm.eval().to(device), cfg


def compute_ig_batch(
    model, xs_batch: torch.Tensor, xt_batch: torch.Tensor, n_steps: int, device: str
) -> torch.Tensor:
    """
    Signed batched IG for lstm_only (uses model.forward, not forward_with_attention).
    `model` is the raw CNNLSTMModel (already unwrapped from the LightningModule by
    the caller) — calling it directly invokes forward(), which returns only the
    prediction. forward_with_attention() is for arch != "lstm_only" and returns an
    extra attention-weights tensor this function does not use.
    xs_batch: (B, T, C, H, W)
    Returns:  (B, T, C, H, W) signed attributions on CPU.
    """
    xs_batch = xs_batch.to(device)
    xt_batch = xt_batch.to(device)
    baseline = torch.zeros_like(xs_batch)
    grads = []
    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    for step in range(n_steps):
        alpha = step / max(n_steps - 1, 1)
        interp = (baseline + alpha * (xs_batch - baseline)).requires_grad_(True)
        pred = model(interp.float(), xt_batch.float())  # (B, 1)
        pred[:, 0].sum().backward()
        grads.append(interp.grad.detach().clone())
    torch.backends.cudnn.enabled = prev
    avg_grads = torch.stack(grads).mean(0)
    return ((xs_batch - baseline) * avg_grads).cpu()


def compute_smoothgrad_batch(
    model,
    xs_batch: torch.Tensor,
    xt_batch: torch.Tensor,
    n_steps: int,
    n_samples: int,
    sigma: float,
    device: str,
) -> torch.Tensor:
    """
    Batched SmoothGrad: averages compute_ig_batch() over n_samples noisy
    copies of the WHOLE batch at once, instead of looping per-sample
    (see CLAUDE.md — an unbatched per-sample XAI loop already cost 2400
    core-hours once; the batch dimension must stay intact here).
    Cost is n_samples x compute_ig_batch, not n_samples x batch_size x.
    xs_batch: (B, T, C, H, W)
    Returns:  (B, T, C, H, W) signed attributions on CPU.
    """
    xs_batch = xs_batch.to(device)
    xt_batch = xt_batch.to(device)
    x_range = xs_batch.max() - xs_batch.min()
    noise_std = sigma * x_range.item()

    acc = torch.zeros_like(xs_batch, device="cpu")
    for i in range(n_samples):
        t0 = time.time()
        noisy = xs_batch + torch.randn_like(xs_batch) * noise_std
        acc += compute_ig_batch(model, noisy, xt_batch, n_steps, device)
        # Fine-grained progress: a single data-batch under smoothgrad can take
        # tens of minutes on CPU, so log every noise sample (not just per
        # batch) to see the per-iteration cost within seconds, not minutes.
        print(
            f"    smoothgrad sample {i + 1}/{n_samples} took {time.time() - t0:.1f}s",
            flush=True,
        )
    return acc / n_samples


def run_fold(
    fold: int,
    partition: str,
    out_dir: Path,
    n_steps: int,
    batch_size: int,
    model_tag: str = "",
    smoothgrad: bool = False,
    sg_sigma: float = 0.1,
    sg_samples: int = 20,
    max_batches: Optional[int] = None,
    max_samples: Optional[int] = None,
):
    """
    max_batches: process only the first N batches, then stop WITHOUT writing
    ig_partial_fold{N}.npz (that file marks a complete, mergeable fold — a
    truncated run must not produce one, or --merge would silently average a
    partial fold in with 4 complete ones). Timing-pilot use only.

    max_samples: use an evenly-spaced subsample of the test set (deterministic,
    covers the full test period rather than just its start) and DOES write a
    complete ig_partial_fold{N}.npz — this is a cheaper but still-valid fold
    result, not a timing pilot. The npz records n_samples_full alongside
    n_samples so a subsampled fold is never silently mistaken for the full one.
    """
    out_path = out_dir / f"ig_partial_fold{fold}.npz"
    if out_path.exists():
        print(f"fold{fold}: partial already exists, skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        torch.set_num_threads(n_threads)
    print(
        f"\n=== fold {fold} | partition={partition} | model_tag={model_tag or '(none)'} "
        f"| device={device} | n_steps={n_steps} | batch={batch_size} "
        f"| smoothgrad={smoothgrad} (n_samples={sg_samples}, sigma={sg_sigma}) ==="
    )

    lm, cfg = load_model(fold, partition, device, model_tag)
    variables = cfg["variables"]

    cfg_path = str(
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(partition, model_tag)
        / f"fold{fold}.yaml"
    )
    dm = LazyDataModule(cfg_path)
    dm.setup()
    # num_workers=0: multiprocessing workers die silently on compute nodes during inference
    from torch.utils.data import DataLoader, Subset

    test_dataset = dm.test_dataset
    n_test_full = len(test_dataset)
    if max_samples is not None and max_samples < n_test_full:
        idx = np.unique(np.linspace(0, n_test_full - 1, max_samples, dtype=int))
        test_dataset = Subset(test_dataset, idx.tolist())
        print(
            f"  SUBSAMPLED test set: {len(idx)}/{n_test_full} samples "
            f"(evenly spaced) — this fold result is NOT the full-test-set result."
        )
    test_dl = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    n_test = len(test_dataset)
    print(f"  Test samples: {n_test} (full test set: {n_test_full})")

    C = len(variables)
    # Get H, W from already-loaded dataset — avoids re-opening the data file on compute nodes
    sample_xs, _, _ = dm.test_dataset[0]
    _, C_check, H, W = sample_xs.shape  # (T, C, H, W)
    assert C_check == C, f"Expected {C} channels, got {C_check}"

    ckpt_path = out_dir / f"ig_ckpt_fold{fold}.npz"
    spatial_signed_sum = np.zeros((C, H, W), dtype=np.float64)
    spatial_abs_sum = np.zeros((C, H, W), dtype=np.float64)
    var_imp_sum = np.zeros(C, dtype=np.float64)
    n_processed = 0
    start_batch = 0

    # Resume from mid-fold checkpoint if present
    if ckpt_path.exists():
        ck = np.load(ckpt_path)
        spatial_signed_sum = ck["spatial_signed_sum"].astype(np.float64)
        spatial_abs_sum = ck["spatial_abs_sum"].astype(np.float64)
        var_imp_sum = ck["var_imp_sum"].astype(np.float64)
        n_processed = int(ck["n_processed"])
        start_batch = int(ck["next_batch"])
        print(
            f"  Resuming fold{fold} from batch {start_batch} ({n_processed} samples done)"
        )

    model = lm.model

    # Partition masking is data-level (applied to xs), not part of the model
    # or LightningModule.forward() — see src/data/masking.py and
    # scripts/train_partition.py (RemoteOnlyLightningModule/LocalOnlyLightningModule).
    # The checkpoint was trained on masked input, so IG must see the same masking.
    if partition == "remote":
        apply_mask = mask_remote
    elif partition == "local":
        apply_mask = mask_local
    else:
        apply_mask = lambda xs: xs  # noqa: E731 — full: no masking

    n_batches_total = (n_test + batch_size - 1) // batch_size
    for batch_idx, (xs, xt, _) in enumerate(test_dl):
        if batch_idx < start_batch:
            continue
        t0 = time.time()
        xs = apply_mask(xs)
        if smoothgrad:
            attr = compute_smoothgrad_batch(
                model, xs, xt, n_steps, sg_samples, sg_sigma, device
            )
        else:
            attr = compute_ig_batch(model, xs, xt, n_steps, device)
        attr_time_mean = attr.mean(dim=1).numpy()  # (B, C, H, W)

        spatial_signed_sum += attr_time_mean.sum(axis=0)
        spatial_abs_sum += np.abs(attr_time_mean).sum(axis=0)
        var_imp_sum += np.abs(attr_time_mean).mean(axis=(2, 3)).sum(axis=0)
        n_processed += xs.shape[0]
        batch_dt = time.time() - t0
        remaining = n_batches_total - (batch_idx + 1)
        print(
            f"  fold{fold}: batch {batch_idx + 1}/{n_batches_total} took {batch_dt:.1f}s "
            f"| {n_processed}/{n_test} samples | ETA rest-of-fold: "
            f"{remaining * batch_dt / 60:.1f} min",
            flush=True,
        )

        if (batch_idx + 1) % 20 == 0:
            # Save mid-fold checkpoint every 20 batches
            np.savez(
                ckpt_path,
                spatial_signed_sum=spatial_signed_sum.astype(np.float32),
                spatial_abs_sum=spatial_abs_sum.astype(np.float32),
                var_imp_sum=var_imp_sum.astype(np.float32),
                n_processed=n_processed,
                next_batch=batch_idx + 1,
            )

        if max_batches is not None and (batch_idx + 1 - start_batch) >= max_batches:
            print(
                f"  fold{fold}: stopping after {max_batches} batches (timing pilot — "
                f"NOT writing {out_path.name}, this fold is incomplete)."
            )
            return

    np.savez(
        out_path,
        spatial_signed_mean=(spatial_signed_sum / n_processed).astype(np.float32),
        spatial_abs_mean=(spatial_abs_sum / n_processed).astype(np.float32),
        var_importance=(var_imp_sum / n_processed).astype(np.float32),
        n_samples=n_processed,
        n_samples_full=n_test_full,
        variables=np.array(variables),
        fold=fold,
    )
    ckpt_path.unlink(missing_ok=True)  # clean up mid-fold checkpoint
    subsampled_note = (
        f"  *** SUBSAMPLE: {n_processed}/{n_test_full} of full test set ***"
        if n_processed < n_test_full
        else ""
    )
    print(f"  Saved: {out_path}  (n={n_processed})")
    if subsampled_note:
        print(subsampled_note)


def merge_and_plot(out_dir: Path, partition: str, model_tag: str = ""):
    partials = sorted(out_dir.glob("ig_partial_fold*.npz"))
    if len(partials) < 5:
        print(f"Only {len(partials)}/5 partials found. Run all folds first.")
        return

    all_var_imp = []
    all_spatial_signed = []
    all_spatial_abs = []
    variables = None

    for p in partials:
        d = np.load(p, allow_pickle=True)
        all_var_imp.append(d["var_importance"])  # (C,)
        all_spatial_signed.append(d["spatial_signed_mean"])  # (C, H, W)
        all_spatial_abs.append(d["spatial_abs_mean"])  # (C, H, W)
        variables = list(d["variables"])
        print(f"  Loaded {p.name}: n={d['n_samples']}, fold={d['fold']}")

    var_imp = np.stack(all_var_imp)  # (5, C)
    spatial_signed = np.stack(all_spatial_signed)  # (5, C, H, W)
    spatial_abs = np.stack(all_spatial_abs)  # (5, C, H, W)

    np.savez(
        out_dir / "ig_signed_partition_merged.npz",
        var_imp_mean=var_imp.mean(0),
        var_imp_std=var_imp.std(0),
        spatial_signed_mean=spatial_signed.mean(0),
        spatial_abs_mean=spatial_abs.mean(0),
        variables=np.array(variables),
    )
    print(f"\nSaved merged: {out_dir}/ig_signed_partition_merged.npz")

    # Land mask/lat/lon must come from the SAME data_dir the model was
    # trained/evaluated on — a stale file here silently reintroduces the
    # 0.25 deg grid-offset bug the v2 data fixed (mask misaligned with the
    # attributions it's overlaid on).
    fold0_cfg_path = (
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(partition, model_tag)
        / "fold0.yaml"
    )
    with open(fold0_cfg_path) as f:
        data_file = yaml.safe_load(f)["data_dir"]
    ds = xr.open_dataset(data_file, engine="netcdf4")
    lat = ds.lat.values
    lon = ds.lon.values
    # land_mask: 1=ocean, 0=land (known_issues.md #2) — invert to get is_land.
    # ptho_bot has its own native mask (land_mask_tbottom, 572px different) —
    # must NOT reuse the SST/atmosphere is_land for it (known_issues.md #2
    # "CORRECTION" note: same bug as dataset.py, fixed there Aug 17 2026; this
    # is the plotting-side counterpart — using the SST mask here would NaN out
    # (or fail to NaN out, depending on data provenance) exactly the wrong 572
    # pixels for ptho_bot's own map).
    is_land = ds["land_mask"].values == 0
    is_land_tbottom = ds["land_mask_tbottom"].values == 0
    ds.close()

    # Same coastal buffer as composite_ig.py (binary_dilation, iterations=2)
    coast_mask = binary_dilation(is_land, iterations=2)
    coast_mask_tbottom = binary_dilation(is_land_tbottom, iterations=2)

    # Partition box mask: for remote/local, half the domain is forced to exactly
    # zero by mask_remote/mask_local (src/data/masking.py), not a genuine "zero
    # attribution" finding. NaN it out separately from land so the two are never
    # visually conflated.
    box_mask = np.zeros_like(coast_mask, dtype=bool)
    if partition == "remote":
        box_mask[_NS_LAT, _NS_LON] = True  # input forced to 0 inside NS box
    elif partition == "local":
        box_mask[:, :] = True
        box_mask[_NS_LAT, _NS_LON] = False  # input forced to 0 outside NS box

    def plot_mask_for(var: str) -> np.ndarray:
        base = coast_mask_tbottom if var == "ptho_bot" else coast_mask
        return base | box_mask

    def add_land_shading(ax, land_mask_2d):
        """Overlay gray shading on land pixels."""
        land_rgba = np.zeros((*land_mask_2d.shape, 4))
        land_rgba[land_mask_2d, :] = [0.85, 0.85, 0.85, 1.0]  # light gray, opaque
        land_rgba[~land_mask_2d, :] = [0, 0, 0, 0]  # transparent over ocean
        ax.imshow(
            land_rgba,
            origin="lower",
            aspect="auto",
            extent=[lon[0], lon[-1], lat[0], lat[-1]],
            zorder=2,
        )

    partition_label = {"full": "Full", "remote": "Remote", "local": "Local"}[partition]
    C = len(variables)

    def plot_var_importance(imp, imp_err, title, out_name):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(C)
        order = np.argsort(imp)[::-1]
        ax.bar(
            x,
            imp[order],
            yerr=imp_err[order] if imp_err is not None else None,
            capsize=4,
            color="#4393c3",
            error_kw=dict(elinewidth=1.2),
        )
        ax.set_xticks(x)
        ax.set_xticklabels([variables[i] for i in order], fontsize=11)
        ax.set_ylabel("Mean |signed IG| (all test samples)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_name}")

    def masked_smoothed(signed_map):
        """(C, H, W) -> masked + smoothed for display, plus robust per-variable vmax."""
        masked = np.stack(
            [
                np.where(plot_mask_for(var), np.nan, signed_map[i])
                for i, var in enumerate(variables)
            ]
        )
        masked = np.stack([_smooth_ignore_nan(masked[i]) for i in range(C)])
        # Robust vmax (99th percentile of |attr|, ocean pixels only) — a single
        # outlier pixel shouldn't wash out the rest of that variable's map.
        vmax = [np.nanpercentile(np.abs(masked[i]), 99) for i in range(C)]
        return masked, vmax

    def plot_spatial_panel(signed_map, title, out_name):
        masked, vmax = masked_smoothed(signed_map)
        fig, axes = plt.subplots(1, C, figsize=(5 * C, 4))
        for i, var in enumerate(variables):
            ax = axes[i]
            im = ax.imshow(
                masked[i],
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax[i],
                vmax=vmax[i],
                extent=[lon[0], lon[-1], lat[0], lat[-1]],
            )
            base = coast_mask_tbottom if var == "ptho_bot" else coast_mask
            add_land_shading(ax, base | box_mask)
            ax.set_title(var, fontsize=11)
            ax.set_xlabel("lon")
            if i == 0:
                ax.set_ylabel("lat")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed IG")
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_name}")

    # ── Per-fold plots ────────────────────────────────────────────────────────
    for fold_i, p in enumerate(partials):
        d = np.load(p, allow_pickle=True)
        fold_num = int(d["fold"])
        plot_var_importance(
            var_imp[fold_i],
            None,
            f"TbotAtm {partition_label} — variable importance (fold {fold_num})",
            f"ig_var_importance_fold{fold_num}.png",
        )
        plot_spatial_panel(
            spatial_signed[fold_i],
            f"TbotAtm {partition_label} — signed IG spatial maps (fold {fold_num})",
            f"ig_spatial_maps_fold{fold_num}.png",
        )

    # ── Merged (5-fold mean) plots ───────────────────────────────────────────
    plot_var_importance(
        var_imp.mean(0),
        var_imp.std(0),
        f"TbotAtm {partition_label} — variable importance (5-fold mean ± std)",
        "ig_var_importance.png",
    )
    plot_spatial_panel(
        spatial_signed.mean(0),
        f"TbotAtm {partition_label} — signed IG spatial maps (5-fold mean)",
        "ig_spatial_maps.png",
    )

    # Per-variable files (for paper-quality figures) — merged mean only, same
    # robust per-variable scale as the merged panel above.
    signed_mean_masked, var_vmax = masked_smoothed(spatial_signed.mean(0))
    for i, var in enumerate(variables):
        fig, ax = plt.subplots(figsize=(8, 4))
        vm = var_vmax[i]
        im = ax.imshow(
            signed_mean_masked[i],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vm,
            vmax=vm,
            extent=[lon[0], lon[-1], lat[0], lat[-1]],
        )
        base = coast_mask_tbottom if var == "ptho_bot" else coast_mask
        add_land_shading(ax, base | box_mask)
        ax.set_title(
            f"Signed IG — {var} (TbotAtm {partition_label}, 5-fold mean)", fontsize=11
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, label="signed IG attribution")
        plt.tight_layout()
        fig.savefig(out_dir / f"ig_spatial_{var}.png", dpi=150)
        plt.close(fig)

    print("\nDone. Outputs in:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, choices=range(5), help="Fold to process (0-4)"
    )
    parser.add_argument(
        "--merge", action="store_true", help="Merge partials and produce plots"
    )
    parser.add_argument(
        "--partition",
        choices=["full", "remote", "local"],
        default="full",
        help="Which partition checkpoints to use",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--smoothgrad",
        action="store_true",
        help="Use SmoothGrad instead of vanilla IG (reduces MaxPool2d grid artefacts)",
    )
    parser.add_argument(
        "--sg_sigma",
        type=float,
        default=0.1,
        help="SmoothGrad noise std as fraction of input range (default: 0.1)",
    )
    parser.add_argument(
        "--sg_samples",
        type=int,
        default=20,
        help="SmoothGrad number of noisy runs per sample (default: 20)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Timing-pilot only: stop after N batches, do NOT write "
        "ig_partial_fold{N}.npz (incomplete fold, not mergeable). "
        "Default: process the whole test set.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use an evenly-spaced subsample of N test samples instead of the "
        "full test set. DOES write a complete ig_partial_fold{N}.npz "
        "(records n_samples_full so it's not mistaken for the full-test "
        "result). Cheaper alternative to processing everything.",
    )
    parser.add_argument(
        "--model_tag",
        default="",
        help="Suffix selecting which trained variant to explain, e.g. 'mse_v2' "
        "-> configs/partition/{partition}_mse_v2/, "
        "experiments/partition/TbotAtm_{partition}_mse_v2_seed42_fold{N}/. "
        "Empty (default) keeps the original TbotAtm_{partition}_seed42_fold{N} models.",
    )
    args = parser.parse_args()

    out_dir_str = args.output_dir or f"experiments/xai_partition_{args.partition}"
    out_dir = REPO_ROOT / out_dir_str
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_and_plot(out_dir, args.partition, args.model_tag)
    elif args.fold is not None:
        run_fold(
            args.fold,
            args.partition,
            out_dir,
            args.n_steps,
            args.batch_size,
            args.model_tag,
            args.smoothgrad,
            args.sg_sigma,
            args.sg_samples,
            args.max_batches,
            args.max_samples,
        )
        # Auto-merge if all 5 partials are present
        if len(list(out_dir.glob("ig_partial_fold*.npz"))) == 5:
            print("\nAll 5 folds complete — running merge automatically.")
            merge_and_plot(out_dir, args.partition, args.model_tag)
    else:
        parser.error("Specify --fold N or --merge")


if __name__ == "__main__":
    main()
