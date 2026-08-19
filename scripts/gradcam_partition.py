"""
gradcam_partition.py — Grad-CAM on TbotAtm_{partition}_seed42_fold{0-4} (no MHW split).

Contrast method for ig_signed_partition.py: same checkpoints, same partition/fold
layout, ALL test samples, no high/low-anomaly or MHW split. Grad-CAM operates on
the last conv layer's activations (shared across all 5 input variables — it
cannot separate per-variable importance the way IG does), so this gives ONE
combined spatial saliency map per fold, not a per-variable bar chart. Useful as
an independent, structurally-smooth cross-check on WHERE the model looks
(native resolution 17x25, bilinear-upsampled — cannot show fine per-pixel
detail or coastal speckle by construction), not on per-variable ranking.

Cost: ~50x cheaper than vanilla IG per fold (one forward+backward per sample,
no n_steps integration loop, no noise-sample multiplier). Still an unbatched
per-sample loop (AttentionGradCAM.compute() takes one window at a time) — time
one fold before committing to all 5.

Usage (one fold per SLURM array task, or run directly — see cost note above):
  python scripts/gradcam_partition.py --fold 0 --partition full --model_tag mse_v2 [--max_samples N] [--output_dir ...]

Merge (after all 5 folds' partials exist):
  python scripts/gradcam_partition.py --merge --partition full --model_tag mse_v2 --output_dir experiments/gradcam_partition_full
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.datamodule import LazyDataModule
from src.data.masking import mask_local, mask_remote
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel
from src.utils.checkpoints import best_ckpt
from src.xai.grad_cam import AttentionGradCAM

REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "partition"


def _config_subdir(partition: str, model_tag: str) -> str:
    return f"{partition}_{model_tag}" if model_tag else partition


def _run_name(partition: str, model_tag: str, fold: int) -> str:
    tag = f"_{model_tag}" if model_tag else ""
    return f"TbotAtm_{partition}{tag}_seed42_fold{fold}"


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


def run_fold(
    fold: int,
    partition: str,
    out_dir: Path,
    model_tag: str = "",
    max_samples: "int | None" = None,
):
    out_path = out_dir / f"gradcam_partial_fold{fold}.npz"
    if out_path.exists():
        print(f"fold{fold}: partial already exists, skipping.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"\n=== fold {fold} | partition={partition} | model_tag={model_tag or '(none)'} | device={device} ==="
    )

    lm, cfg = load_model(fold, partition, device, model_tag)
    engine = AttentionGradCAM(lm)
    print(f"  is_lstm_only={engine.is_lstm_only}")

    cfg_path = str(
        REPO_ROOT
        / "configs"
        / "partition"
        / _config_subdir(partition, model_tag)
        / f"fold{fold}.yaml"
    )
    dm = LazyDataModule(cfg_path)
    dm.setup()
    test_dataset = dm.test_dataset
    n_test_full = len(test_dataset)

    if max_samples is not None and max_samples < n_test_full:
        idx = np.unique(np.linspace(0, n_test_full - 1, max_samples, dtype=int))
        print(
            f"  SUBSAMPLED test set: {len(idx)}/{n_test_full} samples (evenly spaced) "
            f"— this fold result is NOT the full-test-set result."
        )
    else:
        idx = np.arange(n_test_full)

    if partition == "remote":
        apply_mask = mask_remote
    elif partition == "local":
        apply_mask = mask_local
    else:
        apply_mask = lambda xs: xs  # noqa: E731

    sample_xs, _, _ = test_dataset[0]
    _, C, H, W = sample_xs.shape
    cam_sum = np.zeros((H, W), dtype=np.float64)
    n_processed = 0
    t_start = time.time()

    for k, i in enumerate(idx.tolist()):
        xs, xt, _ = test_dataset[int(i)]
        xs = apply_mask(xs.unsqueeze(0)).to(device)
        xt_dev = xt.unsqueeze(0).to(device)
        t0 = time.time()
        cam, _ = engine.compute(xs, xt_dev)
        cam_sum += cam
        n_processed += 1
        if (k + 1) % 100 == 0 or (k + 1) == len(idx):
            dt = time.time() - t0
            elapsed = time.time() - t_start
            eta = (elapsed / (k + 1)) * (len(idx) - (k + 1))
            print(
                f"  fold{fold}: {k+1}/{len(idx)} samples | last={dt:.2f}s "
                f"| elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min",
                flush=True,
            )

    np.savez(
        out_path,
        cam_mean=(cam_sum / n_processed).astype(np.float32),
        n_samples=n_processed,
        n_samples_full=n_test_full,
        fold=fold,
    )
    print(f"  Saved: {out_path}  (n={n_processed}/{n_test_full})")


def merge_and_plot(out_dir: Path, partition: str, model_tag: str = ""):
    partials = sorted(out_dir.glob("gradcam_partial_fold*.npz"))
    if len(partials) < 5:
        print(f"Only {len(partials)}/5 partials found. Run all folds first.")
        return

    cams = []
    for p in partials:
        d = np.load(p, allow_pickle=True)
        cams.append(d["cam_mean"])
        print(
            f"  Loaded {p.name}: n={d['n_samples']}/{d['n_samples_full']}, fold={d['fold']}"
        )

    cam_mean = np.stack(cams).mean(0)
    np.savez(out_dir / "gradcam_partition_merged.npz", cam_mean=cam_mean)

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
    is_land = ds["land_mask"].values == 0
    ds.close()
    coast_mask = binary_dilation(is_land, iterations=2)

    data = np.where(coast_mask, np.nan, cam_mean)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, origin="lower", extent=extent, cmap="YlOrRd", aspect="auto")
    land_rgba = np.zeros((*coast_mask.shape, 4))
    land_rgba[coast_mask, :] = [0.85, 0.85, 0.85, 1.0]
    ax.imshow(land_rgba, origin="lower", extent=extent, aspect="auto", zorder=2)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(f"Grad-CAM — {partition} (5-fold mean, all test samples)")
    plt.colorbar(im, ax=ax, label="Grad-CAM (normalised)")
    plt.tight_layout()
    plt.savefig(out_dir / "gradcam_spatial_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_dir}/gradcam_spatial_map.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--partition", choices=["full", "remote", "local"], default="full"
    )
    parser.add_argument("--model_tag", default="")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (REPO_ROOT / "experiments" / f"gradcam_partition_{args.partition}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_and_plot(out_dir, args.partition, args.model_tag)
    else:
        if args.fold is None:
            raise ValueError("--fold is required unless --merge is set")
        run_fold(args.fold, args.partition, out_dir, args.model_tag, args.max_samples)


if __name__ == "__main__":
    main()
