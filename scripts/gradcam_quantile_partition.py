"""
gradcam_quantile_partition.py — Aug 22 2026, GradCAM for full_gnll_quantile_v2
-family models, matching ig_partition_quantile.py's conventions exactly so the
two methods are directly comparable:
  - Same --config path, same stratified_test_sample() (known_issues.md #57
    P1 -- avoids the chronological-truncation sampling bug).
  - Both heads computed SEPARATELY (mean, quantile), never averaged --
    same discipline as IG (known_issues.md #49/#51), and the reason
    src/xai/grad_cam.py's AttentionGradCAM needed a `head` parameter added
    same day (it used to backward() the raw (batch,2) [mean,log_var]
    output directly, which errors or silently mixes heads for
    gaussian_nll models).
  - Same land-mask-greyout plotting convention as IG (ocean_variables get
    NaN'd over land using land_mask_tbottom, atmospheric vars unmasked).

GradCAM only produces ONE combined spatial map (last-conv-layer activations
are shared across all 5 input variables, it cannot separate per-variable
importance the way IG can) -- this is a structurally-smooth, independent
cross-check on WHERE the model looks, not a per-variable ranking.

Usage:
  python scripts/gradcam_quantile_partition.py \
      --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml \
      --output experiments/figures/xai_gradcam/gradcam_quantile_v2_landfill_fold0 \
      --max_samples 300
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402
from src.utils.sampling import stratified_test_sample  # noqa: E402
from src.xai.grad_cam import AttentionGradCAM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument("--heads", default="mean,quantile")
    args = parser.parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    assert heads and set(heads) <= {
        "mean",
        "quantile",
    }, f"--heads must be a non-empty subset of {{mean,quantile}}, got {heads!r}"

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    dm = LazyDataModule(args.config)
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    print(f"Test samples available: {len(test_indices)}", flush=True)

    import yaml

    cfg = yaml.safe_load(open(args.config))
    run_dir = Path(cfg["output_dir"])
    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    assert model_kwargs.get("quantile_head", False), (
        "expected quantile_head=True in saved model_config.json -- "
        "wrong checkpoint or config drift, do not proceed silently."
    )
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"Loading {ckpt.name}", flush=True)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device
    )
    lm.eval().to(device)

    sample_idx = stratified_test_sample(test_indices, full_ds, args.max_samples)
    print(
        f"Running GradCAM on {len(sample_idx)} samples (stratified by year, out of {len(test_indices)})",
        flush=True,
    )

    H = W = None
    cam_sums = {h: None for h in heads}
    count = 0

    engines = {h: AttentionGradCAM(lm, head=h) for h in heads}

    for i, idx in enumerate(sample_idx):
        xs, xt, _ = full_ds[idx]
        xs = xs.unsqueeze(0).float().to(device)
        xt = xt.unsqueeze(0).float().to(device)
        if H is None:
            H, W = xs.shape[-2], xs.shape[-1]
            for h in heads:
                cam_sums[h] = np.zeros((H, W), dtype=np.float64)

        for h in heads:
            cam, _ = engines[h].compute(xs, xt)
            cam_sums[h] += cam

        count += 1
        if count % 50 == 0:
            print(f"  {count}/{len(sample_idx)}", flush=True)

    mean_cam = {h: (cam_sums[h] / count) for h in heads}
    for h in heads:
        np.save(out_dir / f"gradcam_{h}_head.npy", mean_cam[h].astype(np.float32))
    print(
        f"Saved gradcam_{{{','.join(heads)}}}_head.npy, shape={mean_cam[heads[0]].shape}",
        flush=True,
    )

    import xarray as xr

    nc = xr.open_dataset(DATA_FILE)
    lats = nc.lat.values
    lons = nc.lon.values
    land_mask_tbottom = (
        nc["land_mask_tbottom"].values if "land_mask_tbottom" in nc else None
    )
    # GradCAM's map is variable-agnostic (shared across all input
    # channels) -- grey out land only if ptho_bot is the sole ocean
    # variable and thus the map is "mostly about" ocean+atmosphere mixed;
    # still overlay the land contour for orientation regardless.
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    for h in heads:
        data = mean_cam[h].copy()
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            data, origin="lower", extent=extent, cmap="YlOrRd", aspect="auto"
        )
        if land_mask_tbottom is not None:
            ax.contour(
                lons, lats, land_mask_tbottom, levels=[0.5], colors="k", linewidths=0.6
            )
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(
            f"GradCAM — {h} head — {Path(args.config).parent.name}/{Path(args.config).stem}"
        )
        plt.colorbar(im, ax=ax, label="GradCAM (normalized)")
        plt.tight_layout()
        save_path = out_dir / f"gradcam_{h}_head.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {save_path}", flush=True)

    print(f"\nDone. All maps in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
