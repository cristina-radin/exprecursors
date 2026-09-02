"""
gradientshap_quantile_partition.py — Aug 22 2026, the "Shapley" leg of the
user's original XAI priority list ("GradCam, IG y Shapley"). Full
KernelSHAP is intractable for this input size (window=60, 5 vars,
141x201 grid) -- uses captum's GradientShap instead (Expected Gradients:
Erion et al. 2021), a gradient-based Shapley-value approximation that
integrates over a real DATA-DRAWN baseline distribution + random
interpolation + Gaussian noise, rather than IG's single fixed zero
baseline. Meaningfully different from IG methodologically (distribution
of real reference points, not one arbitrary point), while reusing this
project's established per-sample-batch=1 processing pattern (memory
safety already learned the hard way for IG, see
ig_partition_quantile.py's docstring).

Same conventions as IG/GradCAM this session: stratified_test_sample()
(known_issues.md #57 P1), both heads computed SEPARATELY (never
averaged, known_issues.md #49/#51), same land-mask plotting.

Baseline distribution: N_BASELINE real training-set windows, drawn once
and reused for every analyzed sample (fixed reference distribution, not
per-sample) -- standard Expected Gradients practice.

Usage:
  python scripts/gradientshap_quantile_partition.py \
      --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml \
      --output experiments/figures/xai_gradientshap/gradientshap_quantile_v2_landfill_fold0 \
      --max_samples 300 --n_baseline 16 --n_samples 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import GradientShap

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402
from src.utils.sampling import stratified_test_sample  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument(
        "--n_baseline",
        type=int,
        default=16,
        help="reference (background) samples drawn from train",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="captum GradientShap's internal n_samples",
    )
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
    train_indices = dm.train_dataset.indices
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
    model = lm.model

    # cuDNN LSTM backward-in-eval-mode workaround, same as IG/GradCAM.
    torch.backends.cudnn.enabled = False

    variables = cfg["variables"]
    n_vars = len(variables)

    # Fixed reference (background) distribution: real training windows,
    # drawn once, reused for every analyzed sample (standard Expected
    # Gradients practice, NOT a per-sample baseline).
    rng = np.random.default_rng(42)
    baseline_idx = rng.choice(train_indices, size=args.n_baseline, replace=False)
    baseline_xs = (
        torch.stack([full_ds[int(i)][0] for i in baseline_idx]).float().to(device)
    )
    # x_temporal for the baseline batch isn't used by forward_func (xt is
    # fixed per analyzed sample, passed via closure) -- only x_spatial
    # needs a baseline distribution for GradientShap's interpolation.
    print(
        f"Baseline: {args.n_baseline} real training windows, indices {sorted(baseline_idx.tolist())[:5]}...",
        flush=True,
    )

    def make_forward_fn(xt_fixed, head):
        def forward_fn(xs_batch):
            xt_rep = xt_fixed.expand(xs_batch.shape[0], -1, -1)
            if head == "mean":
                y_hat = model(xs_batch, xt_rep)
                return y_hat[:, 0:1]
            else:
                _, q_pred = model.forward_with_quantile(xs_batch, xt_rep)
                return q_pred

        return forward_fn

    sample_idx = stratified_test_sample(test_indices, full_ds, args.max_samples)
    print(
        f"Running GradientSHAP on {len(sample_idx)} samples (stratified by year, out of {len(test_indices)})",
        flush=True,
    )

    H = W = None
    sums = {h: None for h in heads}
    count = 0

    for i, idx in enumerate(sample_idx):
        xs, xt, _ = full_ds[idx]
        xs = xs.unsqueeze(0).float().to(device)
        xt = xt.unsqueeze(0).float().to(device)
        if H is None:
            H, W = xs.shape[-2], xs.shape[-1]
            for h in heads:
                sums[h] = torch.zeros(n_vars, H, W)

        for h in heads:
            gs = GradientShap(make_forward_fn(xt, h))
            attr = gs.attribute(
                xs, baselines=baseline_xs, n_samples=args.n_samples, stdevs=0.05
            )
            # attr: (1, window, n_vars, H, W) -- mean over window, same
            # reduction as IG's ig[0].mean(0).
            sums[h] += attr[0].mean(0).detach().cpu()

        count += 1
        if count % 25 == 0:
            print(f"  {count}/{len(sample_idx)}", flush=True)

    mean_attr = {h: (sums[h] / count).numpy() for h in heads}
    for h in heads:
        np.save(out_dir / f"gradientshap_{h}_head.npy", mean_attr[h])
    print(
        f"Saved gradientshap_{{{','.join(heads)}}}_head.npy, shape={mean_attr[heads[0]].shape}",
        flush=True,
    )

    import xarray as xr

    nc = xr.open_dataset(DATA_FILE)
    lats = nc.lat.values
    lons = nc.lon.values
    land_mask = nc["land_mask"].values if "land_mask" in nc else None
    land_mask_tbottom = (
        nc["land_mask_tbottom"].values if "land_mask_tbottom" in nc else None
    )
    ocean_variables = set(cfg.get("ocean_variables", []))
    plot_land_masks = {}
    for var in ocean_variables:
        if var == "ptho_bot" and land_mask_tbottom is not None:
            plot_land_masks[var] = land_mask_tbottom
        else:
            plot_land_masks[var] = land_mask

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    for h in heads:
        arr = np.abs(mean_attr[h])
        fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4))
        if n_vars == 1:
            axes = [axes]
        for vi, var in enumerate(variables):
            ax = axes[vi]
            data = arr[vi].copy()
            if var in plot_land_masks and plot_land_masks[var] is not None:
                data[plot_land_masks[var] == 0] = np.nan
                ax.set_facecolor("lightgray")
            im = ax.imshow(
                data, origin="lower", extent=extent, cmap="viridis", aspect="auto"
            )
            if land_mask is not None:
                ax.contour(
                    lons, lats, land_mask, levels=[0.5], colors="k", linewidths=0.5
                )
            ax.set_title(var, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(
            f"GradientSHAP (mean |attr|) — {h} head — {Path(args.config).parent.name}"
        )
        plt.tight_layout()
        save_path = out_dir / f"gradientshap_{h}_head.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {save_path}", flush=True)

    print(f"\nDone. All maps in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
