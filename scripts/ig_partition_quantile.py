"""
ig_partition_quantile.py — Integrated Gradients for full_gnll_quantile_v2,
Aug 21 2026. Replaces ig_simple.py for this model family, which turned
out to be broken independent of the quantile-head gap (known_issues.md
#49): imports a dataset class that no longer exists (MHWDataset ->
LazyDataset), uses the wrong (non-stratified) split, and -- the real bug
-- averages the mean AND log-variance output columns together before
backprop for gaussian_nll models, producing a meaningless IG map.

This script instead reuses the same already-verified building blocks
used everywhere else this session:
  - LazyDataModule for the split + normalization (stratified_kfold-aware,
    identical to what the model was actually trained/evaluated on).
  - best_ckpt() + load_model_config() + CNNLightningModule.
    load_from_checkpoint() for correct checkpoint restoration (matches
    scripts/analysis/quantile_head_recall*.py's pattern).

Computes IG for TWO separate outputs per sample (never averaged
together):
  1. mean head  (model.forward(...), column 0)      -- point forecast
  2. quantile head (model.forward_with_quantile(...)'s q_pred) --
     the actual precursor/exceedance-detection signal (see
     docs/narrative.md's Aug 21 2026 "DECISIVE finding" entry -- this is
     the output that matters for this model, not the mean).

Usage:
  python scripts/ig_partition_quantile.py \
      --config configs/partition/full_gnll_quantile_v2/fold0.yaml \
      --output experiments/figures/ig_quantile_v2_fold0 \
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

# cartopy is not installed in this venv / not in requirements.txt (CLAUDE.md:
# don't add deps without asking first) -- plot with plain matplotlib + a
# land/ocean contour from the data's own land_mask instead of cartopy's
# coastline feature. Ask the user if nicer cartopy maps are wanted later.

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_recall_v2_partition import (  # noqa: E402
    AREA_FRAC_THRESHOLD,
    FIGURES_DIR,
)
from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402
from src.utils.sampling import stratified_test_sample  # noqa: E402


def integrated_gradients_single_output(
    model_fn, x_spatial, x_temporal, n_steps=50, chunk_size=5
):
    """IG for a scalar-per-sample output. model_fn(xs, xt) must return
    shape (batch,) or (batch, 1) -- a SINGLE output column, never a
    multi-column tensor averaged blindly (see module docstring).

    Processes interpolation steps in chunks of `chunk_size` rather than
    all n_steps at once -- found Aug 21 2026 that materializing all steps
    simultaneously OOMs a 40GB A100 on the very first sample: each
    interpolation step multiplies through window_size in the CNN
    encoder (x_flat is (batch*window, n_vars, H, W)), so n_steps=50 x
    window=60 = 3000 images processed at once, not just n_steps=50 as a
    naive "batch of 50" estimate assumed. Chunking keeps peak memory
    bounded by chunk_size instead of n_steps, standard practice for IG
    on memory-constrained models -- accumulates the gradient sum exactly
    the same as the unchunked version (backward() on a chunk's sum, then
    move to the next chunk), not an approximation.
    """
    baseline = torch.zeros_like(x_spatial)
    B = x_spatial.shape[0]
    diff = x_spatial - baseline
    alphas_full = torch.linspace(0, 1, n_steps, device=x_spatial.device)

    grad_sum = torch.zeros_like(x_spatial)
    for start in range(0, n_steps, chunk_size):
        alphas = alphas_full[start : start + chunk_size].view(-1, 1, 1, 1, 1)
        steps = alphas.shape[0]
        interp = baseline.unsqueeze(0) + alphas.unsqueeze(1) * diff.unsqueeze(0)
        interp = interp.view(steps * B, *x_spatial.shape[1:]).requires_grad_(True)
        xt_rep = (
            x_temporal.unsqueeze(0)
            .expand(steps, -1, -1, -1)
            .reshape(steps * B, *x_temporal.shape[1:])
        )

        # cuDNN's fused LSTM kernel doesn't support backward() while the
        # model is in eval() mode ("cudnn RNN backward can only be called
        # in training mode") -- disable cudnn for just this forward/backward
        # (falls back to the generic, slower-but-backward-capable kernel),
        # matching the workaround already present in src/xai/
        # integrated_gradients.py's _integrated_gradients_forward().
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        out = model_fn(interp, xt_rep)
        out = out.reshape(-1)
        out.sum().backward()
        torch.backends.cudnn.enabled = prev_cudnn

        grads = interp.grad.view(steps, B, *x_spatial.shape[1:])
        grad_sum += grads.sum(0).detach()
        del interp, out, grads
        if x_spatial.is_cuda:
            torch.cuda.empty_cache()

    ig = (diff * (grad_sum / n_steps)).detach().cpu()
    return ig  # (B, window, n_vars, lat, lon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=300,
        help="cap on test samples processed -- IG is expensive (n_steps "
        "backward passes per sample), no need to run all ~2900 for a "
        "first real result.",
    )
    parser.add_argument(
        "--heads",
        default="mean,quantile",
        help="comma-separated subset of {mean,quantile,diff} to compute IG "
        "for. 'diff' = q_pred - y_hat_mean, isolating the gradient "
        "direction that differentiates the quantile head from the mean "
        "head directly (see known_issues.md #51 -- the two heads' "
        "population-averaged maps computed separately are 0.998-0.999 "
        "correlated and don't show this on their own).",
    )
    parser.add_argument(
        "--stratify_mhw",
        action="store_true",
        help="split the population average into two groups -- target day IS "
        "a real MHW day (def2, area_frac >= 0.05, same ground truth as "
        "eval_recall_v2_partition.py) vs. is NOT -- instead of one pooled "
        "average over all sampled days. User's explicit request (Aug 24 "
        "2026): the unconditional diff map answers 'what matters on an "
        "average day', not 'what matters on MHW days' specifically. Same "
        "IG cost as without this flag -- same samples, same n_steps, just "
        "accumulated into 2 buckets by outcome instead of 1.",
    )
    args = parser.parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    assert heads and set(heads) <= {
        "mean",
        "quantile",
        "diff",
    }, f"--heads must be a non-empty subset of {{mean,quantile,diff}}, got {heads!r}"

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
    model = lm.model

    def mean_head_fn(xs, xt):
        y_hat, _ = model.forward_with_quantile(xs, xt)
        return y_hat[:, 0]  # mean column only -- never averaged with log_var

    def quantile_head_fn(xs, xt):
        _, q_pred = model.forward_with_quantile(xs, xt)
        return q_pred[:, 0]

    def diff_head_fn(xs, xt):
        y_hat, q_pred = model.forward_with_quantile(xs, xt)
        return q_pred[:, 0] - y_hat[:, 0]

    head_fns = {
        "mean": mean_head_fn,
        "quantile": quantile_head_fn,
        "diff": diff_head_fn,
    }
    head_labels = {
        "mean": "mean_head",
        "quantile": "quantile_head",
        "diff": "diff_head",
    }
    active_fns = {h: head_fns[h] for h in heads}
    print(f"Computing IG for heads: {heads}", flush=True)

    variables = cfg["variables"]
    n_vars = len(variables)
    # Bug fixed Aug 21 2026 (user's methodological review): test_indices[
    # :max_samples] took the FIRST N in chronological order, not a
    # representative draw -- stratified_kfold's test years are non-
    # consecutive (e.g. fold0: 1985,1991,2000,...), so this concentrated
    # 299/300 samples in 1985 alone. Now stratified proportionally by
    # target year, reproducible (seed=42). See src/utils/sampling.py and
    # known_issues.md.
    sample_idx = stratified_test_sample(test_indices, full_ds, args.max_samples)
    print(
        f"Running IG on {len(sample_idx)} samples (capped from {len(test_indices)}, stratified by year)",
        flush=True,
    )

    conditions = ["mhw", "nonmhw"] if args.stratify_mhw else ["all"]
    if args.stratify_mhw:
        area_frac = np.load(FIGURES_DIR / "area_frac_timeseries.npy")
        print(
            f"Stratifying by def2 ground truth (area_frac >= {AREA_FRAC_THRESHOLD})",
            flush=True,
        )

    def sample_condition(idx):
        if not args.stratify_mhw:
            return "all"
        target_idx = idx + full_ds.window_size - 1 + full_ds.lead_time
        return "mhw" if area_frac[target_idx] >= AREA_FRAC_THRESHOLD else "nonmhw"

    H = W = None
    sums = {c: {h: None for h in heads} for c in conditions}
    counts = {c: 0 for c in conditions}

    for i, idx in enumerate(sample_idx):
        xs, xt, _ = full_ds[idx]
        xs = xs.unsqueeze(0).float().to(device)
        xt = xt.unsqueeze(0).float().to(device)
        if H is None:
            H, W = xs.shape[-2], xs.shape[-1]
            for c in conditions:
                for h in heads:
                    sums[c][h] = torch.zeros(n_vars, H, W)

        cond = sample_condition(idx)
        for h in heads:
            ig = integrated_gradients_single_output(
                active_fns[h], xs, xt, n_steps=args.n_steps
            )
            sums[cond][h] += ig[0].mean(0).cpu()
        counts[cond] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sample_idx)}  (counts so far: {counts})", flush=True)

    print(f"Final sample counts per condition: {counts}", flush=True)
    for c in conditions:
        if counts[c] == 0:
            print(
                f"  WARNING: condition '{c}' got 0 samples -- skipping save/plot for it",
                flush=True,
            )

    mean_ig = {
        c: {h: (sums[c][h] / counts[c]).numpy() for h in heads}
        for c in conditions
        if counts[c] > 0
    }
    suffix = {c: ("" if c == "all" else f"_{c}") for c in conditions}
    for c in mean_ig:
        for h in heads:
            np.save(out_dir / f"ig_{head_labels[h]}{suffix[c]}.npy", mean_ig[c][h])
    print(
        f"Saved {', '.join(f'ig_{head_labels[h]}{suffix[c]}.npy' for c in mean_ig for h in heads)}, "
        f"shape={mean_ig[conditions[0]][heads[0]].shape}",
        flush=True,
    )

    import xarray as xr

    nc = xr.open_dataset(DATA_FILE)
    lats = nc.lat.values
    lons = nc.lon.values
    land_mask = nc["land_mask"].values if "land_mask" in nc else None  # 1=ocean
    land_mask_tbottom = (
        nc["land_mask_tbottom"].values if "land_mask_tbottom" in nc else None
    )  # 1=ocean, ptho_bot's own (different) land/ocean boundary -- known_issues.md #2
    nc.close()

    # Land should be greyed out (masked to NaN) only for variables the model
    # itself treats as ocean-only (config's `ocean_variables`, e.g. ptho_bot --
    # zeroed on land at training/inference time, see dataset.py). Atmospheric
    # variables (u10/v10/msl/ssr) are never masked -- ERA5 wind/pressure/
    # radiation are real, physically valid over land (confirmed directly
    # against the raw data, not assumed -- see docs/narrative.md's Aug 21
    # 2026 XAI interpretation entry), so plotting them unmasked is correct,
    # not an oversight.
    ocean_variables = set(cfg.get("ocean_variables", []))
    plot_land_masks = {}
    for var in ocean_variables:
        if var == "ptho_bot" and land_mask_tbottom is not None:
            plot_land_masks[var] = land_mask_tbottom
        else:
            plot_land_masks[var] = land_mask

    for c in mean_ig:
        for h in heads:
            label, data_all = head_labels[h], mean_ig[c][h]
            for i, var in enumerate(variables):
                fig, ax = plt.subplots(figsize=(10, 5))
                data = data_all[i].copy()
                if var in plot_land_masks and plot_land_masks[var] is not None:
                    data[plot_land_masks[var] == 0] = np.nan  # 0 = land
                    ax.set_facecolor("lightgray")
                vmax = np.nanpercentile(np.abs(data), 98)
                im = ax.pcolormesh(
                    lons,
                    lats,
                    data,
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                    shading="auto",
                )
                if land_mask is not None:
                    ax.contour(
                        lons, lats, land_mask, levels=[0.5], colors="k", linewidths=0.6
                    )
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                cond_label = "" if c == "all" else f" — {c} days only"
                title = f"Mean signed IG ({label}) — {var}{cond_label}  (n={counts[c]} test samples)"
                ax.set_title(title, fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.7, label="Mean signed IG")
                plt.tight_layout()
                fname = out_dir / f"ig_{label}{suffix[c]}_{var}.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"  Saved {fname}", flush=True)

    print(f"\nDone. All maps in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
