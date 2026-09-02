"""
occlusion_ptho_bot_sanity_check.py — Aug 21 2026, user-requested cross-check
for known_issues.md #52 (ptho_bot's coastal IG "signal" turned out to be
predominantly a land-masking edge artifact under gradient-based IG).

This uses a fundamentally different attribution method: occlusion
(replace a region with a fixed baseline, measure the actual forward-pass
output delta). It has a different bias structure than IG -- it doesn't
depend on the local gradient shape at the coastal discontinuity, only on
the model's real functional sensitivity to that region. If the same
coastal decay-by-distance pattern shows up here too, that corroborates
the land-masking-artifact explanation with an independent method. If it
doesn't, the original IG coastal spike was more specifically a gradient
artifact rather than a genuine forward-pass dependency.

Runs on the ALREADY-COMMITTED full_gnll_quantile_v2 fold0 checkpoint (not
the land_fill_mode=nearest retrain in flight) -- doesn't need to wait for
that job.

Two perturbation experiments, both on ptho_bot only (the only masked
variable), baseline=0 (same all-zero baseline IG uses, since that IS the
model's actual land-fill value under land_fill_mode="zero"):
  1. Distance-to-coast bins (same bins/method as known_issues.md #52):
     zero out ALL of ptho_bot across the full time window for pixels in
     one distance bin at a time, measure |output delta|.
  2. NS-box open-water vs rest-of-domain open-water (same >5px-from-coast
     restriction used in #52's enrichment check): 2 region perturbations.

Usage:
  python scripts/occlusion_ptho_bot_sanity_check.py \
      --config configs/partition/full_gnll_quantile_v2/fold0.yaml \
      --output experiments/figures/xai_integrated_gradients/occlusion_sanity_fold0 \
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
import yaml
from scipy import ndimage

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.datamodule import LazyDataModule  # noqa: E402
from src.models.cnn_lstm import CNNLightningModule, CNNLSTMModel  # noqa: E402
from src.utils.checkpoints import best_ckpt, load_model_config  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402
from src.utils.sampling import stratified_test_sample  # noqa: E402

# Same bin edges as known_issues.md #52's distance-to-coast quantification.
DIST_BIN_EDGES = [0, 2, 3, 4, 5, 6, 7, 8, 9, np.inf]
DIST_BIN_LABELS = [
    "1-2px",
    "3px",
    "4px",
    "5px",
    "6px",
    "7px",
    "8px",
    "9px",
    ">9px",
]

_NS_LAT = slice(100, 127)
_NS_LON = slice(150, 187)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    dm = LazyDataModule(args.config)
    dm.setup()
    full_ds = dm.test_dataset.dataset
    test_indices = dm.test_dataset.indices
    print(f"Test samples available: {len(test_indices)}", flush=True)

    cfg = yaml.safe_load(open(args.config))
    run_dir = Path(cfg["output_dir"])
    model_kwargs = load_model_config(run_dir, fallback_cfg=cfg)
    assert model_kwargs.get(
        "quantile_head", False
    ), "expected quantile_head=True -- wrong checkpoint/config, do not proceed silently."
    inner = CNNLSTMModel(**model_kwargs)
    ckpt = best_ckpt(run_dir / "checkpoints")
    print(f"Loading {ckpt.name}", flush=True)
    lm = CNNLightningModule.load_from_checkpoint(
        str(ckpt), model=inner, strict=True, map_location=device
    )
    lm.eval().to(device)
    model = lm.model

    variables = cfg["variables"]
    assert "ptho_bot" in variables, "this check is specific to ptho_bot"
    ptho_idx = variables.index("ptho_bot")
    print(f"ptho_bot is variable index {ptho_idx} of {variables}", flush=True)

    import xarray as xr

    nc = xr.open_dataset(DATA_FILE)
    lats = nc.lat.values
    lons = nc.lon.values
    land_mask_tbottom = nc["land_mask_tbottom"].values  # 1=ocean
    land_mask = nc["land_mask"].values if "land_mask" in nc else None
    nc.close()

    is_ocean = land_mask_tbottom == 1
    dist = ndimage.distance_transform_edt(is_ocean)  # 0 on land

    dist_bins = []
    for lo, hi in zip(DIST_BIN_EDGES[:-1], DIST_BIN_EDGES[1:]):
        m = is_ocean & (dist > lo) & (dist <= hi)
        dist_bins.append(m)
        print(f"  bin {lo}-{hi}px: {m.sum()} ocean pixels", flush=True)

    open_water = is_ocean & (dist > 5)
    ns_box_mask = np.zeros_like(is_ocean)
    ns_box_mask[_NS_LAT, _NS_LON] = True
    ns_open_water = open_water & ns_box_mask
    rest_open_water = open_water & ~ns_box_mask
    print(
        f"NS-box open-water (>5px): {ns_open_water.sum()} px, "
        f"rest-of-domain open-water: {rest_open_water.sum()} px",
        flush=True,
    )

    regions = {DIST_BIN_LABELS[i]: dist_bins[i] for i in range(len(dist_bins))}
    regions["ns_box_open_water"] = ns_open_water
    regions["rest_open_water"] = rest_open_water
    region_names = list(regions.keys())
    region_masks_t = {
        name: torch.from_numpy(m.astype(np.float32)).to(device)
        for name, m in regions.items()
    }

    def mean_head_fn(xs, xt):
        y_hat, _ = model.forward_with_quantile(xs, xt)
        return y_hat[:, 0]

    def quantile_head_fn(xs, xt):
        _, q_pred = model.forward_with_quantile(xs, xt)
        return q_pred[:, 0]

    head_fns = {"mean_head": mean_head_fn, "quantile_head": quantile_head_fn}

    # Bug fixed Aug 21 2026 -- see ig_partition_quantile.py's identical fix
    # and src/utils/sampling.py: test_indices[:max_samples] concentrated
    # 299/300 samples in a single year (1985) for fold0. Now stratified
    # proportionally by target year, reproducible (seed=42).
    sample_idx = stratified_test_sample(test_indices, full_ds, args.max_samples)
    print(
        f"Running occlusion on {len(sample_idx)} samples (stratified by year)",
        flush=True,
    )

    deltas = {h: {name: [] for name in region_names} for h in head_fns}

    with torch.no_grad():
        for i, idx in enumerate(sample_idx):
            xs, xt, _ = full_ds[idx]
            xs = xs.unsqueeze(0).float().to(device)
            xt = xt.unsqueeze(0).float().to(device)

            base_out = {h: fn(xs, xt).item() for h, fn in head_fns.items()}

            for name in region_names:
                mask = region_masks_t[name]  # (H, W), 1=occlude
                xs_pert = xs.clone()
                # zero ptho_bot across the whole time window for masked pixels
                keep = 1.0 - mask
                xs_pert[:, :, ptho_idx, :, :] = xs_pert[:, :, ptho_idx, :, :] * keep
                for h, fn in head_fns.items():
                    pert_out = fn(xs_pert, xt).item()
                    deltas[h][name].append(abs(pert_out - base_out[h]))

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(sample_idx)}", flush=True)

    print(
        "\n=== Mean |output delta| per region (occlusion attribution) ===", flush=True
    )
    summary = {}
    for h in head_fns:
        summary[h] = {}
        print(f"\n-- {h} --", flush=True)
        for name in region_names:
            vals = np.array(deltas[h][name])
            mean_d = vals.mean()
            summary[h][name] = mean_d
            print(f"  {name:20s}  mean|Δ|={mean_d:.6f}  n={len(vals)}", flush=True)

        dist_vals = [summary[h][lbl] for lbl in DIST_BIN_LABELS]
        decay = dist_vals[0] / dist_vals[-1] if dist_vals[-1] > 0 else float("inf")
        print(f"  Coastal decay ratio (1-2px / >9px): {decay:.2f}x", flush=True)

        ns_val = summary[h]["ns_box_open_water"]
        rest_val = summary[h]["rest_open_water"]
        ns_px = ns_open_water.sum()
        rest_px = rest_open_water.sum()
        ns_share = ns_val * ns_px / (ns_val * ns_px + rest_val * rest_px)
        pixel_share = ns_px / (ns_px + rest_px)
        enrichment = ns_share / pixel_share if pixel_share > 0 else float("nan")
        print(
            f"  NS-box open-water enrichment: {ns_share * 100:.1f}% of open-water "
            f"|Δ| vs {pixel_share * 100:.1f}% pixel-count share -> {enrichment:.2f}x",
            flush=True,
        )

    np.savez(
        out_dir / "occlusion_summary.npz",
        **{f"{h}_{name}": summary[h][name] for h in head_fns for name in region_names},
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DIST_BIN_LABELS))
    for h, color in zip(head_fns, ["#2166ac", "#e08214"]):
        y = [summary[h][lbl] for lbl in DIST_BIN_LABELS]
        ax.plot(x, y, "o-", label=h, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(DIST_BIN_LABELS)
    ax.set_xlabel("Distance to coast")
    ax.set_ylabel("Mean |output delta| (occlusion attribution)")
    ax.set_title(
        "ptho_bot occlusion sensitivity vs. distance to coast\n(cross-check for known_issues.md #52's IG-based coastal decay)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "occlusion_coastal_decay.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {fig_path}", flush=True)
    print(f"Saved {out_dir / 'occlusion_summary.npz'}", flush=True)


if __name__ == "__main__":
    main()
