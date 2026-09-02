"""
ig_coastal_decay_check.py — Aug 21 2026, item 1: does land_fill_mode=
nearest actually fix the ptho_bot coastal IG artifact (known_issues.md
#52)? Reproduces #52's exact methodology (distance-to-coast binning via
scipy.ndimage.distance_transform_edt on land_mask_tbottom, and the
NS-box-restricted-to-open-water enrichment calc, >5px from coast) on any
saved ig_mean_head.npy/ig_quantile_head.npy, so the committed
(land_fill_mode=zero) and land_fill (land_fill_mode=nearest) runs can be
compared directly, side by side. No GPU, reads already-saved .npy only.

Usage:
  python scripts/analysis/ig_coastal_decay_check.py --dir experiments/figures/xai_integrated_gradients/ig_quantile_v2_fold0 --label committed
  python scripts/analysis/ig_coastal_decay_check.py --dir experiments/figures/xai_integrated_gradients/ig_quantile_v2_landfill_fold0 --label landfill
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import DATA_FILE  # noqa: E402

PTHO_BOT_IDX = 0
BIN_EDGES = [(1, 2), (2, 3), (3, 5), (5, 9), (9, np.inf)]
BIN_LABELS = ["1-2px", "2-3px", "3-5px", "5-9px", ">9px"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    d = Path(args.dir)

    ds = xr.open_dataset(DATA_FILE)
    land_mask_tbottom = ds["land_mask_tbottom"].values  # 1=ocean, 0=land
    lats = ds["lat"].values
    lons = ds["lon"].values
    dist_to_land = distance_transform_edt(land_mask_tbottom)  # px, 0 on land

    ns_lat_i = (lats >= 50.0) & (lats <= 63.0)
    ns_lon_i = (lons >= -5.0) & (lons <= 13.0)
    ns_box_mask = np.zeros_like(land_mask_tbottom, dtype=bool)
    ns_box_mask[np.ix_(ns_lat_i, ns_lon_i)] = True

    open_water_mask = (land_mask_tbottom == 1) & (dist_to_land > 5)
    ns_open_water = open_water_mask & ns_box_mask
    n_ns, n_dom = ns_open_water.sum(), open_water_mask.sum()
    print(
        f"[{args.label}] open-water(>5px): NS-box n={n_ns}, domain n={n_dom}, "
        f"pixel share={n_ns / n_dom:.4f}",
        flush=True,
    )

    for head in ["mean", "quantile"]:
        f = d / f"ig_{head}_head.npy"
        if not f.exists():
            print(f"[{args.label}] {f} not found, skipping", flush=True)
            continue
        arr = np.load(f)  # (5, lat, lon)
        ig_ptho = np.abs(arr[PTHO_BOT_IDX])

        print(f"-- {args.label} / {head}_head --", flush=True)
        bin_means = {}
        for (lo, hi), lbl in zip(BIN_EDGES, BIN_LABELS):
            m = (land_mask_tbottom == 1) & (dist_to_land > lo) & (dist_to_land <= hi)
            bin_means[lbl] = ig_ptho[m].mean()
            print(f"  {lbl:8s} mean|IG|={bin_means[lbl]:.6e}  n={m.sum()}", flush=True)
        decay_ratio = bin_means["1-2px"] / bin_means[">9px"]
        print(f"  Coastal decay ratio (1-2px / >9px): {decay_ratio:.2f}x", flush=True)

        ns_share = ig_ptho[ns_open_water].sum() / ig_ptho[open_water_mask].sum()
        pixel_share = n_ns / n_dom
        enrichment = ns_share / pixel_share
        print(
            f"  NS-box open-water enrichment: {ns_share * 100:.1f}% of open-water "
            f"|IG| vs {pixel_share * 100:.1f}% pixel-count share -> {enrichment:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
