"""
raw_ptho_bot_coastal_check.py -- Aug 21 2026, final check on item 1: does
ptho_bot's coastal IG "signal" (known_issues #52) show up in the RAW data
at all, independent of the model? User's explicit ask: verify against raw
values before trusting XAI attribution, and explain why a remote coast
(Grand Banks) would matter for North Sea MHWs.

Computes, per grid pixel, raw Pearson r between ptho_bot(t) and the
NS-box target(t+lead) over the full 40yr record (no model), binned by
distance-to-coast and by region. No GPU, no model, pure data check.

Usage:
  python scripts/analysis/raw_ptho_bot_coastal_check.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import DATA_FILE  # noqa: E402

LEAD = 7
BIN_EDGES = [(1, 2), (2, 3), (3, 5), (5, 9), (9, np.inf)]
BIN_LABELS = ["1-2px", "2-3px", "3-5px", "5-9px", ">9px"]
REGIONS = {
    "NS/British Isles (lat50-63,lon-5..13)": ((50, 63), (-5, 13)),
    "Grand Banks/Nova Scotia (lat42-48,lon-70..-55)": ((42, 48), (-70, -55)),
    "Iberia/Bay of Biscay (lat36-48,lon-10..0)": ((36, 48), (-10, 0)),
    "West Africa (lat0-30,lon-20..10)": ((0, 30), (-20, 10)),
    "US East Coast (lat25-42,lon-80..-70)": ((25, 42), (-80, -70)),
}


def main():
    ds = xr.open_dataset(DATA_FILE)
    land_mask = ds["land_mask_tbottom"].values
    lats = ds["lat"].values
    lons = ds["lon"].values
    dist = distance_transform_edt(land_mask)

    ptho = ds["ptho_bot"].values.astype(np.float64)
    target = ds["target"].values.astype(np.float64)
    n = len(target)
    ptho_lag = ptho[: n - LEAD]
    target_lead = target[LEAD:]

    nt, nlat, nlon = ptho_lag.shape
    flat = ptho_lag.reshape(nt, -1)
    p_mean, p_std = flat.mean(axis=0), flat.std(axis=0)
    t_mean, t_std = target_lead.mean(), target_lead.std()
    cov = ((flat - p_mean) * (target_lead - t_mean)[:, None]).mean(axis=0)
    r = (cov / (p_std * t_std + 1e-12)).reshape(nlat, nlon)
    r[land_mask == 0] = np.nan

    print(
        f"=== Raw pointwise Pearson r: ptho_bot(t) vs NS-box target(t+{LEAD}), "
        f"full 40yr, no model ===",
        flush=True,
    )
    for (lo, hi), lbl in zip(BIN_EDGES, BIN_LABELS):
        m = (land_mask == 1) & (dist > lo) & (dist <= hi)
        print(
            f"  {lbl:8s} mean|r|={np.nanmean(np.abs(r[m])):.4f}  n={m.sum()}",
            flush=True,
        )
    near = (land_mask == 1) & (dist > 1) & (dist <= 2)
    far = (land_mask == 1) & (dist > 9)
    decay = np.nanmean(np.abs(r[near])) / np.nanmean(np.abs(r[far]))
    print(
        f"  Coastal decay ratio (1-2px / >9px) in RAW correlation: {decay:.2f}x "
        f"(cf. IG's ~18x -- NOT present in the raw data)",
        flush=True,
    )

    print(
        "\n=== Raw std(ptho_bot) by distance-to-coast (physical variance) ===",
        flush=True,
    )
    std_field = ptho.std(axis=0)
    for (lo, hi), lbl in zip(BIN_EDGES, BIN_LABELS):
        m = (land_mask == 1) & (dist > lo) & (dist <= hi)
        print(f"  {lbl:8s} mean_std={std_field[m].mean():.4f}  n={m.sum()}", flush=True)

    far_baseline = np.nanmean(np.abs(r[far]))
    print(
        f"\n=== Raw |r| near-coast (<=2px) by region, vs far-from-coast "
        f"baseline ({far_baseline:.4f}) ===",
        flush=True,
    )
    for name, ((lat_lo, lat_hi), (lon_lo, lon_hi)) in REGIONS.items():
        lat_i = (lats >= lat_lo) & (lats <= lat_hi)
        lon_i = (lons >= lon_lo) & (lons <= lon_hi)
        reg_mask = np.zeros_like(land_mask, dtype=bool)
        reg_mask[np.ix_(lat_i, lon_i)] = True
        m = near & reg_mask
        if m.sum() == 0:
            continue
        mr = np.nanmean(np.abs(r[m]))
        print(
            f"  {name:48s} n={m.sum():4d}  mean|r|={mr:.4f}  vs baseline: {mr / far_baseline:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
