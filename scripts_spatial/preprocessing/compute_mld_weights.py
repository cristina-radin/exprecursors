"""
compute_mld_weights.py — Precompute monthly MLD-based loss weights for physics loss.

Reads monthly MLD from ICON-COAST (140×201, lat 0.25–69.75°),
regrids to SST grid (141×201, lat 0.0–70.0°), applies ocean mask,
and computes normalized inverse-MLD weights.

Output: mld_monthly_weights.npy  shape (12, 141, 201)
  w[m, i, j] = 1 / MLD_clim[m, i, j]  (ocean pixels, normalized per month)
  w[m, i, j] = 0                        (land pixels)

Weight interpretation: thin MLD → high weight (surface heats fast, MHW risk).
Mean ocean weight = 1.0 per month so total loss magnitude matches plain MSE.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

MLD_FILE = "/p/project1/hai_1127/inputs/monthly/raw_data/zos_mld_to.nc"
MASK_FILE = "/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc"
OUT_FILE = Path(__file__).parent / "mld_monthly_weights.npy"
MIN_MLD = 5.0  # meters — floor to avoid w → ∞ in very shallow pixels


def main():
    # ── Load MLD and compute monthly climatology ─────────────────────────────
    print("Loading MLD from ICON-COAST...")
    ds_mld = xr.open_dataset(MLD_FILE)
    mld = ds_mld["mld"].values  # (T=1020, H=140, W=201)
    months = ds_mld.time.dt.month.values
    mld_lat = ds_mld.lat.values  # 0.25, 0.75, ..., 69.75 (140 values)
    ds_mld.close()

    # Monthly climatology (12, 140, 201)
    mld_clim = np.zeros((12, 140, 201), dtype=np.float32)
    for m in range(1, 13):
        mld_clim[m - 1] = np.nanmean(mld[months == m], axis=0)
    print(
        f"  MLD climatology shape: {mld_clim.shape}  "
        f"range=[{mld_clim.min():.1f}, {mld_clim.max():.1f}] m"
    )

    # ── Regrid lat 140→141 ───────────────────────────────────────────────────
    sst_lat = np.arange(0.0, 70.5, 0.5)  # 0.0, 0.5, ..., 70.0  (141 values)
    f_interp = interp1d(
        mld_lat,
        mld_clim,
        axis=1,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    mld_regridded = f_interp(sst_lat).astype(np.float32)  # (12, 141, 201)
    mld_regridded = np.clip(mld_regridded, a_min=MIN_MLD, a_max=None)
    print(f"  Regridded shape: {mld_regridded.shape}")

    # ── Load ocean mask ───────────────────────────────────────────────────────
    print("Loading ocean mask from merged_daily.nc...")
    ds_mask = xr.open_dataset(MASK_FILE)
    # Convention: land_mask=1 → ocean in this file
    ocean_mask = ds_mask["land_mask"].values.astype(bool)  # (141, 201) True=ocean
    ds_mask.close()
    print(f"  Ocean pixels: {ocean_mask.sum()}  Land pixels: {(~ocean_mask).sum()}")

    # ── Compute inverse-MLD weights, normalize, zero land ────────────────────
    weights = np.zeros((12, 141, 201), dtype=np.float32)
    for m in range(12):
        w = 1.0 / mld_regridded[m]  # thin MLD → high weight
        w[~ocean_mask] = 0.0  # land → 0
        w_mean = w[ocean_mask].mean()
        w = w / w_mean  # normalize: mean ocean weight = 1
        weights[m] = w
        ns_w = w[100:127, 150:187][ocean_mask[100:127, 150:187]].mean()
        print(
            f"  Month {m+1:2d}: MLD NS={mld_regridded[m,100:127,150:187][ocean_mask[100:127,150:187]].mean():.1f}m  "
            f"weight NS mean={ns_w:.3f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT_FILE), weights)
    print(f"\nSaved: {OUT_FILE}  shape={weights.shape}  dtype={weights.dtype}")


if __name__ == "__main__":
    main()
