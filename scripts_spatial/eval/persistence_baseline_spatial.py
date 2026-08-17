"""
persistence_baseline_spatial.py — Pixel-wise persistence baseline for spatial forecast.

For each pixel, computes Pearson r between to_anom(t) and to_anom(t + lead_time)
over the test set of each fold. Saves:
  - persistence_r_map.npy   : (H, W) mean r across folds
  - persistence_r_map.png   : map figure

Usage:
  python eval/persistence_baseline_spatial.py --lead 7 --output eval/figures/
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


DATA_FILE = "/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc"
N_FOLDS = 5


def get_test_years(unique_years, fold, n_folds=5):
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(unique_years)
    groups = np.array_split(shuffled, n_folds)
    return set(groups[fold].tolist())


def pearson_r_map(x, y):
    """x, y: (N, H, W) → (H, W) Pearson r."""
    x_m = x - x.mean(0)
    y_m = y - y.mean(0)
    cov = (x_m * y_m).mean(0)
    r = cov / (x.std(0) * y.std(0) + 1e-8)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=int, default=7)
    parser.add_argument("--output", default="eval/figures")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_FILE}...")
    ds = xr.open_dataset(DATA_FILE)
    to_anom = torch.tensor(ds["to_anom"].values, dtype=torch.float32)  # (T, H, W)
    # Convention: land_mask=1 means OCEAN in merged_daily.nc
    lm = torch.tensor(
        ~ds["land_mask"].values.astype(bool), dtype=torch.bool
    )  # True = land
    years = ds.time.dt.year.values
    lats = ds.lat.values
    lons = ds.lon.values
    ds.close()

    ocean_mask = ~lm
    unique_yrs = np.sort(np.unique(years))
    window_size = 60  # must match training config

    r_maps = []
    for fold in range(N_FOLDS):
        test_yrs = get_test_years(unique_yrs, fold)
        # samples: index i → target at i + window_size - 1 + lead_time
        # input (persistence) = to_anom at i + window_size - 1
        total = len(years) - window_size - args.lead + 1
        target_years_arr = np.array(
            [int(years[i + window_size - 1 + args.lead]) for i in range(total)]
        )
        test_idx = [i for i in range(total) if target_years_arr[i] in test_yrs]

        past_frames = to_anom[[i + window_size - 1 for i in test_idx]]  # (N, H, W)
        target_frames = to_anom[[i + window_size - 1 + args.lead for i in test_idx]]

        # Zero land
        past_frames[:, lm] = 0.0
        target_frames[:, lm] = 0.0

        r = pearson_r_map(past_frames, target_frames)
        r[:][lm] = float("nan")
        r_maps.append(r)
        print(
            f"  Fold {fold}: n_test={len(test_idx)}  "
            f"mean_r_ocean={r[ocean_mask].nanmean() if hasattr(r[ocean_mask], 'nanmean') else float(np.nanmean(r[ocean_mask].numpy())):.4f}"
        )

    r_mean = torch.stack(r_maps).nanmean(0).numpy()  # (H, W)
    np.save(out_dir / f"persistence_r_map_lead{args.lead}.npy", r_mean)

    ocean_r = r_mean[ocean_mask.numpy()]
    print(f"\nMean persistence r (ocean): {np.nanmean(ocean_r):.4f}")
    print(f"Median persistence r (ocean): {np.nanmedian(ocean_r):.4f}")

    # Plot
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        im = ax.pcolormesh(
            lons,
            lats,
            r_mean,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.pcolormesh(lons, lats, r_mean, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
    ax.set_title(
        f"Persistence baseline: Pearson r(t, t+{args.lead}d) — 5-fold mean", fontsize=13
    )
    plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    plt.tight_layout()
    fig.savefig(
        out_dir / f"persistence_r_map_lead{args.lead}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved persistence_r_map_lead{args.lead}.png")


if __name__ == "__main__":
    main()
