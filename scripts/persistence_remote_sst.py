"""
persistence_remote_sst.py — Non-local persistence baseline.

Tests whether the masked model (r=0.807) is just doing "remote SST persistence"
by computing how much skill you get from simple linear baselines using
remote Atlantic SST (everything outside the North Sea box).

Baselines computed (for the same kfold test splits as the masked experiment):
  1. r(remote_SST_1day,   NS_target)  — single-day remote SST at t
  2. r(remote_SST_60day_mean, NS_target) — 60-day mean remote SST (same window as model)
  3. r(NS_target_t, NS_target_t+7)    — local NS persistence (upper bound reference)

Usage:
  python eval/persistence_remote_sst.py
  python eval/persistence_remote_sst.py --output experiments/figures/
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

from src.utils.paths import (
    DATA_FILE as DATA_FILE_ENV,
)

DATA_FILE = DATA_FILE_ENV
N_FOLDS = 5
WINDOW = 60
LEAD = 7
# North Sea box indices (lat 50–63°N = idx 100:127, lon -5–13°E = idx 150:187)
NS_LAT = slice(100, 127)
NS_LON = slice(150, 187)


def get_test_years(unique_years, fold, n_folds=5, seed=42):
    """Must match src/data/datamodule.py's real kfold branch EXACTLY, not a
    stand-in for it — this script's whole purpose is comparing against
    "the same kfold test splits as the masked experiment" (see module
    docstring). Previously used np.random.default_rng(0), a DIFFERENT RNG
    algorithm and seed than production's np.random.RandomState(seed=42) —
    known_issues.md #23/#1: RandomState and default_rng give different
    permutations from the same seed, so this computed test years that never
    matched what any real experiment was actually evaluated on. Fixed Aug
    20 2026. seed=42 matches every full_gnll*/full_mse_v2 config's `seed:`.
    """
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_years)
    fold_size = len(unique_years) // n_folds
    return set(shuffled[fold * fold_size : (fold + 1) * fold_size].tolist())


def pearson(a, b):
    r, p = pearsonr(a, b)
    return r, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/figures")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATA_FILE}...")
    ds = xr.open_dataset(DATA_FILE)
    # land_mask=1 means OCEAN in this file
    ocean = ds["land_mask"].values.astype(bool)  # True = ocean
    to_anom = ds["to_anom"].values.astype(np.float32)  # (T, H, W)
    target = ds["target"].values.astype(np.float32)  # (T,) NS basin mean
    years = ds.time.dt.year.values
    ds.close()

    # Remote SST mask: ocean pixels OUTSIDE the North Sea box
    ns_mask = np.zeros(ocean.shape, dtype=bool)
    ns_mask[NS_LAT, NS_LON] = True
    remote_mask = ocean & ~ns_mask  # True = remote ocean

    total = len(years) - WINDOW - LEAD + 1
    unique_yrs = np.sort(np.unique(years))
    target_yrs = np.array([int(years[i + WINDOW - 1 + LEAD]) for i in range(total)])

    r1_all, r60_all, r_ns_all = [], [], []

    print(f"\nRemote ocean pixels: {remote_mask.sum()}  |  NS pixels: {ns_mask.sum()}")
    print(f"Folds: {N_FOLDS}  |  Window: {WINDOW}d  |  Lead: {LEAD}d\n")

    for fold in range(N_FOLDS):
        test_yrs = get_test_years(unique_yrs, fold)
        test_idx = [i for i in range(total) if target_yrs[i] in test_yrs]

        y_true = target[np.array(test_idx) + WINDOW - 1 + LEAD]

        # 1. Single-day remote SST at last day of window (t + window - 1)
        last_day_idx = np.array(test_idx) + WINDOW - 1
        remote_1day = np.array(
            [np.nanmean(to_anom[t][remote_mask]) for t in last_day_idx]
        )

        # 2. 60-day mean remote SST over full window
        remote_60day = np.array(
            [np.nanmean(to_anom[i : i + WINDOW][:, remote_mask]) for i in test_idx]
        )

        # 3. Local NS persistence: NS mean at t+window-1 → predicts NS target at t+window-1+lead
        ns_persist = np.array(
            [np.nanmean(to_anom[t][ns_mask & ocean]) for t in last_day_idx]
        )

        r1, p1 = pearson(remote_1day, y_true)
        r60, p60 = pearson(remote_60day, y_true)
        rns, pns = pearson(ns_persist, y_true)

        r1_all.append(r1)
        r60_all.append(r60)
        r_ns_all.append(rns)
        print(
            f"  Fold {fold} (n={len(test_idx)} samples):  "
            f"remote_1d r={r1:.4f}  remote_60d r={r60:.4f}  NS_persist r={rns:.4f}"
        )

    print(f"\n{'='*60}")
    print(
        f"Remote SST 1-day persistence:   r = {np.mean(r1_all):.4f} ± {np.std(r1_all):.4f}"
    )
    print(
        f"Remote SST 60-day mean:         r = {np.mean(r60_all):.4f} ± {np.std(r60_all):.4f}"
    )
    print(
        f"NS local persistence:           r = {np.mean(r_ns_all):.4f} ± {np.std(r_ns_all):.4f}"
    )
    print("\nMasked model (reference):       r = 0.807 ± 0.038")
    print(
        "\nIf masked model r >> remote baselines → model extracts real non-local signal."
    )
    print("If masked model r ≈ remote persistence → correlation only.")

    # Save as txt
    with open(out_dir / "persistence_remote_sst.txt", "w") as f:
        f.write(
            f"Remote SST 1-day persistence:   r = {np.mean(r1_all):.4f} ± {np.std(r1_all):.4f}\n"
        )
        f.write(
            f"Remote SST 60-day mean:         r = {np.mean(r60_all):.4f} ± {np.std(r60_all):.4f}\n"
        )
        f.write(
            f"NS local persistence:           r = {np.mean(r_ns_all):.4f} ± {np.std(r_ns_all):.4f}\n"
        )
        f.write("Masked model (reference):       r = 0.807 ± 0.038\n")
    print(f"\nSaved to {out_dir / 'persistence_remote_sst.txt'}")


if __name__ == "__main__":
    main()
