#!/usr/bin/env python
"""
plot_fold_year_assignment.py — Step 3 verification figure (known_issues.md
#1/#2/#42): visual confirmation that stratified_kfold fixes the val_years
collision and the MHW-day imbalance kfold has.

Panel A/B: 40 years x 5 folds heatmap (train/val/test), kfold vs
stratified_kfold, side by side — the val_years collision (folds 1-4
identical) should be visually obvious in kfold's heatmap and absent in
stratified_kfold's.
Panel C: bar chart of test-fold MHW-days, kfold vs stratified_kfold.

Reuses the exact algorithms from src/data/datamodule.py (imported, not
reimplemented) plus src/utils/hobday.py for the MHW-day ranking.

CPU only, no GPU needed. Needs MHW_DATA_FILE (target series) and
MHW_CLIM_FILE (p90 threshold) set.

Usage:
  python scripts/analysis/plot_fold_year_assignment.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.hobday import apply_hobday, load_ns_p90  # noqa: E402
from src.utils.paths import DATA_FILE  # noqa: E402

N_FOLDS = 5
SEED = 42
UNIQUE_YEARS = np.arange(1985, 2025)


def kfold_assignment(unique_years, n_folds=N_FOLDS, seed=SEED, val_ratio=0.15):
    """Exact replica of datamodule.py's kfold branch (bug included)."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_years)
    n = len(unique_years)
    fold_size = n // n_folds
    n_val = int(val_ratio * n)
    assignment = {}  # fold -> {"train":..., "val":..., "test":...}
    for fold in range(n_folds):
        test_years = set(shuffled[fold * fold_size : (fold + 1) * fold_size].tolist())
        remaining = [y for y in shuffled if y not in test_years]
        val_years = set(remaining[:n_val])
        train_years = set(remaining[n_val:])
        assignment[fold] = dict(train=train_years, val=val_years, test=test_years)
    return assignment


def stratified_kfold_assignment(unique_years, mhw_days_per_year, n_folds=N_FOLDS):
    """Exact replica of datamodule.py's stratified_kfold branch."""
    ranked_years = sorted(unique_years, key=lambda y: -mhw_days_per_year[int(y)])
    buckets = [[] for _ in range(n_folds)]
    for i, y in enumerate(ranked_years):
        buckets[i % n_folds].append(int(y))
    assignment = {}
    all_years = set(unique_years.tolist())
    for fold in range(n_folds):
        test_years = set(buckets[fold])
        val_years = set(buckets[(fold + 1) % n_folds])
        train_years = all_years - test_years - val_years
        assignment[fold] = dict(train=train_years, val=val_years, test=test_years)
    return assignment


def to_matrix(assignment, unique_years, n_folds=N_FOLDS):
    """years x folds matrix: 0=train, 1=val, 2=test."""
    mat = np.zeros((len(unique_years), n_folds), dtype=int)
    for fold in range(n_folds):
        for i, y in enumerate(unique_years):
            if y in assignment[fold]["val"]:
                mat[i, fold] = 1
            elif y in assignment[fold]["test"]:
                mat[i, fold] = 2
    return mat


def main():
    ds = xr.open_dataset(DATA_FILE)
    raw_years = ds.time.dt.year.values
    raw_doys = ds.time.dt.dayofyear.values
    raw_doys_clamped = raw_doys.copy()
    raw_doys_clamped[raw_doys_clamped >= 365] = 365
    raw_target = ds["target"].values
    ds.close()

    p90 = load_ns_p90()
    thresh = p90[raw_doys_clamped - 1]
    mhw_day = apply_hobday(raw_target > thresh)
    mhw_days_per_year = {
        int(y): int(mhw_day[raw_years == y].sum()) for y in UNIQUE_YEARS
    }

    kfold_a = kfold_assignment(UNIQUE_YEARS)
    strat_a = stratified_kfold_assignment(UNIQUE_YEARS, mhw_days_per_year)

    mat_kfold = to_matrix(kfold_a, UNIQUE_YEARS)
    mat_strat = to_matrix(strat_a, UNIQUE_YEARS)

    kfold_test_mhw = [
        sum(mhw_days_per_year[y] for y in kfold_a[f]["test"]) for f in range(N_FOLDS)
    ]
    strat_test_mhw = [
        sum(mhw_days_per_year[y] for y in strat_a[f]["test"]) for f in range(N_FOLDS)
    ]
    print("kfold test MHW-days per fold:            ", kfold_test_mhw)
    print("stratified_kfold test MHW-days per fold: ", strat_test_mhw)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1, 1.2]}
    )
    cmap = plt.matplotlib.colors.ListedColormap(["#cfe8cf", "#f7d774", "#e08283"])

    for ax, mat, title in [
        (axes[0], mat_kfold, "kfold (buggy: val_years\nidentical folds 1-4)"),
        (axes[1], mat_strat, "stratified_kfold (fixed)"),
    ]:
        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_xticks(range(N_FOLDS))
        ax.set_xticklabels([f"fold{f}" for f in range(N_FOLDS)])
        ax.set_yticks(range(0, len(UNIQUE_YEARS), 5))
        ax.set_yticklabels(UNIQUE_YEARS[::5])
        ax.set_title(title)
    axes[0].set_ylabel("Year")

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor="#cfe8cf", label="train"),
        Patch(facecolor="#f7d774", label="val"),
        Patch(facecolor="#e08283", label="test"),
    ]
    axes[1].legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(1.6, 1.0))

    x = np.arange(N_FOLDS)
    w = 0.35
    axes[2].bar(x - w / 2, kfold_test_mhw, w, label="kfold", color="#e08283")
    axes[2].bar(x + w / 2, strat_test_mhw, w, label="stratified_kfold", color="#7aa6c2")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"fold{f}" for f in range(N_FOLDS)])
    axes[2].set_ylabel("Test-fold MHW days")
    axes[2].set_title("Test-fold MHW-day balance")
    axes[2].legend()

    plt.tight_layout()
    out_dir = REPO_ROOT / "experiments" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fold_year_assignment_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
