"""
persistence_remote_sst.py — Non-local persistence baseline.

Originally: tests whether the local/remote-partition experiment's masked
model was just doing "remote SST persistence" (r=0.807 reference, a
DIFFERENT experiment from the current full_gnll_quantile_v2 work).

Paso 7 (Aug 21 2026): repurposed as the plan's persistence-baseline gate
for the CURRENT committed model (full_gnll_quantile_v2, split_mode=
stratified_kfold) -- "comparar skill de los modelos contra persistencia
lag-7 con los anos de test correctos". Split-year computation extended
to support stratified_kfold (get_test_years_stratified, exact replica of
src/data/datamodule.py's branch / scripts/analysis/plot_fold_year_
assignment.py's already-verified stratified_kfold_assignment()) --
get_test_years (kfold) kept unchanged/available for the old comparison,
not removed, per known_issues.md #1's "never silently change a split
already cited" convention.

Baselines computed (for the actual full_gnll_quantile_v2 test splits):
  1. r(remote_SST_1day,   NS_target)  — single-day remote SST at t
  2. r(remote_SST_60day_mean, NS_target) — 60-day mean remote SST (same window as model)
  3. r(NS_target_t, NS_target_t+7)    — local NS lag-7 persistence (the actual gate)

Usage:
  python scripts/persistence_remote_sst.py --split_mode stratified_kfold
  python scripts/persistence_remote_sst.py --split_mode kfold  # old comparison
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, for `import src...`

from src.utils.paths import (  # noqa: E402
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


def _mhw_days_per_year_raw(target, years, unique_years, doys):
    """Same ranking input as src/data/datamodule.py's stratified_kfold
    branch, computed directly from the raw arrays this script already
    loads (no LazyDataset needed here)."""
    from src.utils.hobday import apply_hobday, load_ns_p90

    p90 = load_ns_p90()
    doys_clamped = doys.copy()
    doys_clamped[doys_clamped >= 365] = 365
    thresh_per_day = p90[doys_clamped - 1]
    mhw_day_bool = apply_hobday(target > thresh_per_day)
    return {int(y): int(mhw_day_bool[years == y].sum()) for y in unique_years}


def get_test_years_stratified(unique_years, fold, mhw_days_per_year, n_folds=5):
    """Exact replica of src/data/datamodule.py's stratified_kfold branch /
    scripts/analysis/plot_fold_year_assignment.py's already-verified
    stratified_kfold_assignment() -- round-robin buckets over years ranked
    by descending MHW-day count. test_years = bucket[fold]."""
    ranked_years = sorted(unique_years, key=lambda y: -mhw_days_per_year[int(y)])
    buckets = [[] for _ in range(n_folds)]
    for i, y in enumerate(ranked_years):
        buckets[i % n_folds].append(int(y))
    return set(buckets[fold])


def pearson(a, b):
    r, p = pearsonr(a, b)
    return r, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/figures/step7_persistence")
    parser.add_argument(
        "--split_mode",
        default="stratified_kfold",
        choices=["stratified_kfold", "kfold"],
        help="stratified_kfold matches the committed full_gnll_quantile_v2 model "
        "(default). kfold reproduces the old comparison against the pre-Paso-3 "
        "local/remote-partition experiment.",
    )
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
    doys = ds.time.dt.dayofyear.values
    ds.close()

    # Remote SST mask: ocean pixels OUTSIDE the North Sea box
    ns_mask = np.zeros(ocean.shape, dtype=bool)
    ns_mask[NS_LAT, NS_LON] = True
    remote_mask = ocean & ~ns_mask  # True = remote ocean

    total = len(years) - WINDOW - LEAD + 1
    unique_yrs = np.sort(np.unique(years))
    target_yrs = np.array([int(years[i + WINDOW - 1 + LEAD]) for i in range(total)])

    from src.utils.hobday import load_ns_p90

    p90 = load_ns_p90()
    doys_clamped_full = doys.copy()
    doys_clamped_full[doys_clamped_full >= 365] = 365
    thresh_full = p90[doys_clamped_full - 1]

    mhw_days_per_year = None
    if args.split_mode == "stratified_kfold":
        mhw_days_per_year = _mhw_days_per_year_raw(target, years, unique_yrs, doys)

    figures_dir = Path(__file__).parent.parent / "experiments" / "figures"
    area_frac_path = figures_dir / "area_frac_timeseries.npy"
    area_frac_full = np.load(area_frac_path) if area_frac_path.exists() else None
    if area_frac_full is None:
        print(
            f"WARNING: {area_frac_path} not found -- def2 (pixel+area) exceedance "
            "detection for persistence will be skipped, only def1 (basin-mean) computed."
        )

    r1_all, r60_all, r_ns_all = [], [], []
    all_y_true, all_ns_persist, all_thresh, all_area_frac = [], [], [], []

    print(f"\nRemote ocean pixels: {remote_mask.sum()}  |  NS pixels: {ns_mask.sum()}")
    print(
        f"Folds: {N_FOLDS}  |  Window: {WINDOW}d  |  Lead: {LEAD}d  |  split_mode: {args.split_mode}\n"
    )

    for fold in range(N_FOLDS):
        if args.split_mode == "stratified_kfold":
            test_yrs = get_test_years_stratified(unique_yrs, fold, mhw_days_per_year)
        else:
            test_yrs = get_test_years(unique_yrs, fold)
        test_idx = [i for i in range(total) if target_yrs[i] in test_yrs]

        target_idx = np.array(test_idx) + WINDOW - 1 + LEAD
        y_true = target[target_idx]

        # 1. Single-day remote SST at last day of window (t + window - 1)
        last_day_idx = np.array(test_idx) + WINDOW - 1
        remote_1day = np.array(
            [np.nanmean(to_anom[t][remote_mask]) for t in last_day_idx]
        )

        # 2. 60-day mean remote SST over full window
        remote_60day = np.array(
            [np.nanmean(to_anom[i : i + WINDOW][:, remote_mask]) for i in test_idx]
        )

        # 3. Local NS persistence: NS mean at t+window-1 -> predicts NS target at t+window-1+lead
        ns_persist = np.array(
            [np.nanmean(to_anom[t][ns_mask & ocean]) for t in last_day_idx]
        )

        r1, p1 = pearson(remote_1day, y_true)
        r60, p60 = pearson(remote_60day, y_true)
        rns, pns = pearson(ns_persist, y_true)

        r1_all.append(r1)
        r60_all.append(r60)
        r_ns_all.append(rns)
        all_y_true.append(y_true)
        all_ns_persist.append(ns_persist)
        all_thresh.append(thresh_full[target_idx])
        if area_frac_full is not None:
            all_area_frac.append(area_frac_full[target_idx])
        print(
            f"  Fold {fold} (n={len(test_idx)} samples, test_years={sorted(test_yrs)}):  "
            f"remote_1d r={r1:.4f}  remote_60d r={r60:.4f}  NS_persist r={rns:.4f}"
        )

    y_true_c = np.concatenate(all_y_true)
    ns_persist_c = np.concatenate(all_ns_persist)
    thresh_c = np.concatenate(all_thresh)

    # Exceedance detection: does lag-7 persistence alone (no model at all)
    # flag the same MHW days our committed model is evaluated on? This is
    # the real "does the model add anything over trivial persistence" gate
    # for the precursor-detection framing, not just point-forecast r.
    ext1 = y_true_c > thresh_c
    n1 = int(ext1.sum())
    persist_recall = (
        (ns_persist_c[ext1] > thresh_c[ext1]).mean() if n1 else float("nan")
    )
    persist_flagged = ns_persist_c > thresh_c
    persist_precision = (persist_flagged & ext1).sum() / max(1, persist_flagged.sum())

    persist_recall2 = persist_precision2 = n2 = None
    if area_frac_full is not None:
        area_frac_c = np.concatenate(all_area_frac)
        ext2 = area_frac_c >= 0.05  # MedECC threshold, matches docs/narrative.md
        n2 = int(ext2.sum())
        persist_recall2 = (
            (ns_persist_c[ext2] > thresh_c[ext2]).mean() if n2 else float("nan")
        )
        persist_flagged2 = ns_persist_c > thresh_c
        persist_precision2 = (persist_flagged2 & ext2).sum() / max(
            1, persist_flagged2.sum()
        )

    print(f"\n{'='*60}")
    print(
        f"Remote SST 1-day persistence:   r = {np.mean(r1_all):.4f} +/- {np.std(r1_all):.4f}"
    )
    print(
        f"Remote SST 60-day mean:         r = {np.mean(r60_all):.4f} +/- {np.std(r60_all):.4f}"
    )
    print(
        f"NS lag-7 local persistence:     r = {np.mean(r_ns_all):.4f} +/- {np.std(r_ns_all):.4f}"
    )
    print(
        f"NS lag-7 persistence exceedance detection (def1, basin-mean, no model at all): "
        f"recall={persist_recall*100:.1f}%  precision={persist_precision*100:.1f}%  (n={n1})"
    )
    if n2 is not None:
        print(
            f"NS lag-7 persistence exceedance detection (def2, pixel+area>=0.05, no model at all): "
            f"recall={persist_recall2*100:.1f}%  precision={persist_precision2*100:.1f}%  (n={n2})"
        )
    print(
        "\nCommitted model, full_gnll_quantile_v2 (job 29426208, pooled 5 folds, Aug 21 2026):"
    )
    print("  mean head:      r = 0.87 (point forecast)")
    print(
        "  quantile head:  def1 recall=80.7% precision=27.8%  |  "
        "def2 recall=48.1% precision=84.9%  (see docs/narrative.md)"
    )
    print(
        "\nGate: model's quantile-head exceedance recall/precision must clearly beat "
        "naive lag-7 persistence's own recall/precision above for the model to be "
        "adding real precursor-detection value, not just riding persistence."
    )

    # Save as txt
    with open(out_dir / "persistence_remote_sst.txt", "w") as f:
        f.write(f"split_mode: {args.split_mode}\n")
        f.write(
            f"Remote SST 1-day persistence:   r = {np.mean(r1_all):.4f} +/- {np.std(r1_all):.4f}\n"
        )
        f.write(
            f"Remote SST 60-day mean:         r = {np.mean(r60_all):.4f} +/- {np.std(r60_all):.4f}\n"
        )
        f.write(
            f"NS lag-7 local persistence:     r = {np.mean(r_ns_all):.4f} +/- {np.std(r_ns_all):.4f}\n"
        )
        f.write(
            f"NS lag-7 persistence exceedance detection (def1): recall={persist_recall*100:.1f}%  "
            f"precision={persist_precision*100:.1f}%  (n={n1})\n"
        )
        if n2 is not None:
            f.write(
                f"NS lag-7 persistence exceedance detection (def2): recall={persist_recall2*100:.1f}%  "
                f"precision={persist_precision2*100:.1f}%  (n={n2})\n"
            )
        f.write(
            "Committed model full_gnll_quantile_v2: mean r=0.87 | quantile_head def1 "
            "recall=80.7%/precision=27.8%  def2 recall=48.1%/precision=84.9%\n"
        )
    print(f"\nSaved to {out_dir / 'persistence_remote_sst.txt'}")


if __name__ == "__main__":
    main()
