"""
PyTorch Lightning DataModule for the North Sea MHW dataset.

Main goal of this script: split the years into train/val/test folds so that
every test set holds a similar share of MHW days, which makes the event-based
metrics (POD, FAR, F1) comparable across folds. (The model itself never sees
the MHW labels; it is trained on the continuous target.)

Builds a stratified k-fold split: ranks calendar years by MHW-day count
(most first; ties by increasing year), deals them into n_folds buckets 
following this pattern: (0, 1, ..., n-1, then n-1, ..., 0, and so on), 
and uses one bucket as test, the next bucket as val, and the rest as train. 
This keeps MHW-day representation balanced across folds and gives every fold a
different val set. MHW days follow src/utils/hobday.py (Hobday et al. 2016) on the full contiguous
target series.

Purge at split boundaries: a sample i covers days
i .. i + window_size - 1 + lead_time, so two samples share days when
|i - j| <= window_size - 1 + lead_time. Test samples are never removed; val
samples that share days with a test sample are removed; train samples that
share days with a kept val sample or a test sample are removed. Embargo 0
(no extra gap beyond the shared days), a deliberate choice.
"""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.utils.hobday import (
    MAX_GAP,
    MIN_DURATION,
    PERCENTILE,
    REF_YEARS,
    SMOOTH_WIDTH,
    WINDOW_HALF_WIDTH,
    detect,
    noleap_doy,
    threshold,
)
from src.utils.paths import DATA_FILE

from .dataset import LazyDataset


def _mhw_days_per_year(full_ds, unique_years):
    """Count MHW days per calendar year in the full target series."""
    target = full_ds.target.numpy()
    years = full_ds.years
    doys = noleap_doy(full_ds.ds.time.dt.month.values, full_ds.ds.time.dt.day.values)
    thresh = threshold(
        target,
        years,
        doys,
        ref_years=REF_YEARS,
        window_half_width=WINDOW_HALF_WIDTH,
        percentile=PERCENTILE,
        smooth_width=SMOOTH_WIDTH,
    )
    mhw = detect(target > thresh[doys - 1], min_duration=MIN_DURATION, max_gap=MAX_GAP)
    return {int(y): int(mhw[years == y].sum()) for y in unique_years}


def _rank_years(mhw_days_per_year):
    """Years sorted by MHW days, most first; ties by increasing year."""
    return sorted(mhw_days_per_year, key=lambda y: (-mhw_days_per_year[y], y))


def _serpentine_buckets(ranked_years, n_folds):
    """Deal ranked years into n_folds buckets: 0..n-1, then n-1..0, and so on."""
    buckets = [[] for _ in range(n_folds)]
    for i, y in enumerate(ranked_years):
        rnd, pos = divmod(i, n_folds)
        buckets[pos if rnd % 2 == 0 else n_folds - 1 - pos].append(int(y))
    return buckets


def _purge(train_idx, val_idx, test_idx, gap):
    """Remove val samples within `gap` of a test sample, then train samples
    within `gap` of a kept val sample or a test sample. Test is untouched.
    Returns (train, val) as sorted lists. gap=0 removes nothing (the three
    index sets are disjoint).
    """

    def _far_from(idx, protected):
        idx = np.asarray(sorted(idx), dtype=np.int64)
        protected = np.asarray(sorted(protected), dtype=np.int64)
        if len(idx) == 0 or len(protected) == 0:
            return idx.tolist()
        pos = np.searchsorted(protected, idx)
        left = protected[np.clip(pos - 1, 0, len(protected) - 1)]
        right = protected[np.clip(pos, 0, len(protected) - 1)]
        dist = np.minimum(np.abs(idx - left), np.abs(idx - right))
        return idx[dist > gap].tolist()

    val_kept = _far_from(val_idx, test_idx)
    train_kept = _far_from(train_idx, list(val_kept) + list(test_idx))
    return train_kept, val_kept


def _print_split(
    fold,
    n_folds,
    train_years,
    val_years,
    test_years,
    mhw_days_per_year,
    train_indices,
    val_indices,
    test_indices,
    n_train_purged,
    n_val_purged,
):
    """Print each split's years, MHW-day counts, sample counts and purged samples."""

    def _days(years_set):
        return sum(mhw_days_per_year[int(y)] for y in years_set)

    print(f"\nStratified k-fold (fold={fold}/{n_folds}):")
    print(f"  train_years ({len(train_years)}): {sorted(train_years)}")
    print(
        f"    MHW days: {_days(train_years)}, samples: {len(train_indices)} "
        f"({n_train_purged} purged)"
    )
    print(f"  val_years   ({len(val_years)}): {sorted(val_years)}")
    print(
        f"    MHW days: {_days(val_years)}, samples: {len(val_indices)} "
        f"({n_val_purged} purged)"
    )
    print(f"  test_years  ({len(test_years)}): {sorted(test_years)}")
    print(f"    MHW days: {_days(test_years)}, samples: {len(test_indices)}")


class LazyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config: dict,
    ):
        super().__init__()

        self.config = config

        if "data_dir" in self.config:
            raise ValueError(
                "'data_dir' in the config is not read: the data file comes from "
                "$MHW_DATA_FILE. Remove the key from the yaml."
            )
        self.data_dir = str(DATA_FILE)  # $MHW_DATA_FILE
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]

        self.split_mode = self.config["split_mode"]
        if self.split_mode != "stratified_kfold":
            raise ValueError(
                f"split_mode must be 'stratified_kfold', got {self.split_mode!r}"
            )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.target_mean = None
        self.target_std = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return  # already set up — avoid reloading data on second call

        full_ds = LazyDataset(self.data_dir, config=self.config)
        total_size = len(full_ds)

        fold = self.config["fold"]
        n_folds = self.config["n_folds"]

        # Year of the target day for each sample.
        target_years = np.array(
            [
                int(full_ds.years[i + full_ds.window_size - 1 + full_ds.lead_time])
                for i in range(total_size)
            ]
        )
        unique_years = np.unique(target_years)

        # Rank years by MHW-day count, most extreme first.
        mhw_days_per_year = _mhw_days_per_year(full_ds, unique_years)

        # Deal the ranked years into n_folds buckets in serpentine order.
        buckets = _serpentine_buckets(_rank_years(mhw_days_per_year), n_folds)

        # This fold's bucket is test, the next bucket is val, the rest is train.
        test_years = set(buckets[fold])
        val_years = set(buckets[(fold + 1) % n_folds])
        train_years = set(unique_years.tolist()) - test_years - val_years

        # Samples whose target day falls in a train/val/test year.
        train_indices = [i for i in range(total_size) if target_years[i] in train_years]
        val_indices = [i for i in range(total_size) if target_years[i] in val_years]
        test_indices = [i for i in range(total_size) if target_years[i] in test_years]

        # Purge samples that share days with a higher-priority split,
        # before compute_stats so the stats use the purged train set.
        gap = full_ds.window_size - 1 + full_ds.lead_time
        n_train, n_val = len(train_indices), len(val_indices)
        train_indices, val_indices = _purge(
            train_indices, val_indices, test_indices, gap
        )

        _print_split(
            fold,
            n_folds,
            train_years,
            val_years,
            test_years,
            mhw_days_per_year,
            train_indices,
            val_indices,
            test_indices,
            n_train - len(train_indices),
            n_val - len(val_indices),
        )

        full_ds.compute_stats(train_indices)
        self.target_mean = full_ds.target_mean
        self.target_std = full_ds.target_std

        self.train_dataset = Subset(full_ds, train_indices)
        self.val_dataset = Subset(full_ds, val_indices)
        self.test_dataset = Subset(full_ds, test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
