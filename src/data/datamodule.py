"""
PyTorch Lightning DataModule for the North Sea MHW dataset.

Builds a stratified k-fold split: ranks calendar years by MHW-day count,
deals them round-robin into n_folds buckets, and uses one bucket as test,
the next bucket as val, and the rest as train. This keeps MHW-day
representation balanced across folds and gives every fold a different val
set.
"""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Subset

from src.utils.hobday import apply_hobday, load_ns_p90
from src.utils.paths import DATA_FILE

from .dataset import LazyDataset


def _mhw_days_per_year(full_ds, unique_years):
    """Count MHW days per calendar year in the full target series."""
    p90 = load_ns_p90()  # (365,)
    raw_doys = full_ds.doys.copy()
    raw_doys[raw_doys >= 365] = 365
    raw_years = full_ds.years
    raw_target = full_ds.target.numpy()
    thresh_per_day = p90[raw_doys - 1]
    mhw_day_bool = apply_hobday(raw_target > thresh_per_day)
    return {int(y): int(mhw_day_bool[raw_years == y].sum()) for y in unique_years}


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
):
    """Print each split's years, MHW-day counts, and sample counts."""

    def _days(years_set):
        return sum(mhw_days_per_year[int(y)] for y in years_set)

    print(f"\nStratified k-fold (fold={fold}/{n_folds}):")
    print(f"  train_years ({len(train_years)}): {sorted(train_years)}")
    print(f"    MHW days: {_days(train_years)}, samples: {len(train_indices)}")
    print(f"  val_years   ({len(val_years)}): {sorted(val_years)}")
    print(f"    MHW days: {_days(val_years)}, samples: {len(val_indices)}")
    print(f"  test_years  ({len(test_years)}): {sorted(test_years)}")
    print(f"    MHW days: {_days(test_years)}, samples: {len(test_indices)}")


class LazyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config_path: str = "config.yaml",
    ):
        super().__init__()
        self.save_hyperparameters()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

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
        self.target_mean = 0.0
        self.target_std = 1.0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return  # already set up — avoid reloading data on second call

        full_ds = LazyDataset(self.data_dir, config_path=self.hparams.config_path)
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

        # Deal the ranked years round-robin into n_folds buckets.
        ranked_years = sorted(unique_years, key=lambda y: -mhw_days_per_year[int(y)])
        buckets = [[] for _ in range(n_folds)]
        for i, y in enumerate(ranked_years):
            buckets[i % n_folds].append(int(y))

        # This fold's bucket is test, the next bucket is val, the rest is train.
        test_years = set(buckets[fold])
        val_years = set(buckets[(fold + 1) % n_folds])
        train_years = set(unique_years.tolist()) - test_years - val_years

        # Samples whose target day falls in a train/val/test year.
        train_indices = [i for i in range(total_size) if target_years[i] in train_years]
        val_indices = [i for i in range(total_size) if target_years[i] in val_years]
        test_indices = [i for i in range(total_size) if target_years[i] in test_years]

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
