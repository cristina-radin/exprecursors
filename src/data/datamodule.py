"""
PyTorch Lightning DataModule for daily climate data.

Split modes (config: split_mode):
  "temporal"        — past→train, middle→val, future→test
  "random"          — random split at sample level (leakage from autocorrelation)
  "block_year"      — years assigned randomly to splits; no sample overlap within
                       a year, all periods represented in train (default)
  "kfold"           — legacy k-fold. KNOWN BUG (known_issues.md #1, Aug 20
                       2026): val_years is identical across folds 1..n_folds-1
                       (only fold 0 differs) because val is always the first
                       n_val years of "remaining" years, and "remaining"'s
                       leading years only change when the removed test block
                       itself starts at index 0. Kept as-is (not fixed
                       in-place) so it doesn't silently change any existing
                       kfold-mode experiment already cited in narrative.md —
                       use "stratified_kfold" for new work instead.
  "stratified_kfold" — fixed k-fold: val/test years come from rotating,
                       non-overlapping buckets built by round-robin
                       assignment over years ranked by Hobday MHW-day count
                       (descending), so (a) every fold gets a different
                       val_years set by construction, and (b) MHW-day
                       representation is balanced across folds instead of
                       landing wherever the random permutation happens to
                       put it. See known_issues.md #1/#2 for the exact
                       measurements that motivated this.
"""

from typing import Optional

import numpy as np
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Subset

from .dataset import LazyDataset


def _mhw_days_per_year(full_ds, unique_years):
    """MHW-day count per calendar year, from the raw daily target series
    (Hobday basin-mean definition) -- reuses apply_hobday()/load_ns_p90()
    rather than reimplementing exceedance/persistence logic (see
    known_issues.md #1's meta-hallazgo on duplicated-and-wrong split code)."""
    from src.utils.hobday import apply_hobday, load_ns_p90

    p90 = load_ns_p90()  # (365,)
    raw_doys = full_ds.doys.copy()
    raw_doys[raw_doys >= 365] = 365
    raw_years = full_ds.years
    raw_target = full_ds.target.numpy()
    thresh_per_day = p90[raw_doys - 1]
    mhw_day_bool = apply_hobday(raw_target > thresh_per_day)
    return {int(y): int(mhw_day_bool[raw_years == y].sum()) for y in unique_years}


def _print_year_split(
    mode_label, train_years, val_years, test_years, mhw_days_per_year
):
    """Mandatory diagnostic print (plan rule, Aug 20 2026: 'sacar las
    configs reales resueltas como output' + 'comprobar siempre el numero
    de dias bajo MHW') -- the exact year lists, not just counts, for
    every year-based split_mode (block_year, kfold, stratified_kfold), so
    a glance at the SLURM log confirms what's actually in each split
    without having to reconstruct it by hand."""

    def _days(years_set):
        return sum(mhw_days_per_year[int(y)] for y in years_set)

    print(f"\n{mode_label}:")
    print(f"  train_years ({len(train_years)}): {sorted(train_years)}")
    print(f"    MHW days: {_days(train_years)}")
    print(f"  val_years   ({len(val_years)}): {sorted(val_years)}")
    print(f"    MHW days: {_days(val_years)}")
    print(f"  test_years  ({len(test_years)}): {sorted(test_years)}")
    print(f"    MHW days: {_days(test_years)}")


class LazyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config_path: str = "config.yaml",
    ):
        super().__init__()
        self.save_hyperparameters()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir = self.config["data_dir"]
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.seed = self.config["seed"]
        self.train_ratio = self.config["train_ratio"]
        self.val_ratio = self.config["val_ratio"]
        self.test_ratio = self.config["test_ratio"]
        self.split_mode = self.config.get("split_mode", "block_year")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"

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
        rng = np.random.default_rng(self.seed)

        if self.split_mode == "block_year":
            # Year of the TARGET for each sample
            target_years = np.array(
                [
                    int(full_ds.years[i + full_ds.window_size - 1 + full_ds.lead_time])
                    for i in range(total_size)
                ]
            )

            unique_years = np.unique(target_years)
            n_years = len(unique_years)
            shuffled_years = rng.permutation(unique_years)

            n_train = int(self.train_ratio * n_years)
            n_val = int(self.val_ratio * n_years)

            train_years = set(shuffled_years[:n_train])
            val_years = set(shuffled_years[n_train : n_train + n_val])
            test_years = set(shuffled_years[n_train + n_val :])

            train_indices = [
                i for i in range(total_size) if target_years[i] in train_years
            ]
            val_indices = [i for i in range(total_size) if target_years[i] in val_years]
            test_indices = [
                i for i in range(total_size) if target_years[i] in test_years
            ]

            _print_year_split(
                f"Block-year split (fold seed={self.seed})",
                train_years,
                val_years,
                test_years,
                _mhw_days_per_year(full_ds, unique_years),
            )

            split_label = (
                f"Block-year split  "
                f"(train {n_train} yrs, val {n_val} yrs, "
                f"test {n_years - n_train - n_val} yrs)"
            )

        elif self.split_mode == "kfold":
            fold = self.config.get("fold", 0)
            n_folds = self.config.get("n_folds", 5)

            target_years = np.array(
                [
                    int(full_ds.years[i + full_ds.window_size - 1 + full_ds.lead_time])
                    for i in range(total_size)
                ]
            )
            unique_years = np.unique(target_years)
            n_years = len(unique_years)
            # Use legacy RandomState for compatibility with Aug 11 checkpoints.
            # The original (uncommitted) kfold code used np.random.RandomState,
            # not default_rng. Changing this breaks split compatibility.
            kfold_rng = np.random.RandomState(self.seed)
            shuffled_years = kfold_rng.permutation(unique_years)

            fold_size = n_years // n_folds
            test_years = set(shuffled_years[fold * fold_size : (fold + 1) * fold_size])
            remaining_years = [y for y in shuffled_years if y not in test_years]
            n_val = int(self.val_ratio * n_years)
            val_years = set(remaining_years[:n_val])
            train_years = set(remaining_years[n_val:])

            train_indices = [
                i for i in range(total_size) if target_years[i] in train_years
            ]
            val_indices = [i for i in range(total_size) if target_years[i] in val_years]
            test_indices = [
                i for i in range(total_size) if target_years[i] in test_years
            ]

            _print_year_split(
                f"K-fold (fold={fold}/{n_folds}) -- KNOWN BUG: val_years is "
                "identical across folds 1..n_folds-1, see known_issues.md #1",
                train_years,
                val_years,
                test_years,
                _mhw_days_per_year(full_ds, unique_years),
            )

            split_label = (
                f"K-fold  (fold={fold}/{n_folds}, "
                f"train={len(train_years)} yrs, "
                f"val={len(val_years)} yrs, "
                f"test={len(test_years)} yrs)"
            )

        elif self.split_mode == "stratified_kfold":
            fold = self.config.get("fold", 0)
            n_folds = self.config.get("n_folds", 5)

            target_years = np.array(
                [
                    int(full_ds.years[i + full_ds.window_size - 1 + full_ds.lead_time])
                    for i in range(total_size)
                ]
            )
            unique_years = np.unique(target_years)

            # Rank years by Hobday MHW-day count (descending) -- reuses
            # apply_hobday()/load_ns_p90() rather than reimplementing
            # exceedance/persistence logic (known_issues.md pattern of
            # duplicated-and-wrong split/threshold code, e.g. #1's
            # meta-hallazgo on persistence_remote_sst.py/ig_simple.py).
            mhw_days_per_year = _mhw_days_per_year(full_ds, unique_years)

            ranked_years = sorted(
                unique_years, key=lambda y: -mhw_days_per_year[int(y)]
            )
            buckets = [[] for _ in range(n_folds)]
            for i, y in enumerate(ranked_years):
                buckets[i % n_folds].append(int(y))

            test_years = set(buckets[fold])
            val_years = set(buckets[(fold + 1) % n_folds])
            train_years = set(unique_years.tolist()) - test_years - val_years

            train_indices = [
                i for i in range(total_size) if target_years[i] in train_years
            ]
            val_indices = [i for i in range(total_size) if target_years[i] in val_years]
            test_indices = [
                i for i in range(total_size) if target_years[i] in test_years
            ]

            # Mandatory diagnostic print (known_issues.md #1/#2): the exact
            # year lists, not just counts, so a glance at the log confirms
            # val_years doesn't collide with another fold's and that all
            # three splits carry real MHW representation.
            _print_year_split(
                f"Stratified k-fold (fold={fold}/{n_folds})",
                train_years,
                val_years,
                test_years,
                mhw_days_per_year,
            )
            print(
                "  NOTE: this mode ignores train_ratio/val_ratio/test_ratio "
                f"from the config (got train_ratio={self.train_ratio}, "
                f"val_ratio={self.val_ratio}, test_ratio={self.test_ratio}) — "
                f"the split is fixed by the {n_folds}-bucket rotation instead, "
                f"giving ~{len(train_years)}/{len(val_years)}/{len(test_years)} "
                f"years (train/val/test), not the configured ratios."
            )

            split_label = (
                f"Stratified k-fold  (fold={fold}/{n_folds}, "
                f"train={len(train_years)} yrs, "
                f"val={len(val_years)} yrs, "
                f"test={len(test_years)} yrs)"
            )

        elif self.split_mode == "random":
            train_size = int(self.train_ratio * total_size)
            val_size = int(self.val_ratio * total_size)
            train_indices = sorted(
                rng.choice(total_size, size=train_size, replace=False)
            )
            remaining = sorted(set(range(total_size)) - set(train_indices))
            remaining_arr = np.array(remaining)
            rng.shuffle(remaining_arr)
            val_indices = sorted(remaining_arr[:val_size].tolist())
            test_indices = sorted(remaining_arr[val_size:].tolist())
            split_label = "Random split"

        else:  # temporal
            train_size = int(self.train_ratio * total_size)
            val_size = int(self.val_ratio * total_size)
            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, train_size + val_size))
            test_indices = list(range(train_size + val_size, total_size))
            split_label = "Temporal split"

        full_ds.compute_stats(train_indices)
        self.target_mean = full_ds.target_mean
        self.target_std = full_ds.target_std

        self.train_dataset = Subset(full_ds, train_indices)
        self.val_dataset = Subset(full_ds, val_indices)
        self.test_dataset = Subset(full_ds, test_indices)

        print(f"\n{split_label} (mode='{self.split_mode}'):")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val:   {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples")

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
