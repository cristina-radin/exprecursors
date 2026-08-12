"""
PyTorch Lightning DataModule for daily climate data.

Split modes (config: split_mode):
  "temporal"   — past→train, middle→val, future→test
  "random"     — random split at sample level (leakage from autocorrelation)
  "block_year" — years assigned randomly to splits; no sample overlap within
                 a year, all periods represented in train (default)
"""

from typing import Optional
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import yaml

from .dataset import LazyDataset


class LazyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        config_path: str = "/p/project1/hai_1127/radin1/exprecursors/config.yaml",
    ):
        super().__init__()
        self.save_hyperparameters()

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.data_dir    = self.config["data_dir"]
        self.batch_size  = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.seed        = self.config["seed"]
        self.train_ratio = self.config["train_ratio"]
        self.val_ratio   = self.config["val_ratio"]
        self.test_ratio  = self.config["test_ratio"]
        self.split_mode  = self.config.get("split_mode", "block_year")

        total = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"

        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None
        self.target_mean   = 0.0
        self.target_std    = 1.0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_dataset is not None:
            return  # already set up — avoid reloading data on second call

        full_ds    = LazyDataset(self.data_dir, config_path=self.hparams.config_path)
        total_size = len(full_ds)
        rng        = np.random.default_rng(self.seed)

        if self.split_mode == "block_year":
            # Year of the TARGET for each sample
            target_years = np.array([
                int(full_ds.years[i + full_ds.window_size - 1 + full_ds.lead_time])
                for i in range(total_size)
            ])

            unique_years   = np.unique(target_years)
            n_years        = len(unique_years)
            shuffled_years = rng.permutation(unique_years)

            n_train = int(self.train_ratio * n_years)
            n_val   = int(self.val_ratio   * n_years)

            train_years = set(shuffled_years[:n_train])
            val_years   = set(shuffled_years[n_train:n_train + n_val])
            test_years  = set(shuffled_years[n_train + n_val:])

            train_indices = [i for i in range(total_size) if target_years[i] in train_years]
            val_indices   = [i for i in range(total_size) if target_years[i] in val_years]
            test_indices  = [i for i in range(total_size) if target_years[i] in test_years]

            split_label = (f"Block-year split  "
                           f"(train {n_train} yrs, val {n_val} yrs, "
                           f"test {n_years - n_train - n_val} yrs)")

        elif self.split_mode == "random":
            train_size    = int(self.train_ratio * total_size)
            val_size      = int(self.val_ratio   * total_size)
            train_indices = sorted(rng.choice(total_size, size=train_size, replace=False))
            remaining     = sorted(set(range(total_size)) - set(train_indices))
            remaining_arr = np.array(remaining)
            rng.shuffle(remaining_arr)
            val_indices   = sorted(remaining_arr[:val_size].tolist())
            test_indices  = sorted(remaining_arr[val_size:].tolist())
            split_label   = "Random split"

        else:  # temporal
            train_size    = int(self.train_ratio * total_size)
            val_size      = int(self.val_ratio   * total_size)
            train_indices = list(range(0, train_size))
            val_indices   = list(range(train_size, train_size + val_size))
            test_indices  = list(range(train_size + val_size, total_size))
            split_label   = "Temporal split"

        full_ds.compute_stats(train_indices)
        self.target_mean = full_ds.target_mean
        self.target_std  = full_ds.target_std

        self.train_dataset = Subset(full_ds, train_indices)
        self.val_dataset   = Subset(full_ds, val_indices)
        self.test_dataset  = Subset(full_ds, test_indices)

        print(f"\n{split_label} (mode='{self.split_mode}'):")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Val:   {len(val_indices)} samples")
        print(f"  Test:  {len(test_indices)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
