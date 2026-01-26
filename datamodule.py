"""
PyTorch Lightning DataModule for climate data.
"""

from typing import Optional
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from dataset import LazyDataset


class LazyDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for climate data.
    
    Args:
        train_files: Path or pattern to training data files.
        val_files: Path or pattern to validation data files. If none, will select randomly from train_files
        test_files: Path or pattern to test data files.
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        train_val_test_split: Tuple of (train, val, test) fractions
        seed: Random seed for splitting
    """
    
    def __init__(
        self,
        train_files: str,
        val_files: str = None,
        test_files: str = None,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: tuple = (0.85, 0.15),
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for each stage (fit, validate, test, predict).
        Currently, we load the full dataset and split it into train/val/test.
        You could also provide separate datasets for each stage, but needs to be implemented.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """

        self.stage = stage

        if self.val_files is not None:
            self.train_dataset = LazyDataset(self.train_files)
            self.val_dataset = LazyDataset(self.val_files)
        else :
            full_ds = LazyDataset(self.train_files)
            # Calculate split sizes
            total_size = len(full_ds)
            train_size = int(self.train_val_split[0] * total_size)
            val_size = int(self.train_val_split[1] * total_size)
            # Split dataset
            self.train_dataset, self.val_dataset = random_split(
                full_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if self.test_files is not None:
            self.test_dataset = LazyDataset(self.test_files)
        else:
            self.test_dataset = None
            if stage == 'test':
                raise ValueError("Test files must be provided for test stage.")
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )



