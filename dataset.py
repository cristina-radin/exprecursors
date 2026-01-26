"""
Dataset classes for loading climate model data.
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class LazyDataset(Dataset):
    """
    Dataset class for loading climate model data from disk.
    
    Args:
        file_pattern: Glob pattern or path to data files (e.g., '/path/to/data/*.nc')
        lazy_loading: If True, load data from disk during training; otherwise load all data into memory
        offset: Time offset between precursor and target variables
    """
    
    def __init__(
        self,
        file_name: str,
        lazy_loading=True,
        offset=1,
    ):
        super().__init__()
        self.file_name = file_name
        self.lazy_loading = lazy_loading
        self.offset = offset

        self.ds = xr.open_dataset(self.file_name)
        if not self.lazy_loading:
            self.ds = self.ds.load()  # Load all data into memory upfront


    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        TODO This depends on how exactly we sample the data!
        """
        return len(self.ds.time) - self.offset
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load. During training, this will usually be a random index according to how large the dataset is (given by len(self))
            
        Returns:
            Tuple of (input_data, target_data) as tensors
        """
        x_ds = self.ds["precursor"].isel(time=idx).values # .values loads data into memory as numpy array
        y_ds = self.ds["target"].isel(time=idx+self.offset).values

        input_data = torch.tensor(x_ds, dtype=torch.float32) # could experiment with dtype=float16 if dataset is very large.
        target_data = torch.tensor(y_ds, dtype=torch.float32)
        
        return input_data, target_data
