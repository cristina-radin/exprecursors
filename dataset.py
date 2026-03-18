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
import yaml


class LazyDataset(Dataset):
    """
    Dataset class for loading climate model data from disk.
    
    Args:
        file_pattern: Glob pattern or path to data files (e.g., '/path/to/data/*.nc')
        lazy_loading: If True, load data from disk during training; otherwise load all data into memory
        offset: Time offset between precursor and target variables
        variables: List of variable names to load from the dataset (e.g., ["zos", "mld", "to", "target"])
        normalize: Whether to normalize the input variables
        norm_stats: Optional dictionary of normalization statistics (mean, std) for each variable. If None, will compute from the dataset (only if lazy_loading=False)
    """
    
    def __init__(
        self,
        file_name: str,
        lazy_loading=False,
        offset=1,
        variables=None,
        normalize: bool = None,
        norm_stats: Optional[dict] = None,
        add_temporal_features: bool = None,  
        config_path: str = "/p/project1/hai_1127/radin1/exprecursors/config.yaml"
        #self.input_time_steps
    ):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
            if variables is None:
                variables = config['variables']
            if normalize is None:
                normalize = config.get('normalize', True)
            if add_temporal_features is None:
                add_temporal_features = config.get('add_temporal_features', True)


        self.file_name = file_name
        self.lazy_loading = lazy_loading
        self.offset = offset
        self.variables = variables
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.add_temporal_features = add_temporal_features

        # you can open mutliple netcdf files here without acutally loading the data into memory
        # A zarr dataset would be even quicker than nc
        self.ds = xr.open_mfdataset(self.file_name, parallel=True)
        #self.input_time_steps

        self.ds = self.ds[variables]  # Select only the variables we need to save memory
        if 'land_mask' in variables:
            land_mask_data = self.ds['land_mask'].values
            if land_mask_data.ndim == 3:
                land_mask_data = land_mask_data[0]  # Si tiene time, coger primero
            self.tierra_indices = land_mask_data == 0  # True donde hay tierra
            print(f"Land: {self.tierra_indices.mean():.1%} of the map")

        if self.add_temporal_features:
            self.years = self.ds.time.dt.year.values
            self.months = self.ds.time.dt.month.values
            self.days = self.ds.time.dt.day.values
            self.year_min = self.years.min()
            self.year_max = self.years.max()
            print(f"Temporal coverage {self.year_min}-{self.year_max}")
        
        if not self.lazy_loading:
            self.ds = self.ds.load()  # Load all data into memory upfront
            if self.normalize and self.norm_stats is None:
                self.norm_stats = self._compute_stats()

    def _compute_stats(self):
        """Compute mean and std for each variable in the dataset for normalization.
        ONLY OVER OCEAN PIXELS (LAND_MASK=0)"""
        stats = {}
        physical_vars = [v for v in self.variables if v not in ['target', 'land_mask']]
        
        for var in physical_vars:
            data = self.ds[var].values  # [time, lat, lon]
            
            data_con_nan = data.copy()
            data_con_nan[:, self.tierra_indices] = np.nan
            
            # Statistics ignoring land pixels (NaN)
            mean = np.nanmean(data_con_nan)
            std = np.nanstd(data_con_nan) + 1e-8
            
            stats[var] = {'mean': mean, 'std': std}
          
        self.input_means = torch.tensor([stats[v]['mean'] for v in physical_vars], dtype=torch.float32).view(-1, 1, 1)
        self.input_stds = torch.tensor([stats[v]['std'] for v in physical_vars], dtype=torch.float32).view(-1, 1, 1)
    
        return stats
    
    def get_norm_stats(self):
        return self.norm_stats

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        TODO This depends on how exactly we sample the data!
        """
        return len(self.ds.time) - self.offset
    
    def __getitem__(self, idx):
        print(f"Loading sample {idx}")


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a sample from the dataset.
        
        Args:
            idx: Index of the sample to load. During training, this will usually be a random index according to how large the dataset is (given by len(self))
            
        Returns:
            Tuple of (input_data, target_data) as tensors
        """
        physical_vars = [v for v in self.variables if v not in ['target', 'land_mask']]
        all_vars = [v for v in self.variables if v != 'target']
        
        x_ds = self.ds[all_vars].isel(time=idx).to_array().values
        y_ds = self.ds["target"].isel(time=idx+self.offset).values

        input_data = torch.tensor(x_ds, dtype=torch.float32)  # [vars, lat, lon]
        target_data = torch.tensor(y_ds, dtype=torch.float32).unsqueeze(0)
        
        land_mask = input_data[len(physical_vars)]
        physical = input_data[:len(physical_vars)]
        
        #Normalize with nans to not take into account land pixels
        tierra_mask = (land_mask == 0)
        physical[:, tierra_mask] = float('nan')
        
        if self.normalize and hasattr(self, 'input_means'):
            physical = (physical.float() - self.input_means.float()) / self.input_stds.float()
        
        # Nan to zero for the model
        physical = torch.nan_to_num(physical, nan=0.0)
        
        input_data = torch.cat([physical, land_mask.unsqueeze(0)], dim=0)
        
        if self.add_temporal_features:
            #print(f"Adding temporal features...")
            year_norm = (self.years[idx] - self.year_min) / (self.year_max - self.year_min)
            month_sin = np.sin(2 * np.pi * self.months[idx] / 12)
            month_cos = np.cos(2 * np.pi * self.months[idx] / 12)
            
            _, lat, lon = input_data.shape
            year_channel = torch.ones(1, lat, lon) * year_norm
            month_sin_channel = torch.ones(1, lat, lon) * month_sin
            month_cos_channel = torch.ones(1, lat, lon) * month_cos
            
            input_data = torch.cat([input_data, year_channel, month_sin_channel, month_cos_channel], dim=0)
        
        input_data = input_data.float()
        target_data = target_data.float()
        return input_data.float(), target_data.float()


    def get_class_balance(self):
        n_pos = 0
        n_total = len(self)
        
        for i in range(n_total):
            _, y = self[i]
            if y.item() == 1:
                n_pos += 1
            if (i+1) % 50 == 0:
                print(f"  Progress: {i+1}/{n_total}")
        
        n_neg = n_total - n_pos
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"\nClass balance:")
        print(f"  Total samples: {n_total}")
        print(f"  Positives (MHW): {n_pos} ({n_pos/n_total*100:.2f}%)")
        print(f"  Negatives (no MHW): {n_neg} ({n_neg/n_total*100:.2f}%)")
        print(f"  pos_weight recommended: {pos_weight:.2f}")
        
        return torch.tensor([pos_weight])