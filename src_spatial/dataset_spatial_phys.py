"""
dataset_spatial_phys.py — Dataset for spatial SST prediction with physics loss support.

Thin subclass of SpatialDataset. The only difference: __getitem__ also returns
target_month (int 0-11) so the training loop can index into MLD weight maps.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset_spatial import SpatialDataset


class SpatialDatasetPhys(SpatialDataset):
    """Identical to SpatialDataset but returns (x_spatial, x_temporal, y, month)."""

    def __getitem__(self, idx):
        x_spatial, x_temporal, y = super().__getitem__(idx)
        target_idx = idx + self.window_size - 1 + self.lead_time
        target_month = int(self.months[target_idx]) - 1  # 0-11
        return x_spatial, x_temporal, y, torch.tensor(target_month, dtype=torch.long)
