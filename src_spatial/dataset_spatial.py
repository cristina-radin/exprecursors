"""
dataset_spatial.py — Dataset for spatial SST anomaly prediction.

Single data file: merged_daily.nc (all variables: to_anom, ptho_bot, ERA5, land masks)
  to_anom:           SST anomaly (pre-computed, 11-day CDO window, ref 1985-2014)
  ptho_bot:          bottom temperature (raw → climatology removed on-the-fly)
  u10/v10/msl/ssr:   ERA5 (raw → climatology removed on-the-fly)
  land_mask:         SST ocean mask
  land_mask_tbottom: bottom-temperature ocean mask (used for ptho_bot)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset

DATA_FILE = "/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc"


class SpatialDataset(Dataset):

    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.variables = config["variables"]
        self.ocean_vars = set(config.get("ocean_variables", []))
        self.window_size = config.get("window_size", 60)
        self.lead_time = config.get("lead_time", 7)
        self.clim_ref_start = config.get("clim_ref_start", 1985)
        self.clim_ref_end = config.get("clim_ref_end", 2014)
        self.clim_window = config.get("clim_window", 5)

        ds = xr.open_dataset(DATA_FILE)

        self.years = ds.time.dt.year.values
        self.months = ds.time.dt.month.values
        self.doys = ds.time.dt.dayofyear.values
        self.year_min = int(self.years.min())
        self.year_max = int(self.years.max())

        # SST land mask for all surface variables
        # Convention in merged_daily.nc: land_mask=1 means OCEAN, 0 means LAND
        lm = ds["land_mask"].values.astype(bool)
        self.ocean_mask = torch.tensor(lm, dtype=torch.bool)  # True = ocean
        self.land_mask = torch.tensor(~lm, dtype=torch.bool)  # True = land

        # Bottom-temperature mask for ptho_bot
        if "land_mask_tbottom" in ds.data_vars:
            lm_bot = ds["land_mask_tbottom"].values.astype(bool)
            self.land_mask_tbottom = torch.tensor(
                ~lm_bot, dtype=torch.bool
            )  # True = land
        else:
            self.land_mask_tbottom = self.land_mask

        print(f"Loading variables from {DATA_FILE.split('/')[-1]}...")
        self.data = {}
        for v in self.variables:
            self.data[v] = torch.tensor(ds[v].values, dtype=torch.float32)

        # Spatial target: to_anom field at t + lead_time
        self.target_spatial = torch.tensor(ds["to_anom"].values, dtype=torch.float32)
        ds.close()
        print(f"  Variables: {self.variables}")

        # Climatology for ERA5 vars (same logic as original dataset.py)
        self.clim_means: dict = {}
        self._compute_clim()

        # Normalisation stats (set by compute_stats after split)
        self.input_means: Optional[torch.Tensor] = None
        self.input_stds: Optional[torch.Tensor] = None
        self.target_pix_std: Optional[torch.Tensor] = None  # (H, W) per-pixel std

        print(
            f"Dataset ready: {len(self)} samples  |  window={self.window_size}  lead={self.lead_time}"
        )
        print(f"  Variables: {self.variables}")
        print(f"  Ocean pixels: {self.ocean_mask.sum().item()}")

    def _compute_clim(self):
        vars_need_clim = [v for v in self.variables if v != "to_anom"]
        if not vars_need_clim:
            return

        ref_mask = (self.years >= self.clim_ref_start) & (
            self.years <= self.clim_ref_end
        )
        ref_doys = self.doys[ref_mask].copy()
        ref_doys[ref_doys == 366] = 365
        half_w = self.clim_window // 2

        print(f"Computing climatology ({self.clim_ref_start}-{self.clim_ref_end})...")
        for var in vars_need_clim:
            ref_data = self.data[var][ref_mask].numpy()
            clim = np.zeros(
                (365, ref_data.shape[1], ref_data.shape[2]), dtype=np.float32
            )
            for d in range(1, 366):
                window_doys = np.array(
                    [((d - 1 + off) % 365) + 1 for off in range(-half_w, half_w + 1)]
                )
                mask = np.isin(ref_doys, window_doys)
                if mask.any():
                    clim[d - 1] = ref_data[mask].mean(axis=0)
            self.clim_means[var] = torch.tensor(clim, dtype=torch.float32)

    def compute_stats(self, train_indices: list):
        """Compute normalisation stats from training samples only."""
        print("Computing normalisation stats from training data...")

        # Input: per-variable global mean/std over training time steps and ocean pixels
        train_t = sorted(
            set(idx + t for idx in train_indices for t in range(self.window_size))
        )
        means, stds = [], []
        for var in self.variables:
            frames = self.data[var][train_t]  # (N_train, H, W)
            ocean_vals = frames[:, self.ocean_mask]  # (N_train, n_ocean)
            valid = ocean_vals[~torch.isnan(ocean_vals)]  # exclude NaN coastal pixels
            means.append(valid.mean().item())
            stds.append(valid.std().item() + 1e-8)

        self.input_means = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.input_stds = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)

        # Target: per-pixel std over training target time steps
        target_t = sorted(
            set(idx + self.window_size - 1 + self.lead_time for idx in train_indices)
        )
        target_frames = self.target_spatial[target_t]  # (N, H, W)
        # std per pixel (ocean only), mean ≈ 0 since to_anom is already centred
        pix_std = target_frames.std(dim=0)  # (H, W)
        pix_std[self.land_mask] = 1.0  # land pixels: no normalization
        pix_std = torch.nan_to_num(
            pix_std, nan=1.0
        )  # coastal NaN pixels: no normalization
        pix_std = torch.clamp(pix_std, min=1e-4)
        self.target_pix_std = pix_std

        print(f"  Input means: {[f'{m:.3f}' for m in means]}")
        print(f"  Input stds:  {[f'{s:.3f}' for s in stds]}")
        print(f"  Target std (ocean mean): {pix_std[self.ocean_mask].mean():.4f}")

    def __len__(self):
        return len(self.years) - self.window_size - self.lead_time + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_spatial:  (T, C, H, W)  normalised input window
            x_temporal: (T, 3)        year_norm, month_sin, month_cos
            y:          (H, W)        normalised spatial target
        """
        frames = []
        temporal = []

        for t in range(idx, idx + self.window_size):
            frame = torch.stack([self.data[v][t] for v in self.variables], dim=0)

            # Land mask: ptho_bot uses its own mask, other ocean vars use SST mask
            for i, var in enumerate(self.variables):
                if var in self.ocean_vars:
                    mask = (
                        self.land_mask_tbottom if var == "ptho_bot" else self.land_mask
                    )
                    frame[i, mask] = float("nan")

            # Climatology removal for ERA5 vars
            if self.clim_means:
                doy = min(int(self.doys[t]), 365)
                for i, var in enumerate(self.variables):
                    if var in self.clim_means:
                        frame[i] -= self.clim_means[var][doy - 1]

            # Normalise inputs
            if self.input_means is not None:
                frame = (frame - self.input_means) / self.input_stds

            frame = torch.nan_to_num(frame, nan=0.0)
            frames.append(frame)

            yr = (self.years[t] - self.year_min) / max(self.year_max - self.year_min, 1)
            temporal.append(
                torch.tensor(
                    [
                        yr,
                        float(np.sin(2 * np.pi * self.months[t] / 12)),
                        float(np.cos(2 * np.pi * self.months[t] / 12)),
                    ],
                    dtype=torch.float32,
                )
            )

        x_spatial = torch.stack(frames, dim=0)  # (T, C, H, W)
        x_temporal = torch.stack(temporal, dim=0)  # (T, 3)

        # Spatial target
        target_idx = idx + self.window_size - 1 + self.lead_time
        y = self.target_spatial[target_idx].clone()  # (H, W)
        y[self.land_mask] = 0.0
        if self.target_pix_std is not None:
            y = y / self.target_pix_std
        y = torch.nan_to_num(y, nan=0.0)

        return x_spatial, x_temporal, y
