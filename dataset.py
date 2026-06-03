"""
Dataset class for daily climate data (MHW precursor detection).

Input:  sliding window of `window_size` days → [window_size, n_vars, lat, lon]
Target: mean SST anomaly over North Sea at t + lead_time (scalar, normalized)

Anomalisation:
  to_anom  — already an anomaly (ICON-COAST, ref 1985-2014); left unchanged.
  u10, v10, msl, ssr — ERA5 absolute values; day-of-year climatology subtracted
                       here using reference period 1985-2014, 5-day running window,
                       consistent with the CDO preprocessing of to_anom.
"""

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import yaml


class LazyDataset(Dataset):
    """
    Dataset for daily merged NetCDF (merged_daily.nc).

    Normalization stats (input and target) are NOT computed here.
    Call compute_stats(train_indices) from the DataModule after splitting,
    so stats are derived from training data only.
    """

    def __init__(
        self,
        file_name: str,
        config_path: str = "/p/project1/hai_1127/radin1/exprecursors/config.yaml",
    ):
        super().__init__()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.file_name       = file_name
        self.variables       = config["variables"]
        self.ocean_variables = set(config.get("ocean_variables", self.variables))
        self.normalize       = config.get("normalize", True)
        self.window_size     = config.get("window_size", 60)
        self.lead_time       = config.get("lead_time", 7)
        self.clim_ref_start    = config.get("clim_ref_start", 1985)
        self.clim_ref_end      = config.get("clim_ref_end",   2014)
        self.clim_window       = config.get("clim_window",    5)
        self.detrend_variables = set(config.get("detrend_variables", []))
        self.detrend_target    = config.get("detrend_target", False)

        self.ds = xr.open_mfdataset(file_name, parallel=True)

        # Temporal coordinates
        self.years    = self.ds.time.dt.year.values
        self.months   = self.ds.time.dt.month.values
        self.doys     = self.ds.time.dt.dayofyear.values  # 1–366
        self.year_min = self.years.min()
        self.year_max = self.years.max()

        # Land mask: True where land — only applied to ocean variables
        land_mask        = torch.tensor(self.ds["land_mask"].values, dtype=torch.float32)
        self.tierra_mask = (land_mask == 0)

        # Pre-load variables into memory
        print("Loading data into memory...")
        self.data = {}
        for var in self.variables:
            self.data[var] = torch.tensor(
                self.ds[var].values, dtype=torch.float32
            )  # (time, lat, lon)
        self.target = torch.tensor(
            self.ds["target"].values, dtype=torch.float32
        )  # (time,)

        print(f"  Variables:  {self.variables}")
        print(f"  Ocean vars: {sorted(self.ocean_variables)}")
        print(f"  Time steps: {len(self.ds.time)}")
        print(f"  Window: {self.window_size} days, lead: {self.lead_time} days")
        print(f"  Samples: {len(self)}")

        # Stats — overwritten by compute_stats()
        self.clim_means   = {}   # var → (365, lat, lon) climatology
        self.trend_slopes = {}   # var → (lat, lon)  linear trend slope [units/day]
        self.trend_ref_t  = 0    # reference timestep index for trend centering
        self.input_means  = None
        self.input_stds   = None
        self.target_mean  = 0.0
        self.target_std   = 1.0

    # ------------------------------------------------------------------
    # Climatology
    # ------------------------------------------------------------------

    def _compute_clim(self) -> None:
        """
        Compute day-of-year running-mean climatology for ERA5 variables
        (all variables except to_anom, which is already an anomaly).

        Reference period : clim_ref_start – clim_ref_end (default 1985-2014).
        Window           : ±(clim_window//2) days (default 5, consistent with
                           CDO ydrunmean,5 used for to_anom preprocessing).
        Leap day (DOY 366) mapped to DOY 365 (Dec 31).

        Result stored in self.clim_means: {var: tensor(365, lat, lon)}.
        """
        vars_to_anom = [v for v in self.variables if v != "to_anom"]
        if not vars_to_anom:
            return

        ref_mask = (self.years >= self.clim_ref_start) & (self.years <= self.clim_ref_end)
        ref_doys = self.doys[ref_mask].copy()
        ref_doys[ref_doys == 366] = 365  # map leap days → Dec 31

        half_w = self.clim_window // 2

        print(f"\nComputing day-of-year climatology "
              f"({self.clim_ref_start}-{self.clim_ref_end}, "
              f"window={self.clim_window})...")

        for var in vars_to_anom:
            ref_data = self.data[var][ref_mask].numpy()  # (n_ref, lat, lon)
            clim     = np.zeros((365, ref_data.shape[1], ref_data.shape[2]),
                                dtype=np.float32)

            for d in range(1, 366):
                window_doys = np.array([
                    ((d - 1 + off) % 365) + 1
                    for off in range(-half_w, half_w + 1)
                ])
                mask = np.isin(ref_doys, window_doys)
                if mask.any():
                    clim[d - 1] = ref_data[mask].mean(axis=0)

            self.clim_means[var] = torch.tensor(clim, dtype=torch.float32)
            print(f"  {var}: clim mean={clim.mean():.4f}, std={clim.std():.4f}")

    # ------------------------------------------------------------------
    # Linear detrending (pixel-wise, full period)
    # ------------------------------------------------------------------

    def _compute_trend(self) -> None:
        """
        Fit a pixel-wise linear trend over the full dataset period for each
        variable listed in self.detrend_variables, AFTER climatology removal.

        The trend (slope) is stored in self.trend_slopes so it can be
        subtracted in __getitem__: anomaly_detrended = anomaly - slope*(t - t_ref)
        where t is the timestep index and t_ref = midpoint of the series.

        This removes long-term drift (e.g. warming trend in ptho_bot) while
        preserving interannual variability.
        """
        if not self.detrend_variables:
            return

        T = len(self.doys)
        t = np.arange(T, dtype=np.float32)
        self.trend_ref_t = float(t.mean())   # center to avoid large intercepts

        print(f"\nComputing pixel-wise linear trend for: {sorted(self.detrend_variables)}")

        for var in self.detrend_variables:
            if var not in self.data:
                print(f"  {var}: not in variables, skipping")
                continue

            data = self.data[var].numpy().astype(np.float32)  # (T, lat, lon)

            # Subtract climatology first so trend is on anomalies
            if var in self.clim_means:
                doys = self.doys.copy()
                doys[doys == 366] = 365
                for i, doy in enumerate(doys):
                    data[i] -= self.clim_means[var][doy - 1].numpy()

            lat, lon = data.shape[1], data.shape[2]
            t_c = t - self.trend_ref_t  # centered time axis

            # Vectorised OLS: slope = cov(t,x) / var(t)
            t_var  = float((t_c ** 2).mean())
            slopes = np.zeros((lat, lon), dtype=np.float32)
            for i in range(lat):
                for j in range(lon):
                    pixel = data[:, i, j]
                    if np.isfinite(pixel).all():
                        slopes[i, j] = float((t_c * pixel).mean()) / t_var

            self.trend_slopes[var] = torch.tensor(slopes, dtype=torch.float32)
            print(f"  {var}: mean slope={slopes.mean():.6f}/day  "
                  f"({slopes.mean()*3650:.4f} per decade)")

    # ------------------------------------------------------------------
    # Normalisation stats (call after split)
    # ------------------------------------------------------------------

    def compute_stats(self, train_indices) -> None:
        """
        1. Compute day-of-year climatology from 1985-2014 (independent of split).
        2. Compute mean/std from training data on climatology-removed values.

        Args:
            train_indices: list/array of sample indices in the train set.
        """
        train_indices = list(train_indices)

        # Step 1 — climatology (reference period, independent of split)
        self._compute_clim()
        self._compute_trend()

        # Step 2 — mean/std on training time span
        t_start = min(train_indices)
        t_end   = max(train_indices) + self.window_size

        print("\nComputing normalisation stats from train data only...")
        means, stds = [], []

        for var in self.variables:
            data = self.data[var][t_start:t_end].clone()  # (T, lat, lon)

            # Subtract day-of-year climatology for ERA5 variables
            if var in self.clim_means:
                abs_ts = np.arange(t_start, t_start + len(data))
                doys   = self.doys[abs_ts].copy()
                doys[doys == 366] = 365
                for i, doy in enumerate(doys):
                    data[i] -= self.clim_means[var][doy - 1]

            if var in self.ocean_variables:
                data[:, self.tierra_mask] = float("nan")
                mean = float(torch.nanmean(data))
                std  = float(torch.std(data[~torch.isnan(data)])) + 1e-8
            else:
                mean = float(data.mean())
                std  = float(data.std()) + 1e-8

            means.append(mean)
            stds.append(std)
            print(f"  {var}: mean={mean:.4f}, std={std:.4f}")

        self.input_means = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.input_stds  = torch.tensor(stds,  dtype=torch.float32).view(-1, 1, 1)

        # Detrend target (full period, pixel-wise scalar)
        if self.detrend_target:
            T      = len(self.target)
            t      = np.arange(T, dtype=np.float64)
            t_c    = t - t.mean()
            target_np = self.target.numpy().astype(np.float64)
            slope  = float((t_c * target_np).mean() / (t_c ** 2).mean())
            self.target_trend_slope = slope
            self.target_trend_ref_t = float(t.mean())
            self.target = torch.tensor(
                target_np - slope * t_c, dtype=torch.float32
            )
            print(f"  target detrended: slope={slope*3650:.4f}/decade")
        else:
            self.target_trend_slope = 0.0
            self.target_trend_ref_t = 0.0

        # Target stats from training target timestamps
        target_idx_list  = [i + self.window_size - 1 + self.lead_time
                            for i in train_indices]
        train_targets    = self.target[torch.tensor(target_idx_list, dtype=torch.long)]
        self.target_mean = float(train_targets.mean())
        self.target_std  = float(train_targets.std()) + 1e-8
        print(f"  target: mean={self.target_mean:.4f}, std={self.target_std:.4f}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ds.time) - self.window_size - self.lead_time + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_spatial:  (window_size, n_vars, lat, lon) — anomalised + normalised
            x_temporal: (window_size, 3)                — year_norm, month_sin, month_cos
            y:          (1,)                            — normalised North Sea SST anomaly
        """
        window_spatial  = []
        window_temporal = []

        for t in range(idx, idx + self.window_size):
            # --- Spatial frame ---
            frame = torch.stack([self.data[v][t] for v in self.variables], dim=0)

            # Land mask (ocean variables only)
            for i, var in enumerate(self.variables):
                if var in self.ocean_variables:
                    frame[i, self.tierra_mask] = float("nan")

            # Subtract day-of-year climatology (ERA5 variables only)
            if self.clim_means:
                doy = int(self.doys[t])
                if doy >= 365:
                    doy = 365
                for i, var in enumerate(self.variables):
                    if var in self.clim_means:
                        frame[i] -= self.clim_means[var][doy - 1]

            # Subtract linear trend (detrend_variables only)
            if self.trend_slopes:
                t_c = float(t) - self.trend_ref_t
                for i, var in enumerate(self.variables):
                    if var in self.trend_slopes:
                        frame[i] -= self.trend_slopes[var] * t_c

            # Normalise
            if self.normalize and self.input_means is not None:
                frame = (frame - self.input_means) / self.input_stds

            frame = torch.nan_to_num(frame, nan=0.0)
            window_spatial.append(frame)

            # --- Temporal features ---
            year_norm = (self.years[t] - self.year_min) / (self.year_max - self.year_min)
            month_sin = np.sin(2 * np.pi * self.months[t] / 12)
            month_cos = np.cos(2 * np.pi * self.months[t] / 12)
            window_temporal.append(
                torch.tensor([year_norm, month_sin, month_cos], dtype=torch.float32)
            )

        x_spatial  = torch.stack(window_spatial,  dim=0)  # (window_size, n_vars, lat, lon)
        x_temporal = torch.stack(window_temporal, dim=0)  # (window_size, 3)

        target_idx = idx + self.window_size - 1 + self.lead_time
        y_raw = self.target[target_idx]
        y = ((y_raw - self.target_mean) / self.target_std).unsqueeze(0)

        return x_spatial, x_temporal, y
