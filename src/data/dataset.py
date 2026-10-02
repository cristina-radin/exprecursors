"""
Dataset class for daily climate data (MHW precursor detection).

Input:  sliding window of `window_size` days -> [window_size, n_vars, lat, lon]
Target: mean SST anomaly over the North Sea at t + lead_time (scalar, normalised)

All variables in merged_daily.nc arrive as day-of-year anomalies; this module
does not subtract any further climatology.
"""

import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset


class LazyDataset(Dataset):
    """
    Dataset for daily merged NetCDF (merged_daily.nc).

    Normalisation stats (input and target) are not computed here. Call
    compute_stats(train_indices) from the DataModule after splitting, so
    stats are derived from training data only.
    """

    def __init__(
        self,
        file_name: str,
        config_path: str = "config.yaml",
    ):
        super().__init__()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.file_name = file_name
        self.variables = config["variables"]
        self.ocean_variables = set(config["ocean_variables"])
        self.normalize = config["normalize"]
        self.window_size = config["window_size"]
        self.lead_time = config["lead_time"]
        # Opt-in: __getitem__ returns a 4-tuple (..., target_doy) instead of
        # the standard 3-tuple when True. Off by default so the ~20 existing
        # call sites that unpack `xs, xt, y = ...` (scripts/, tests/) are
        # unaffected — only a loss variant that needs a DOY-dependent
        # threshold (e.g. focal-weighted NLL) should set this.
        self.return_target_doy = config.get("return_target_doy", False)

        self.ds = xr.open_mfdataset(file_name, parallel=True, engine="netcdf4")

        # Temporal coordinates
        self.years = self.ds.time.dt.year.values
        self.months = self.ds.time.dt.month.values
        self.doys = self.ds.time.dt.dayofyear.values  # 1–366
        self.year_min = self.years.min()
        self.year_max = self.years.max()

        # ptho_bot uses land_mask_tbottom, not land_mask: the two masks
        # disagree on which pixels are land.
        land_mask = torch.tensor(self.ds["land_mask"].values, dtype=torch.float32)
        self.is_land = land_mask == 0
        self.land_masks = {}
        for var in self.ocean_variables:
            if var == "ptho_bot":
                if "land_mask_tbottom" not in self.ds:
                    raise ValueError(
                        "ptho_bot is an ocean_variable but 'land_mask_tbottom' "
                        "is not in the dataset — cannot mask it correctly."
                    )
                tbottom_mask = torch.tensor(
                    self.ds["land_mask_tbottom"].values, dtype=torch.float32
                )
                self.land_masks[var] = tbottom_mask == 0
            else:
                self.land_masks[var] = self.is_land

        # land_fill_mode="nearest" replaces each land pixel with the value
        # of its nearest ocean neighbour, instead of leaving it at 0.
        self.land_fill_mode = config["land_fill_mode"]
        if self.land_fill_mode not in ("zero", "nearest"):
            raise ValueError(
                f"land_fill_mode must be 'zero' or 'nearest', got {self.land_fill_mode!r}"
            )
        self._land_fill_idx = {}
        if self.land_fill_mode == "nearest":
            from scipy.ndimage import distance_transform_edt

            for var in self.ocean_variables:
                lm_np = self.land_masks[var].numpy()
                _, (nearest_row, nearest_col) = distance_transform_edt(
                    lm_np, return_distances=True, return_indices=True
                )
                land_rows, land_cols = np.where(lm_np)
                src_rows = nearest_row[land_rows, land_cols]
                src_cols = nearest_col[land_rows, land_cols]
                self._land_fill_idx[var] = (
                    torch.as_tensor(land_rows, dtype=torch.long),
                    torch.as_tensor(land_cols, dtype=torch.long),
                    torch.as_tensor(src_rows, dtype=torch.long),
                    torch.as_tensor(src_cols, dtype=torch.long),
                )
                n_land = len(land_rows)
                print(
                    f"  land_fill_mode='nearest': {var} — filling {n_land} land "
                    f"pixels from their nearest ocean neighbor (every timestep)"
                )

        # Pre-load variables into memory
        print("Loading data into memory...")
        self.data = {}
        for var in self.variables:
            self.data[var] = torch.tensor(
                self.ds[var].values, dtype=torch.float32
            )  # (time, lat, lon)
            if self.land_fill_mode == "nearest" and var in self.ocean_variables:
                land_rows, land_cols, src_rows, src_cols = self._land_fill_idx[var]
                self.data[var][:, land_rows, land_cols] = self.data[var][
                    :, src_rows, src_cols
                ]
        self.target = torch.tensor(
            self.ds["target"].values, dtype=torch.float32
        )  # (time,)

        # hobday_smooth_target adds the correction for mean_clim's missing
        # 31-day smoothing to the target, before stats are computed from it.
        self.hobday_smooth_target = config["hobday_smooth_target"]
        if self.hobday_smooth_target:
            # Imported here, not at module level: it needs MHW_CLIM_FILE,
            # which configs that leave this flag off should not require.
            from src.utils.hobday import load_ns_mean_clim_smooth_delta

            delta = load_ns_mean_clim_smooth_delta()  # (365,), physical units
            doys_clamped = self.doys.copy()
            doys_clamped[doys_clamped >= 365] = (
                365  # leap day -> 365, same as elsewhere
            )
            delta_per_t = torch.tensor(delta[doys_clamped - 1], dtype=torch.float32)
            target_mean_before = float(self.target.mean())
            target_std_before = float(self.target.std())
            self.target = self.target + delta_per_t
            print(
                f"  hobday_smooth_target=True: delta min={delta.min():.4f} "
                f"max={delta.max():.4f} RMS={np.sqrt((delta**2).mean()):.4f} degC"
            )
            print(
                f"  target mean/std before={target_mean_before:.4f}/{target_std_before:.4f}  "
                f"after={float(self.target.mean()):.4f}/{float(self.target.std()):.4f}"
            )

        print(f"  Variables:  {self.variables}")
        print(f"  Ocean vars: {sorted(self.ocean_variables)}")
        print(f"  Time steps: {len(self.ds.time)}")
        print(f"  Window: {self.window_size} days, lead: {self.lead_time} days")
        print(f"  Samples: {len(self)}")

        # Normalisation stats, set by compute_stats() after the data is split.
        self.input_means = None
        self.input_stds = None
        self.target_mean = None
        self.target_std = None

    def compute_stats(self, train_indices) -> None:
        """
        Compute input mean/std (ocean pixels only, for ocean variables) and
        target mean/std.

        Input stats are computed over the contiguous slice from the first
        to the last train sample's window, which can include val/test years
        that fall between train years (docs/open_issues.md, #57 P2-2).
        Target stats use only the exact train sample indices.

        Args:
            train_indices: list/array of sample indices in the train set.
        """
        train_indices = list(train_indices)

        t_start = min(train_indices)
        t_end = max(train_indices) + self.window_size

        print(
            f"\nComputing input normalisation stats over the contiguous slice "
            f"[{t_start}:{t_end}] ({t_end - t_start} days, years "
            f"{int(self.years[t_start])}-{int(self.years[t_end - 1])}) -- this "
            f"may include val/test years that fall between train years."
        )
        means, stds = [], []

        for var in self.variables:
            data = self.data[var][t_start:t_end].clone()  # (T, lat, lon)

            if var in self.ocean_variables:
                # Exclude land pixels from the mean/std regardless of
                # land_fill_mode, so normalisation always reflects real
                # ocean variability.
                data[:, self.land_masks[var]] = float("nan")
                mean = float(torch.nanmean(data))
                std = float(torch.std(data[~torch.isnan(data)])) + 1e-8
            else:
                mean = float(data.mean())
                std = float(data.std()) + 1e-8

            means.append(mean)
            stds.append(std)
            print(f"  {var}: mean={mean:.4f}, std={std:.4f}")

        self.input_means = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.input_stds = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)

        # Target stats from training target timestamps only.
        target_idx_list = [
            i + self.window_size - 1 + self.lead_time for i in train_indices
        ]
        train_targets = self.target[torch.tensor(target_idx_list, dtype=torch.long)]
        self.target_mean = float(train_targets.mean())
        self.target_std = float(train_targets.std()) + 1e-8
        print(
            f"  target (train samples only, n={len(train_targets)}): "
            f"mean={self.target_mean:.4f}, std={self.target_std:.4f}"
        )

    def __len__(self) -> int:
        return len(self.ds.time) - self.window_size - self.lead_time + 1

    def __getitem__(self, idx: int):
        """
        Returns (x_spatial, x_temporal, y), plus target_doy if
        self.return_target_doy (order: x_spatial, x_temporal, y, [target_doy]):
            x_spatial:  (window_size, n_vars, lat, lon) — anomalised + normalised
            x_temporal: (window_size, 3)                — year_norm, month_sin, month_cos
            y:          (1,)                            — normalised North Sea SST anomaly
            target_doy: ()                               — day-of-year (1-365) of the
                TARGET day (idx + window_size - 1 + lead_time), for focal-weighted
                loss variants that need to look up a DOY-dependent threshold
                (e.g. Hobday p90). Leap day (366) folded into 365, same
                convention as the climatology-subtraction branch above.
        """
        if self.target_mean is None or self.target_std is None:
            raise RuntimeError(
                "compute_stats(train_indices) has not been called: "
                "target_mean/target_std are None."
            )
        if self.normalize and self.input_means is None:
            raise RuntimeError(
                "normalize=True but compute_stats(train_indices) has not "
                "been called: input_means/input_stds are None."
            )

        window_spatial = []
        window_temporal = []

        for t in range(idx, idx + self.window_size):
            # --- Spatial frame ---
            frame = torch.stack([self.data[v][t] for v in self.variables], dim=0)

            # Land mask (ocean variables only) -- skipped when
            # land_fill_mode="nearest": self.data[var] already has land
            # pixels filled from their nearest ocean neighbor (done once
            # in __init__), so there's nothing left to NaN out here.
            if self.land_fill_mode == "zero":
                for i, var in enumerate(self.variables):
                    if var in self.ocean_variables:
                        frame[i, self.land_masks[var]] = float("nan")

            # Normalise
            if self.normalize:
                frame = (frame - self.input_means) / self.input_stds

            frame = torch.nan_to_num(frame, nan=0.0)
            window_spatial.append(frame)

            # --- Temporal features ---
            year_norm = (self.years[t] - self.year_min) / (
                self.year_max - self.year_min
            )
            month_sin = np.sin(2 * np.pi * self.months[t] / 12)
            month_cos = np.cos(2 * np.pi * self.months[t] / 12)
            window_temporal.append(
                torch.tensor([year_norm, month_sin, month_cos], dtype=torch.float32)
            )

        x_spatial = torch.stack(
            window_spatial, dim=0
        )  # (window_size, n_vars, lat, lon)
        x_temporal = torch.stack(window_temporal, dim=0)  # (window_size, 3)

        target_idx = idx + self.window_size - 1 + self.lead_time
        y_raw = self.target[target_idx]
        y = ((y_raw - self.target_mean) / self.target_std).unsqueeze(0)

        extra = []
        if self.return_target_doy:
            target_doy = int(self.doys[target_idx])
            if target_doy >= 365:
                target_doy = 365
            extra.append(torch.tensor(target_doy, dtype=torch.long))

        return (x_spatial, x_temporal, y, *extra)
