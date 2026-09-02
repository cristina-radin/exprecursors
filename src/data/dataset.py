"""
Dataset class for daily climate data (MHW precursor detection).

Input:  sliding window of `window_size` days → [window_size, n_vars, lat, lon]
Target: mean SST anomaly over North Sea at t + lead_time (scalar, normalized)

Anomalisation:
  All variables in merged_daily.nc (to_anom, ptho_bot, u10, v10, msl, ssr) are
  already anomalies — climatology (day-of-year mean, ±5-day window, ref
  1985-2014) is removed upstream in preprocess_all.py. This module does NOT
  subtract any further climatology (see docs/data.md). A prior version of this
  docstring/code re-subtracted a second, mismatched-window (±2-day) climatology
  for every variable except to_anom — found and removed 2026-08-18
  (known_issues.md, double-anomalisation bug).
"""

import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset

# NOTE: src.utils.hobday is intentionally imported lazily inside __init__
# (only when hobday_smooth_target=True), not at module level here — it
# transitively requires the MHW_CLIM_FILE env var (src.utils.paths.CLIM_FILE
# raises at import time if unset), and dataset.py must stay importable
# without that var for every existing config that doesn't use this flag.


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
        config_path: str = "config.yaml",
    ):
        super().__init__()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.file_name = file_name
        self.variables = config["variables"]
        self.ocean_variables = set(config.get("ocean_variables", self.variables))
        self.normalize = config.get("normalize", True)
        self.window_size = config.get("window_size", 60)
        self.lead_time = config.get("lead_time", 7)
        self.clim_ref_start = config.get("clim_ref_start", 1985)
        self.clim_ref_end = config.get("clim_ref_end", 2014)
        self.clim_window = config.get("clim_window", 5)
        self.detrend_variables = set(config.get("detrend_variables", []))
        self.detrend_target = config.get("detrend_target", False)
        # Opt-in: __getitem__ returns a 4-tuple (..., target_doy) instead of
        # the standard 3-tuple when True. Off by default so the ~20 existing
        # call sites that unpack `xs, xt, y = ...` (scripts/, tests/) are
        # unaffected — only a loss variant that needs a DOY-dependent
        # threshold (e.g. focal-weighted NLL) should set this.
        self.return_target_doy = config.get("return_target_doy", False)
        # Opt-in: __getitem__ additionally returns a normalized "state"
        # scalar (Aug 23 2026) -- the target's own value at the LAST day
        # of the input window (index i+window_size-1), i.e. exactly what
        # lag-persistence uses as its prediction, normalized the same way
        # as y. The model never otherwise sees the target itself as input
        # (only ptho_bot, a different variable) -- this tests whether
        # giving the network explicit access to "today's true state"
        # beats the post-hoc linear hybrid (docs/narrative.md, Aug 23
        # 2026 incremental-value-regression entry) via nonlinear/
        # state-dependent combination. Off by default, same
        # backward-compatibility reasoning as return_target_doy above.
        # Mutually orthogonal to return_target_doy -- both can be True at
        # once if ever needed (not currently exercised by any config).
        self.use_state_feature = config.get("use_state_feature", False)

        self.ds = xr.open_mfdataset(file_name, parallel=True, engine="netcdf4")

        # Temporal coordinates
        self.years = self.ds.time.dt.year.values
        self.months = self.ds.time.dt.month.values
        self.doys = self.ds.time.dt.dayofyear.values  # 1–366
        self.year_min = self.years.min()
        self.year_max = self.years.max()

        # Land mask: True where land — only applied to ocean variables.
        # ptho_bot has its OWN native land/ocean boundary (land_mask_tbottom),
        # 572 pixels different from the SST/atmosphere land_mask (known_issues.md
        # #2). Using land_mask for ptho_bot falsely zeros 572 real ocean pixels,
        # creating an artificial land/ocean edge in the input the CNN reacts to.
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

        # Opt-in: instead of the default hard land=0 (a spatially-flat
        # constant region butting up against real, spatially-varying
        # ocean values -- an artificial edge a CNN can trivially learn to
        # detect, independent of any real precursor physics; see the
        # comment above this block, and known_issues.md), fill land
        # pixels with their nearest-ocean-neighbor's value at every
        # timestep. Removes the flat-region cliff without inventing fake
        # bathymetry -- the land value simply extends the nearest real
        # ocean reading. Default "zero" (old behavior) -- opt-in like
        # every other flag added this project (pooling, padding_mode,
        # hobday_smooth_target, ...).
        self.land_fill_mode = config.get("land_fill_mode", "zero")
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

        # Opt-in: mean_clim.nc's mean_clim never received the 31-day smooth
        # Hobday et al. 2016 also prescribes for the climatology mean (only
        # p90_thresh gets it, at runtime — see known_issues.md). Correcting
        # this shifts `target` itself (not just MHW classification), so it
        # must run before compute_stats() computes target_mean/target_std.
        # Default False — no existing config's target changes unless it
        # explicitly opts in (same pattern as pooling/padding_mode).
        self.hobday_smooth_target = config.get("hobday_smooth_target", False)
        if self.hobday_smooth_target:
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

        # Stats — overwritten by compute_stats()
        self.clim_means = {}  # var → (365, lat, lon) climatology
        self.trend_slopes = {}  # var → (lat, lon)  linear trend slope [units/day]
        self.trend_ref_t = 0  # reference timestep index for trend centering
        self.input_means = None
        self.input_stds = None
        self.target_mean = 0.0
        self.target_std = 1.0

    # ------------------------------------------------------------------
    # Climatology
    # ------------------------------------------------------------------

    def _compute_clim(self) -> None:
        """
        No-op: every variable in merged_daily.nc is already a day-of-year
        anomaly, computed upstream in preprocess_all.py (see docs/data.md).
        self.clim_means is left empty, so __getitem__ and compute_stats()
        skip the climatology-subtraction branch entirely.

        Kept as a method (rather than deleted outright) so a future dataset
        file that ships absolute values can restore per-variable climatology
        by populating vars_to_anom again.
        """
        vars_to_anom = []
        if not vars_to_anom:
            return

        ref_mask = (self.years >= self.clim_ref_start) & (
            self.years <= self.clim_ref_end
        )
        ref_doys = self.doys[ref_mask].copy()
        ref_doys[ref_doys == 366] = 365  # map leap days → Dec 31

        half_w = self.clim_window // 2

        print(
            f"\nComputing day-of-year climatology "
            f"({self.clim_ref_start}-{self.clim_ref_end}, "
            f"window={self.clim_window})..."
        )

        for var in vars_to_anom:
            ref_data = self.data[var][ref_mask].numpy()  # (n_ref, lat, lon)
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
        self.trend_ref_t = float(t.mean())  # center to avoid large intercepts

        print(
            f"\nComputing pixel-wise linear trend for: {sorted(self.detrend_variables)}"
        )

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
            t_var = float((t_c**2).mean())
            slopes = np.zeros((lat, lon), dtype=np.float32)
            for i in range(lat):
                for j in range(lon):
                    pixel = data[:, i, j]
                    if np.isfinite(pixel).all():
                        slopes[i, j] = float((t_c * pixel).mean()) / t_var

            self.trend_slopes[var] = torch.tensor(slopes, dtype=torch.float32)
            print(
                f"  {var}: mean slope={slopes.mean():.6f}/day  "
                f"({slopes.mean()*3650:.4f} per decade)"
            )

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
        t_end = max(train_indices) + self.window_size

        print("\nComputing normalisation stats from train data only...")
        means, stds = [], []

        for var in self.variables:
            data = self.data[var][t_start:t_end].clone()  # (T, lat, lon)

            # Subtract day-of-year climatology for ERA5 variables
            if var in self.clim_means:
                abs_ts = np.arange(t_start, t_start + len(data))
                doys = self.doys[abs_ts].copy()
                doys[doys == 366] = 365
                for i, doy in enumerate(doys):
                    data[i] -= self.clim_means[var][doy - 1]

            if var in self.ocean_variables:
                # Normalization stats must always reflect the true ocean
                # data distribution, excluding land, regardless of
                # land_fill_mode -- land_fill_mode only controls what
                # value the MODEL sees at land positions in the input
                # tensor (0 vs. nearest-ocean-neighbor), it must not
                # change what "typical ocean variability" means. Bug
                # found Aug 21 2026 (user caught it): this branch used to
                # gate on land_fill_mode == "zero", so land_fill_mode=
                # "nearest" fell into the `else` branch below and
                # computed mean/std over ALL pixels including the 9473
                # land pixels (now filled with copied ocean-neighbor
                # values) -- inflated ptho_bot's std by +23% (0.2775 ->
                # 0.3425, confirmed against the saved training logs) and
                # shifted its mean, silently attenuating every real ocean
                # anomaly by ~19% for every land_fill_mode=nearest run so
                # far. See known_issues.md and docs/narrative.md.
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

        # Detrend target (full period, pixel-wise scalar)
        if self.detrend_target:
            T = len(self.target)
            t = np.arange(T, dtype=np.float64)
            t_c = t - t.mean()
            target_np = self.target.numpy().astype(np.float64)
            slope = float((t_c * target_np).mean() / (t_c**2).mean())
            self.target_trend_slope = slope
            self.target_trend_ref_t = float(t.mean())
            self.target = torch.tensor(target_np - slope * t_c, dtype=torch.float32)
            print(f"  target detrended: slope={slope*3650:.4f}/decade")
        else:
            self.target_trend_slope = 0.0
            self.target_trend_ref_t = 0.0

        # Target stats from training target timestamps
        target_idx_list = [
            i + self.window_size - 1 + self.lead_time for i in train_indices
        ]
        train_targets = self.target[torch.tensor(target_idx_list, dtype=torch.long)]
        self.target_mean = float(train_targets.mean())
        self.target_std = float(train_targets.std()) + 1e-8
        print(f"  target: mean={self.target_mean:.4f}, std={self.target_std:.4f}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ds.time) - self.window_size - self.lead_time + 1

    def __getitem__(self, idx: int):
        """
        Returns (x_spatial, x_temporal, y), plus target_doy if
        self.return_target_doy, plus state if self.use_state_feature
        (order: x_spatial, x_temporal, y, [target_doy], [state]):
            x_spatial:  (window_size, n_vars, lat, lon) — anomalised + normalised
            x_temporal: (window_size, 3)                — year_norm, month_sin, month_cos
            y:          (1,)                            — normalised North Sea SST anomaly
            target_doy: ()                               — day-of-year (1-365) of the
                TARGET day (idx + window_size - 1 + lead_time), for focal-weighted
                loss variants that need to look up a DOY-dependent threshold
                (e.g. Hobday p90). Leap day (366) folded into 365, same
                convention as the climatology-subtraction branch above.
            state:      (1,)                            — the target's own
                normalised value at the LAST day of the input window
                (idx + window_size - 1), i.e. exactly what lag-persistence
                uses as its prediction. Normalised with the SAME
                target_mean/target_std as y (same physical quantity, same
                scale) -- not a new leak, this day is always strictly
                before the target day since lead_time >= 1.
        """
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
        if self.use_state_feature:
            state_idx = idx + self.window_size - 1
            state_raw = self.target[state_idx]
            state = ((state_raw - self.target_mean) / self.target_std).unsqueeze(0)
            extra.append(state)

        return (x_spatial, x_temporal, y, *extra)
