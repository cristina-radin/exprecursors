# Data

## Files (Levante)

| File | Content | `.env` variable |
|---|---|---|
| `/work/bg1446/u241379/data/merged_daily.nc` | model inputs and target, daily, 1985-2024, 0.5° grid (141 x 201) | `MHW_DATA_FILE` |
| `/work/bg1446/u241379/data/sst_climatology_doy.nc` | SST climatology (`mean_clim`) and p90 (`p90_thresh`) per day of year, native 0.1° grid. The code only reads `mean_clim` (target correction); `p90_thresh` is not used. | `MHW_CLIM_FILE` |

Both files were built outside this repo. Preprocessing scripts: `/work/bg1446/u241379/data/preprocessing/`.

## Variables in `merged_daily.nc`

| Variable | Source | Used as |
|---|---|---|
| `ptho_bot` | ICON, bottom temperature | input (the only ocean variable) |
| `u10`, `v10`, `msl`, `ssr` | ERA5 | inputs |
| `target` | ICON SST | target (see below) |
| `to_anom` | ICON SST anomaly | not used by the model |
| `land_mask`, `land_mask_tbottom` | masks | see below |

All variables are **already anomalies** (daily climatology removed). The code does not
subtract any climatology from the inputs.

## Target

`target` = mean SST anomaly over the North Sea box (51-62.5 N, 5.2 W-13.2 E), computed at
the native 0.1° resolution: one value per day. With `hobday_smooth_target: true` (set in
all configs in `configs/`), the code adds a small correction so that the anomaly is taken
with respect to the 31-day smoothed climatology, as in Hobday et al. (2016)
(`src/utils/hobday.py`). The correction is approximate: it averages the climatology over
lat 50-63 N, lon 5 W-13 E without area weighting, not exactly over the target box (known
mismatch, see the project notes).

The model predicts the target 7 days ahead (`lead_time`) from the previous 60 days of
inputs (`window_size`).

## Normalisation (in `src/data/dataset.py`)

z-score with one mean and one standard deviation per variable, computed only from the
training days of each fold. Land pixels of `ptho_bot` are first filled with the nearest
ocean value when the data is loaded (`land_fill_mode: nearest`); they are then left out of
the mean and standard deviation. The target is normalised the same way; predictions are
saved both normalised and in °C.

## Land masks

Convention: `1 = ocean`, `0 = land` (the opposite of a usual boolean "is land").

The file has two masks. `land_mask` is shifted by about one pixel along the coasts (it was
built from a source mask on a grid offset by 0.25°); 572 coastal pixels differ from
`land_mask_tbottom`. The code uses `land_mask_tbottom` for `ptho_bot`, the
only ocean variable, so current runs are not affected. **Any new ocean variable must not
use `land_mask`.**

## Marine heatwaves

Hobday et al. (2016), in `src/utils/hobday.py`. The threshold is computed from the series
itself: for each day of year, the p90 of a +-5-day window pooled over 1985-2014, then a
31-day periodic smoothing. A day is above threshold when the anomaly is strictly greater.
Runs shorter than 5 days are dropped first; the remaining events are then merged across
gaps of up to 2 days.
