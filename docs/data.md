# Data pipeline

All variables in `merged_daily.nc` are **already anomalies**. `dataset.py` does not
subtract any climatology — do not add anomalisation steps there.

## Raw sources

| File | Content | Resolution |
|------|---------|------------|
| `temperature.nc` | ICON-OES-MM-COAST SST | ~0.1° native |
| `ptho_bot_all_data.nc` | ICON bottom temp, prepared by collaborator (u241379) | 0.5° |
| `msl_ssr_daily_1985_2025.nc` | ERA5 MSL + SSR | 0.5° |
| `uv10_daily_1985_2025.nc` | ERA5 U10 + V10 | 0.5° |
| `land_mask_05.nc` | Land mask (1=ocean, 0=land) | 0.5° |

## Step 1 — SST anomaly (`run_compute_anomalies.sh` + `compute_sst_anomalies.py`)

1. CDO `ydrunmean,11` on 1985–2014 → `mean_clim.nc` (DOY mean, ±5-day window)
2. CDO `ydaysub` (full period − mean_clim) → `anom_full.nc`
3. CDO `ydrunpctl,90,11` on reference anomalies → `p90_raw.nc`
4. Python: apply land mask (NaN), save:
   - `sst_climatology_doy.nc` — `mean_clim` + `p90_thresh`, 365 DOYs, 0.1° grid
   - `t_anomalies_daily_v2.nc` — `to_anom` full period, 0.1° grid

Note: `p90_thresh` is stored **without** the 31-day smooth (a comment in the shell
script mentioned it but it was never implemented). The smooth is applied at runtime
in evaluation scripts only.

## Step 2 — Merge (`preprocess_all.py`) → `merged_daily.nc`

- `to_anom`: regrid 0.1° → 0.5° (linear interpolation)
- `ptho_bot`: regrid 0.5° (lat only) + Python DOY climatology (±5 days, ref 1985–2014)
- `msl, ssr, u10, v10`: Python DOY climatology (±5 days, ref 1985–2014)
- `target`: NS basin-mean of `to_anom` at native 0.1° (lat 51–62.5, lon −5.2 to 13.2),
  computed **before** regrid → scalar per timestep
- `land_mask`: from `land_mask_05.nc`, nearest-neighbour interpolation to 0.5°

## Variable definitions

**`to_anom` (model target)**
Standard daily SST anomaly: SST minus the daily climatological mean (±5-day window
per day-of-year, reference period 1985–2014). Zero-centered.
Never describe `to_anom` as "SST minus P90" — the model predicts a continuous anomaly value.

**`target`** — NS basin-mean of `to_anom` at 0.1° native resolution. Scalar per day.
`dataset.py` additionally applies Z-score normalisation (training stats only) and
optional linear detrending (`detrend_target`, default False).

**MHW classification — post-processing only**
The P90 threshold (`p90_thresh`, stored in `sst_climatology_doy.nc`) is used only to
classify predicted `to_anom` values as MHW days after inference. It is not part of
the model target. A day is MHW when `to_anom > p90_thresh(DOY)`, with:
- persistence: ≥5 consecutive days above threshold
- gap merging: gaps ≤2 days filled (Hobday et al. 2016)

Note: `eval_onset_skill.py::load_ns_p90()` applies a 31-day circular smooth to the
spatially-averaged NS `p90_thresh` at inference time. This is not in the stored data.

Equivalence: `sst > p90(sst, DOY)` ⟺ `anom > p90(anom, DOY)`.

**`land_mask` (inverted convention)**
In `merged_daily.nc`, `land_mask = 1` means **ocean**, `land_mask = 0` means **land** —
opposite of standard boolean convention. `dataset.py` inverts this at load time:
```python
self.is_land = (land_mask == 0)   # True where land → set to NaN
```
Do not use `land_mask` directly as a boolean without inverting.
