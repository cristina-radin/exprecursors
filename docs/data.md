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

**Known gap — `mean_clim` is not 31-day smoothed (`known_issues.md` #40)**:
Hobday et al. 2016 prescribes smoothing both the climatological mean and the
p90 threshold with an additional 31-day moving average after the windowed
calculation. `p90_thresh` gets this (at runtime, `load_ns_p90()`) but
`mean_clim` — and therefore `to_anom`/`target` — never has. Quantified: RMS
0.046°C, max 0.131°C (NS-box mean) — real but modest, smaller than current
model MAE. Opt-in fix available without regenerating raw data: config flag
`hobday_smooth_target: true` (`LazyDataset`) applies the exact correction via
`src/utils/hobday.py::load_ns_mean_clim_smooth_delta()`. Default `False` —
no existing experiment's target changes unless a config explicitly opts in.
Regenerating `mean_clim.nc` itself at the source (`preprocess_all.py`,
outside this repo) is a separate, larger-blast-radius decision, not done.

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

**`merged_daily.nc` has TWO land masks, and `land_mask` has a known grid bug**
There are two mask variables: `land_mask` (built from `land_mask_05.nc`, the
SST/atmosphere mask) and `land_mask_tbottom` (built from `land_mask_tbottom_05.nc`,
ptho_bot's own mask). Both follow the same 1=ocean convention, but they are **not**
interchangeable:

- `land_mask` (the SST-derived one) has a confirmed bug: its source file,
  `land_mask_05.nc`, sits on a latitude grid offset by 0.25° from the ERA5
  target grid used by everything else (140 grid points, `[0.25, 0.75, ...]`,
  vs the target's 141 points, `[0.0, 0.5, ...]`). The merge step
  (`preprocess_all.py`) regrids it onto the target grid with nearest-neighbour
  interpolation, which silently snaps this offset — the result is a coastline
  shifted by about one pixel almost everywhere in the domain (572 pixels,
  every coastline, confirmed uniform — not a few bad spots).
- `land_mask_tbottom` does **not** have this problem — its source file is
  already on the correct, ERA5-aligned grid.

Found Aug 18 2026 while investigating spurious coastal sign-flip artifacts in
`ptho_bot`'s signed-IG maps: `dataset.py` was masking `ptho_bot` with the buggy
`land_mask` instead of `land_mask_tbottom`, silently zeroing 572 pixels of real,
physically valid bottom-temperature data on every single training sample. Fixed
in `dataset.py` — `ptho_bot` now uses `land_mask_tbottom` specifically; every
other ocean variable still uses `land_mask` (there currently aren't any others
in the active TbotAtm configs, so this is the variable that actually mattered).

**The root cause (the grid offset in `land_mask_05.nc` itself) is fixed only at
the source script**, `preprocess_all.py` (lives outside this git repo, at
`/p/project1/hai_1127/inputs/daily/preprocess_data/`), so a *future* full
regeneration of `merged_daily.nc` from raw sources will produce a correct
`land_mask` automatically. The *current* `merged_daily.nc` on disk still has
the old, buggy `land_mask` — it was not modified in place (that would be a
high-risk edit to a 9.9 GB file everyone reads from; the safety check blocked
it and the user chose the safer path instead). A corrected copy already exists
at `merged_daily_v2.nc` (same directory): `land_mask` there is replaced with
`land_mask_tbottom`'s values; every other variable (`to_anom`, `ptho_bot`,
`u10`, `v10`, `msl`, `ssr`, `target`, `land_mask_tbottom`) is verified
byte-identical to the original.

**UPDATE Aug 18 2026 — `merged_daily_v2.nc` promoted to canonical for the
`full_mse_v2` re-launch.** The "nothing points to it yet" decision above is
now partially superseded: `configs/partition/full_mse_v2/fold{0-4}.yaml` use
`merged_daily_v2.nc` as `data_dir`, specifically so that `land_mask` and
`land_mask_tbottom` are the same array for every variable (`to_anom`,
`ptho_bot`, ERA5) — no more per-variable mask special-casing risk. Verified
before use: `land_mask` (v2) == `land_mask_tbottom` (v2) exactly (`np.array_equal`
True); every other variable (`to_anom`, `ptho_bot`, `u10`, `v10`, `msl`, `ssr`,
`target`, `land_mask_tbottom`) is byte-identical to `merged_daily.nc` (verified
directly, not assumed); 572 pixels changed in `land_mask` vs v1, all on
coastlines (see comparison PNGs generated Aug 18 2026). **Other configs not
yet migrated** (`configs/partition/full_gnll/*`, `configs/kfold/*`, etc.) still
point at `merged_daily.nc` — migrate them individually when/if they are
relaunched, do not assume the switch is repo-wide. Full writeup and
before/after comparison figures: `known_issues.md` #2 and
`audit_plan.md` → "Pending decision — land_mask grid-offset fix".

**Also relevant to any relaunch**: `known_issues.md` #28 — `dataset.py` was
silently re-subtracting a second, mismatched-window climatology on top of the
already-anomalised variables described above. Fixed Aug 18 2026, same day as
this land-mask migration. Both fixes are in `configs/partition/full_mse_v2/`.
