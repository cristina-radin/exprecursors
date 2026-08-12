# Contributing to MHW Precursors

See the [README](README.md) for what the project does and how to use it.


## Language

All commit messages, code, and comments must be in English.


## Development setup

Load the HPC environment and activate the project venv:

```bash
module --force purge && module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /path/to/repo/venv/bin/activate
cd /path/to/repo
```

Then install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```


## pip command cheatsheet

| Task | Command |
|------|---------|
| Activate environment | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Freeze current env | `pip freeze > requirements.txt` |
| Run tests | `pytest tests/` |
| Run pre-commit on all files | `pre-commit run --all-files` |
| Train (kfold) | `python train.py --config configs/kfold/TbotAtm.yaml` |
| Train (partition) | `python partition/train_partition.py --config configs/partition/remote.yaml --fold 0` |
| Eval onset skill | `python partition/eval_onset_skill.py --mode remote_only` |


## Data pipeline

All variables in `merged_daily.nc` are **already anomalies**. `dataset.py` does not
subtract any climatology — do not add anomalisation steps there.

### Raw sources

| File | Content | Resolution |
|------|---------|------------|
| `temperature.nc` | ICON-OES-MM-COAST SST | ~0.1° native |
| `ptho_bot_all_data.nc` | ICON bottom temp, prepared by collaborator (u241379) | 0.5° |
| `msl_ssr_daily_1985_2025.nc` | ERA5 MSL + SSR | 0.5° |
| `uv10_daily_1985_2025.nc` | ERA5 U10 + V10 | 0.5° |
| `land_mask_05.nc` | Land mask (1=ocean, 0=land) | 0.5° |

### Step 1 — SST anomaly (`run_compute_anomalies.sh` + `compute_sst_anomalies.py`)

1. CDO `ydrunmean,11` on 1985–2014 → `mean_clim.nc` (DOY mean, ±5-day window)
2. CDO `ydaysub` (full period − mean_clim) → `anom_full.nc`
3. CDO `ydrunpctl,90,11` on reference anomalies → `p90_raw.nc`
4. Python: apply land mask (NaN), save:
   - `sst_climatology_doy.nc` — `mean_clim` + `p90_thresh`, 365 DOYs, 0.1° grid
   - `t_anomalies_daily_v2.nc` — `to_anom` full period, 0.1° grid

Note: `p90_thresh` is stored **without** the 31-day smooth (a comment in the shell
script mentioned it but it was never implemented). The smooth is applied at runtime
in evaluation scripts only.

### Step 2 — Merge (`preprocess_all.py`) → `merged_daily.nc`

- `to_anom`: regrid 0.1° → 0.5° (linear interpolation)
- `ptho_bot`: regrid 0.5° (lat only) + Python DOY climatology (±5 days, ref 1985–2014)
- `msl, ssr, u10, v10`: Python DOY climatology (±5 days, ref 1985–2014)
- `target`: NS basin-mean of `to_anom` at native 0.1° (lat 51–62.5, lon −5.2 to 13.2),
  computed **before** regrid → scalar per timestep
- `land_mask`: from `land_mask_05.nc`, nearest-neighbour interpolation to 0.5°

### Variable definitions

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


## Data

One data file, no exceptions:

```
$MHW_DATA_FILE  (see .env.example)
```

Never use any file ending in `_OLD` or `merged_daily_deepSST.nc`.
Old experiment configs may still point to the wrong file — always override `data_dir`
at runtime, never by editing old config files in `experiments/`.


## Code quality

`pre-commit` runs `black`, `ruff`, and `detect-secrets` on commit.

Run the full suite locally before pushing (should be automatic given the pre-commit hooks):

```bash
pre-commit run --all-files
```

If a hook fails, fix the underlying issue rather than bypassing the hook.
If `detect-secrets` flags a false positive (e.g. a hardcoded HPC path), allowlist inline:

```python
DATA = "/some/sensitive/path/data.nc"  # pragma: allowlist secret
```


## Testing

```bash
pytest tests/
```

Three tests are mandatory — do not merge if any fails:

| Test | What it checks |
|------|----------------|
| `test_splits.py` | No calendar year appears in more than one fold |
| `test_masking.py` | NS box (lat[100:127], lon[150:187]) is exactly zero in remote outputs |
| `test_checkpoints.py` | No two folds produce bit-identical prediction arrays |

New behaviour must ship with a test.


## SLURM

Set `SLURM_ACCOUNT` and `SLURM_MAIL` env vars before submitting (see `.env.example`).
Partition: `booster`.

Two rules before every submission:

1. **Dry-run first.** Test on the login node with a small `n_samples` and `--device cpu`.
   A job that fails in 10 seconds after queuing wastes budget.
2. **Incremental saves.** Any job longer than 1 hour must write one `.npz` per fold
   immediately after that fold finishes. Never accumulate results and save at the end.


## Results

All numbers that appear in the paper must be traceable to a row in:

```
results/all_results.csv
```

Schema: `experiment, model, fold, seed, metric, value, git_commit, date`

Raw arrays (`.npy`, `.npz`) are gitignored. The CSV is versioned.


## Pull requests

- Branch prefix: `feat/`, `fix/`, `docs/`, `chore/`
- Keep each PR focused on one logical change.
- In the PR description, explain *what* changed and *why*.
- Commit messages: imperative mood, focused on the *why*.
- Do not add processing steps, parameters, or plots not explicitly requested —
  ask first if in doubt.
