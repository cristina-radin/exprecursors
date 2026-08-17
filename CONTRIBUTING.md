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

### Scalar pipeline (NS point target)

| Task | Command |
|------|---------|
| Activate environment | `source venv/bin/activate` |
| Install dependencies | `pip install -r requirements.txt` |
| Freeze current env | `pip freeze > requirements.txt` |
| Run tests | `pytest tests/` |
| Run pre-commit on all files | `pre-commit run --all-files` |
| Train (kfold) | `python train.py --config configs/kfold/TbotAtm.yaml` |
| Train (partition) | `python scripts/train_partition.py --config configs/partition/remote.yaml --mode remote_only` |
| Eval onset skill | `python scripts/eval_onset_skill.py --mode remote_only` |

### Spatial pipeline (2D to_anom field target)

Two parallel pipelines share the same repo and `results/all_results.csv`.
Experiment outputs (checkpoints, `test_preds.npy`) live outside the repo at
`/p/project1/hai_1127/radin1/spatial_forecast/experiments/` (heavy, gitignored).

| Task | Command |
|------|---------|
| Train spatial (standard) | `python scripts_spatial/train_spatial.py --config configs/spatial/SSTAtm_fold0.yaml` |
| Train spatial (physics loss) | `python scripts_spatial/train_spatial_phys.py --config configs/spatial/SSTAtm_phys_fold0.yaml` |
| Eval persistence baseline | `python scripts_spatial/eval/persistence_baseline_spatial.py` |
| Eval MHW onset maps | `python scripts_spatial/eval/mhw_onset_skill.py` *(see known_issues.md NF-S-5 — MHW criterion invalid)* |

Both pipelines share: `docs/known_issues.md`, `results/all_results.csv`.


## Data

One data file, no exceptions:

```
$MHW_DATA_FILE  (see .env.example)
```

Never use any file ending in `_OLD` or `merged_daily_deepSST.nc`.
Old experiment configs may still point to the wrong file — always override `data_dir`
at runtime, never by editing old config files in `experiments/`.

Full pipeline, raw sources, and variable definitions: see [`docs/data.md`](docs/data.md).


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

> **Gotcha**: the hook scans *staged* content, not the file on disk. If you add the
> pragma after running `git add`, you must `git add` the file again so the pragma
> enters the index — otherwise the hook will still flag the old staged version.


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
