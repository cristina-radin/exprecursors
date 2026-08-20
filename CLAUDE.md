# MHW Precursors — Claude reference

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, commands, testing, and PR conventions.

---

## Critical reminders (always apply, no exceptions)

**Data file**
Set `MHW_DATA_FILE` env var pointing to `merged_daily.nc` (see `.env.example`).
Never: anything with `_OLD` or `merged_daily_deepSST.nc`.

**Incremental saves**
Any SLURM job longer than 1 hour must save one `.npz` per fold immediately
after that fold finishes. Never accumulate results and save at the end.

**SLURM email**
`#SBATCH` directive lines are read literally by sbatch, NOT through the shell —
`${VAR}` inside a `#SBATCH` line is never expanded (confirmed empirically,
see known_issues.md). Never write `#SBATCH --mail-user=${SOME_VAR}` — it
silently submits the literal unexpanded string as the address and no mail
is ever sent. Always pass `--mail-user=you@example.com` (and `--account=...`)
directly on the `sbatch` command line instead. Never hardcode an email inside
a submit script's `#SBATCH` block.

## Code discipline
- No silent fallbacks: if a value can't be parsed/found, raise an error with a
  clear message — never default to 0.0, None, or "skip" without printing a warning.
- No broad exception catching — let errors surface.
- Vectorize: prefer PyTorch/NumPy/xarray batch operations over per-sample Python
  loops, especially for IG/inference (the 2,400 core-hour loss on Aug 11 came from
  an unbatched per-sample loop).
- Don't add new dependencies to requirements.txt without asking first.

---

## Project structure

Two parallel pipelines share the same repo, data file, and results table:

### Scalar pipeline (NS point target)
```
src/data/     dataset.py, datamodule.py, masking.py
src/models/   cnn_lstm.py
src/xai/      integrated_gradients.py, grad_cam.py, utils.py
src/utils/    checkpoints.py, hobday.py, metrics.py
scripts/      train.py, train_partition.py, eval_*.py, ...
scripts/slurm/ SLURM submit scripts
scripts/analysis/ causal_triangulation.py, check_tau_methodology.py, thermal_inertia_test.py
configs/kfold/   {TbotAtm,SSTAtm}.yaml
configs/partition/ {remote,local}.yaml, local|remote/fold0-4.yaml
tests/        test_splits.py, test_masking.py, test_checkpoints.py
```

### Spatial pipeline (2D to_anom field target)
```
src_spatial/  dataset_spatial.py, dataset_spatial_phys.py,
              model_spatial.py, model_spatial_phys.py
scripts_spatial/           train_spatial.py, train_spatial_phys.py
scripts_spatial/eval/      mhw_onset_skill.py, persistence_baseline_spatial.py
scripts_spatial/preprocessing/ compute_mld_weights.py
configs/spatial/ SSTAtm_fold0.yaml, TbotAtm_fold0.yaml, SSTAtm_phys_fold0.yaml
```
Experiment outputs (checkpoints, test_preds.npy) live in:
`/p/project1/hai_1127/radin1/spatial_forecast/experiments/` (heavy, not in git).
Both pipelines share: `docs/known_issues.md`, `results/all_results.csv`.

## Where things live (read the file, don't reimplement)

| Concern | File |
|---------|------|
| NS box masking — single source | `src/data/masking.py` (`_NS_LAT`, `_NS_LON`, `mask_remote`, `mask_local`) |
| Partition LM classes | `scripts/train_partition.py` |
| Checkpoint selection | `src/utils/checkpoints.py` (`best_ckpt`) |
| Hobday MHW classification | `src/utils/hobday.py` (`load_ns_p90`, `apply_hobday`) |
| Phase-based skill metrics | `src/utils/metrics.py` (`skill_by_phase`) |
| Spatial dataset | `src_spatial/dataset_spatial.py` |
| Spatial model (standard) | `src_spatial/model_spatial.py` |
| Spatial model (physics) | `src_spatial/model_spatial_phys.py` |
| Paper numbers | `results/all_results.csv` |
| Scientific decisions | `docs/narrative.md` |
| Env var definitions | `.env.example` |

---

**Before adding anything not explicitly requested — ask first.**
**Before starting a task, ask me any questions you have.**
**If you consider that there is a common and important rule that is not written here, suggest it**


