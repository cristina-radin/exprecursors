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
Set `SLURM_MAIL` env var before submitting. Never hardcode an email in submit scripts.

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

```
src/data/     dataset.py, datamodule.py, masking.py
src/models/   cnn_lstm.py
src/xai/      integrated_gradients.py, grad_cam.py, utils.py
src/utils/    checkpoints.py, hobday.py, metrics.py
scripts/      train.py, train_partition.py, eval_*.py, ...
scripts/slurm/ SLURM submit scripts
scripts/analysis/ causal_triangulation.py, check_tau_methodology.py, thermal_inertia_test.py
configs/      kfold/{TbotAtm,SSTAtm}.yaml, partition/{remote,local}.yaml, partition/local|remote/fold0-4.yaml
              # naming: configs/{experiment_type}/{variant}.yaml
tests/        test_splits.py, test_masking.py, test_checkpoints.py
archive/      poster_egu2026/, old scripts
```

## Where things live (read the file, don't reimplement)

| Concern | File |
|---------|------|
| NS box masking — single source | `src/data/masking.py` (`_NS_LAT`, `_NS_LON`, `mask_remote`, `mask_local`) |
| Partition LM classes | `scripts/train_partition.py` |
| Checkpoint selection | `src/utils/checkpoints.py` (`best_ckpt`) |
| Hobday MHW classification | `src/utils/hobday.py` (`load_ns_p90`, `apply_hobday`) |
| Phase-based skill metrics | `src/utils/metrics.py` (`skill_by_phase`) |
| Paper numbers | `results/all_results.csv` |
| Scientific decisions | `docs/narrative.md` |
| Env var definitions | `.env.example` |

---

**Before adding anything not explicitly requested — ask first.**
**Before starting a task, ask me any questions you have.**
**If you consider that there is a common and important rule that is not written here, suggest it**


