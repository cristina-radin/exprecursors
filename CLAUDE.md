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

---

## Project structure

```
src/data/     dataset.py, datamodule.py, masking.py
src/models/   cnn_lstm.py
src/xai/      integrated_gradients.py, grad_cam.py, utils.py
src/utils/    checkpoints.py, hobday.py, metrics.py
scripts/      train.py, train_partition.py, eval_*.py, ...
scripts/slurm/ SLURM submit scripts
configs/      kfold/{TbotAtm,SSTAtm}.yaml, partition/{remote,local}.yaml
analysis/     causal_triangulation.py, granger.py, ...
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

---

**Before adding anything not explicitly requested — ask first.**
