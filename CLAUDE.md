# MHW Precursors — Claude reference

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, commands, testing, and PR conventions.

---

## Critical reminders (always apply, no exceptions)

**Data file**
Only: `/p/project1/hai_1127/inputs/daily/preprocess_data/merged_daily.nc`
Never: anything with `_OLD` or `merged_daily_deepSST.nc`.

**Incremental saves**
Any SLURM job longer than 1 hour must save one `.npz` per fold immediately
after that fold finishes. Never accumulate results and save at the end.

**SLURM email**
Always: `cristina.radin@uni-hamburg.de` — never the gmail address.

---

## Where things live (read the file, don't reimplement)

| Concern | File |
|---------|------|
| NS box masking (remote / local) | `partition/train_partition.py` (`_NS_LAT`, `_NS_LON`) |
| Checkpoint selection (best val_loss) | `eval/ensemble_skill.py` (`_best_ckpt`) |
| Hobday MHW classification | `eval/composite_ig_hobday.py` (`load_ns_p90`) |
| Paper numbers | `results/all_results.csv` |
| Scientific decisions | `docs/narrative.md` |

---

**Before adding anything not explicitly requested — ask first.**
