# MHW precursors — Claude reference

See README.md for setup and commands.

## Rule for this branch (`rebuild`)
Nothing enters without the owner's review. Every file must answer "which number
or figure of the paper comes out of here?". Exploratory work goes in `sandbox/`.
No code change without the owner's explicit approval in that same turn, even if
it belongs to a step already described. Every code change in `rebuild` must
pass `tools/equivalence.py --check` before commit. If it does not pass, stop
and report. One test file per source file. Parametrise inside it; do not add
test files per experiment. A new experiment is a new yaml, never a new script.
No new script without first stating which paper figure or number it produces.
Budget: 15-20 code files for the whole paper. If a change would exceed it,
stop and ask. The equivalence harness only proves that the trained model's
predictions are unchanged. Any change to training code also needs the smoke
test.

## Critical conventions
- **Valid data**: `$MHW_DATA_FILE` -> `merged_daily.nc`. Never a file with `_OLD`
  in its name, never `merged_daily_deepSST.nc`.
- **Git**: `origin` is GitLab Codebase (`https://codebase.helmholtz.cloud/hereon-ksn/exprecursors.git`).
  Never push to GitHub: it is a read-only mirror fed by Codebase. **Never push.
  The owner pushes manually** — do not assume git credentials (SSH key,
  `~/.git-credentials`) are configured on whatever machine this session runs
  on; they usually are not, and that is expected, not a problem to fix.
- **SLURM**: `#SBATCH` lines are read literally, `${VAR}` is never expanded
  there; pass `--account` and `--mail-user` on the `sbatch` command line. Any job
  longer than 1 h saves one `.npz` per fold as soon as that fold ends, never at
  the end.
- **No silent fallbacks**: if a value cannot be parsed or found, raise with a
  clear message. No broad `except`.
- **Reporting skill**: always say whether r is mean-of-folds or pooled across
  folds; they are not interchangeable.
- **Losses**: never compare models trained with different losses (e.g. GNLL vs
  MSE) without saying so; the difference is confounded by optimisation.
- **Relaunching a run (#58)**: never train into an `output_dir` that already has
  checkpoints. `best_ckpt()` picks the lowest `val_loss` across the whole
  directory and can silently return a checkpoint of the old run. Use a clean one.
- **Split RNG (#23)**: `np.random.RandomState(seed)` and
  `np.random.default_rng(seed)` give different permutations from the same seed.
  Do not swap one for the other in split code.
- Do not add dependencies to `requirements.txt` without asking.
