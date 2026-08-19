#!/bin/bash
# Submit TbotAtm Full GNLL+quantile-head partition training, all 5 folds.
#
# Adapted from submit_gnll_partition.sh — only the config path and job-name
# changed. Ported from JUWELS to MPCDF Raven Aug 19 2026 (JUWELS had no
# accessible data/venv for this session): partition=booster -> gpu1 (single
# A100, node-shareable, matches JUWELS' 1-GPU request), module stack
# Stages/2025+GCCcore+Python/3.12.3 (JSC) -> python-waterboa/2024.06 (MPCDF,
# also Python 3.12.x). Account is mmm_gpu (see `sacctmgr show user`), not
# mmm_cpu (the login default) — GPU partitions require the GPU account.
# If submitting on a different cluster again, replace partition/module/
# account for that system before use. Fold0 of the plain full_gnll run took
# ~1h01m wall time on JUWELS (same architecture minus the aux head) — but
# measured directly on this Raven A100 (job 29403121, fold0_shorttest, Aug
# 19 2026) training runs ~5.4 min/epoch (589 batches/epoch @ ~1.83 it/s).
# With EarlyStopping(patience=30), a slow-to-converge fold could take
# 3-4+ hours, well past the JUWELS estimate — --time=12:00:00 leaves real
# margin instead of assuming the JUWELS number transfers directly.
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# "${VAR}" syntax there is NOT expanded. Account/mail-user must come from
# the SBATCH_ACCOUNT / SBATCH_MAIL_USER env vars (recognised by sbatch as
# option defaults), or be passed on the CLI.
#
# Usage:
#   source .env   # must have WANDB_ENTITY and WANDB_PROJECT set
#   export SBATCH_ACCOUNT=mmm_gpu SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL scripts/slurm/submit_gnll_quantile_partition.sh
# (or: sbatch --account=... --mail-user=... --export=ALL ...)

#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=partition_gnllq
#SBATCH --array=0-4
#SBATCH --output=slurm-partition-%x-%A_%a.out
#SBATCH --error=slurm-partition-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Export SBATCH_ACCOUNT or pass --account on the CLI." >&2
    exit 1
fi

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/full_gnll_quantile/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "GNLL+quantile Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
