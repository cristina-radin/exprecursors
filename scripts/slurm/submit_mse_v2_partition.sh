#!/bin/bash
# Submit TbotAtm Full MSE partition training, all 5 folds.
# Uses merged_daily_v2.nc (unified land_mask == land_mask_tbottom, known_issues.md #2)
# and MSE loss (gaussian_nll: false) — sanity-check run alongside the GNLL
# architecture while GNLL trust is being re-established, and to validate the
# double-anomalisation climatology fix (known_issues.md #28) on real training.
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# "${VAR}" syntax there is NOT expanded. Account/mail-user must come from
# the SBATCH_ACCOUNT / SBATCH_MAIL_USER env vars (recognised by sbatch as
# option defaults), or be passed on the CLI.
#
# Usage:
#   source .env   # must have WANDB_ENTITY and WANDB_PROJECT set
#   export SBATCH_ACCOUNT=your_account SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL scripts/slurm/submit_mse_v2_partition.sh
# (or: sbatch --account=... --mail-user=... --export=ALL ...)

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=partition
#SBATCH --array=0-4
#SBATCH --output=slurm-partition-%x-%A_%a.out
#SBATCH --error=slurm-partition-%x-%A_%a.err
#SBATCH --mail-type=FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Export SBATCH_ACCOUNT or pass --account on the CLI." >&2
    exit 1
fi

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/full_mse_v2/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "MSE+v2 Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
