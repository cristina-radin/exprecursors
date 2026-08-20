#!/bin/bash
# Submit partition training (remote_only or local_only), all 5 folds.
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# ${VAR} does NOT get expanded there. --account and --mail-user must be
# passed on the sbatch CLI instead.
#
# Usage:
#   source .env          # sets MHW_DATA_FILE, MHW_EXPERIMENTS_DIR, etc.
#   sbatch --account=your_account --mail-user=you@example.com --export=ALL,MODE=remote_only scripts/slurm/submit_train.sh
#   sbatch --account=your_account --mail-user=you@example.com --export=ALL,MODE=local_only  scripts/slurm/submit_train.sh

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

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/${MODE//_only/}/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "Mode: ${MODE}  Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode "${MODE}"

echo "End: $(date)"
