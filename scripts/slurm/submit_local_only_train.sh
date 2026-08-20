#!/bin/bash
# Submit local-only training, all 5 folds.
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# ${VAR} does NOT get expanded there. --account and --mail-user must be
# passed on the sbatch CLI instead.
#
# Usage:
#   source .env
#   sbatch --account=your_account --mail-user=you@example.com scripts/slurm/submit_local_only_train.sh

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=local_only_tr
#SBATCH --array=0-4
#SBATCH --output=slurm-local_only-%x-%A_%a.out
#SBATCH --error=slurm-local_only-%x-%A_%a.err
#SBATCH --mail-type=FAIL

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/local.yaml"

echo "Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_local_only.py --config "${CFG}" --fold "${SLURM_ARRAY_TASK_ID}"

echo "End: $(date)"
