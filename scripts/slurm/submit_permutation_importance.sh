#!/bin/bash
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=perm_imp
#SBATCH --output=slurm-perm_imp-%x-%j.out
#SBATCH --error=slurm-perm_imp-%x-%j.err
#SBATCH --mail-type=FAIL,END

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set." >&2
    exit 1
fi

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

FOLD="${FOLD:-0}"
PARTITION="${PARTITION:-full}"
MODEL_TAG="${MODEL_TAG:-mse_v2}"

echo "Fold: ${FOLD}  Partition: ${PARTITION}  Model: ${MODEL_TAG}  Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/permutation_importance.py \
    --fold "${FOLD}" \
    --partition "${PARTITION}" \
    --model_tag "${MODEL_TAG}" \
    --batch_size 64

echo "End: $(date)"
