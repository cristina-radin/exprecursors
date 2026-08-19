#!/bin/bash
# Batched test-set inference for TbotAtm_{PARTITION}_{MODEL_TAG}_seed42_fold{0-4}
# — produces per-fold test_predictions.npz plus one 5-fold comparison figure
# (predicted vs. true to_anom time series). Forward pass only, no training,
# no gradients — single short job, not an array (loops over the 5 folds
# internally in scripts/eval_partition_timeseries.py).
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# "${VAR}" syntax there is NOT expanded. Account/mail-user must come from
# the SBATCH_ACCOUNT / SBATCH_MAIL_USER env vars (recognised by sbatch as
# option defaults), or be passed on the CLI.
#
# Usage:
#   source .env
#   export SBATCH_ACCOUNT=your_account SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL,PARTITION=full,MODEL_TAG=mse_v2 scripts/slurm/submit_eval_partition_timeseries.sh

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_ts
#SBATCH --output=slurm-eval_ts-%x-%j.out
#SBATCH --error=slurm-eval_ts-%x-%j.err
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

PARTITION="${PARTITION:-full}"
MODEL_TAG="${MODEL_TAG:-}"

echo "Partition: ${PARTITION}  Model tag: ${MODEL_TAG:-(none)}  Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/eval_partition_timeseries.py \
    --partition "${PARTITION}" \
    --model_tag "${MODEL_TAG}"

echo "End: $(date)"
