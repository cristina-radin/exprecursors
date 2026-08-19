#!/bin/bash
# PILOT run: Grad-CAM on TbotAtm_{PARTITION}_{MODEL_TAG}_seed42_fold{FOLD}.
#
# Same account constraint as submit_ig_partition_smoothgrad_pilot.sh — hai_1127
# has NO CPU partition on JUWELS, GPU is not a cost lever here.
#
# Purpose: measure real per-sample GPU cost of gradcam_partition.py (one
# forward+backward per sample, unbatched loop — see script docstring) on a
# small MAX_SAMPLES subsample BEFORE committing to a full fold or all 5.
#
# Usage:
#   source .env
#   export SBATCH_ACCOUNT=your_account SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL,PARTITION=full,MODEL_TAG=mse_v2,FOLD=0,MAX_SAMPLES=50 \
#       scripts/slurm/submit_gradcam_pilot.sh

#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=gradcam_pilot
#SBATCH --output=slurm-gradcam_pilot-%x-%j.out
#SBATCH --error=slurm-gradcam_pilot-%x-%j.err
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
# NOTE: no colon here on purpose. "${MODEL_TAG:-mse_v2}" (with colon) treats
# an explicitly-passed empty string the same as unset and silently falls back
# to "mse_v2" — that bug sent 10 remote/local jobs to the wrong (mse_v2)
# checkpoint set on 2026-08-19 before being caught. "${MODEL_TAG-mse_v2}"
# (no colon) only substitutes when the var is truly unset, so an intentional
# empty MODEL_TAG= (untagged remote/local checkpoints) is preserved.
MODEL_TAG="${MODEL_TAG-mse_v2}"
FOLD="${FOLD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/gradcam_partition_full_pilot}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"

echo "PILOT — Fold: ${FOLD}  Partition: ${PARTITION}  Model tag: ${MODEL_TAG}  Output: ${OUTPUT_DIR}  max_samples=${MAX_SAMPLES}  Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/gradcam_partition.py \
    --fold "${FOLD}" \
    --partition "${PARTITION}" \
    --model_tag "${MODEL_TAG}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_samples "${MAX_SAMPLES}"

echo "End: $(date)"
echo "This wrote gradcam_partial_fold${FOLD}.npz on ${MAX_SAMPLES} samples (a valid but"
echo "subsampled partial, n_samples_full records the true fold size). Read the"
echo "per-100-samples timing lines above to extrapolate full-fold / 5-fold cost."
