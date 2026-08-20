#!/bin/bash
# Submit signed IG on TbotAtm_{PARTITION}_seed42_fold{0-4} (no MHW split).
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# "${VAR}" syntax there is NOT expanded and silently produces an invalid
# value. Account/mail-user must instead come from the SBATCH_ACCOUNT /
# SBATCH_MAIL_USER env vars (recognised by sbatch itself as option
# defaults), or be passed on the CLI.
#
# Usage:
#   source .env
#   export SBATCH_ACCOUNT=your_account   # --mail-user must be passed on the sbatch CLI (see comment above)
#   sbatch --mail-user=you@example.com --export=ALL scripts/slurm/submit_ig_partition.sh
# (or: sbatch --account=... --mail-user=... --export=ALL ...)
#
# To explain a tagged model variant (e.g. TbotAtm_full_mse_v2_seed42_fold{N})
# instead of the untagged run, also export MODEL_TAG:
#   sbatch --export=ALL,MODEL_TAG=mse_v2 scripts/slurm/submit_ig_partition.sh

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=ig_partition
#SBATCH --array=0-4
#SBATCH --output=slurm-ig_partition-%x-%A_%a.out
#SBATCH --error=slurm-ig_partition-%x-%A_%a.err
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

echo "Fold: ${SLURM_ARRAY_TASK_ID}  Partition: ${PARTITION}  Model tag: ${MODEL_TAG:-(none)}  Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/ig_signed_partition.py \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --partition "${PARTITION}" \
    --model_tag "${MODEL_TAG}" \
    --n_steps 50 \
    --batch_size 8

echo "End: $(date)"
