#!/bin/bash
# Real (non-pilot) signed SmoothGrad-IG run on TbotAtm_{PARTITION}_seed42_fold{FOLD}.
#
# Cost-aware defaults, chosen after a timing pilot (job 14203778, Aug 18 2026)
# measured 18.1s per SmoothGrad noise-sample at n_steps=50/batch_size=8 on GPU
# (develbooster) — a full un-subsampled fold (~365 batches, sg_samples=20)
# would cost ~37 GPU-hours, ~185 GPU-hours for all 5 folds. Account hai_1127
# has GPU-only allocation (no CPU partition — see submit_ig_partition_smoothgrad_pilot.sh),
# so MAX_SAMPLES subsampling is the cost lever, not CPU vs GPU.
#
# MAX_SAMPLES (default 300) uses an evenly-spaced subsample of the test set
# (see run_fold's max_samples logic) — DOES write a complete, mergeable
# ig_partial_fold{N}.npz, with n_samples_full recorded alongside n_samples so
# it's never mistaken for the full-test-set result. ~300 samples matches the
# order of magnitude that already gave clean composite maps in
# experiments/xai_composite/ (135/118 samples). At 300 samples/fold this run
# costs ~3.8 GPU-hours; do NOT bump MAX_SAMPLES up without recomputing that
# estimate first (cost scales ~linearly with it).
#
# One fold per submission (no array) — run fold0 first, inspect the maps,
# then decide with the user whether/how many more folds to submit.
#
# #SBATCH directives are read literally by sbatch, not through the shell.
# Account/mail-user must come from SBATCH_ACCOUNT / SBATCH_MAIL_USER env
# vars or be passed on the CLI.
#
# Usage:
#   source .env
#   export SBATCH_ACCOUNT=your_account SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL,PARTITION=full,MODEL_TAG=mse_v2,FOLD=0,MAX_SAMPLES=300 \
#       scripts/slurm/submit_ig_partition_smoothgrad.sh

#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --job-name=ig_partition_sg
#SBATCH --output=slurm-ig_partition_sg-%x-%j.out
#SBATCH --error=slurm-ig_partition_sg-%x-%j.err
#SBATCH --mail-type=FAIL,END

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
MODEL_TAG="${MODEL_TAG:-mse_v2}"
FOLD="${FOLD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/xai_partition_full_smoothgrad}"
SG_SIGMA="${SG_SIGMA:-0.1}"
SG_SAMPLES="${SG_SAMPLES:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"

echo "Fold: ${FOLD}  Partition: ${PARTITION}  Model tag: ${MODEL_TAG}  Output: ${OUTPUT_DIR}  max_samples=${MAX_SAMPLES}  Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/ig_signed_partition.py \
    --fold "${FOLD}" \
    --partition "${PARTITION}" \
    --model_tag "${MODEL_TAG}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_steps 50 \
    --batch_size 8 \
    --smoothgrad \
    --sg_sigma "${SG_SIGMA}" \
    --sg_samples "${SG_SAMPLES}" \
    --max_samples "${MAX_SAMPLES}"

echo "End: $(date)"
