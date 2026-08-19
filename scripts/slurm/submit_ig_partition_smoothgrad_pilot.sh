#!/bin/bash
# PILOT run: signed SmoothGrad-IG on TbotAtm_{PARTITION}_seed42_fold{FOLD}.
#
# NOTE: account hai_1127 has NO CPU partition access on JUWELS — only
# booster/develbooster/develbooster-sw/largebooster (all GPU). Verified via
# `sacctmgr show assoc account=hai_1127` (Aug 18 2026): submitting to
# --partition=batch fails with "Invalid account or account/partition
# combination". So this pilot (and the real run) must use GPU; the cost
# lever is sg_samples / max_batches / test-set size, not CPU vs GPU.
#
# Purpose: measure real per-batch / per-noise-sample GPU cost of the new
# SmoothGrad path (n_samples x compute_ig_batch, see compute_smoothgrad_batch
# in ig_signed_partition.py) BEFORE committing to a full 5-fold array job.
# Single fold, --max_batches caps it to a few batches so this fails cheap
# instead of burning GPU-hours. Check the .out log for
# "smoothgrad sample i/20 took Xs" and "batch N/M took Xs" lines, then
# extrapolate: full fold cost ~= n_batches x batch_time; full experiment
# ~= 5 x that (only after this pilot looks sane).
#
# #SBATCH directives are read literally by sbatch, not through the shell —
# "${VAR}" syntax there is NOT expanded. Account/mail-user must come from
# SBATCH_ACCOUNT / SBATCH_MAIL_USER env vars or be passed on the CLI.
#
# Usage:
#   source .env
#   export SBATCH_ACCOUNT=your_account SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL,PARTITION=full,MODEL_TAG=mse_v2,FOLD=0,MAX_BATCHES=1 \
#       scripts/slurm/submit_ig_partition_smoothgrad_pilot.sh
#
# MAX_BATCHES (default 1) stops the run early for timing purposes — with the
# default n_steps=50/batch_size=8/sg_samples=20 this does NOT produce a usable
# ig_partial_fold{N}.npz, only console timing lines. Once the real per-batch
# GPU cost is known, decide with the user how to run the full fold(s)
# (raise MAX_BATCHES / adjust sg_samples / submit the real --array=0-4 job).

#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --job-name=ig_partition_sg_pilot
#SBATCH --output=slurm-ig_partition_sg_pilot-%x-%j.out
#SBATCH --error=slurm-ig_partition_sg_pilot-%x-%j.err
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
MODEL_TAG="${MODEL_TAG:-mse_v2}"
FOLD="${FOLD:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/xai_partition_full_smoothgrad}"
SG_SIGMA="${SG_SIGMA:-0.1}"
SG_SAMPLES="${SG_SAMPLES:-20}"
MAX_BATCHES="${MAX_BATCHES:-1}"

echo "PILOT — Fold: ${FOLD}  Partition: ${PARTITION}  Model tag: ${MODEL_TAG}  Output: ${OUTPUT_DIR}  max_batches=${MAX_BATCHES}  Start: $(date)"

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
    --max_batches "${MAX_BATCHES}"

echo "End: $(date)"
echo "This is a TIMING PILOT: it stops after ${MAX_BATCHES} batches and does NOT write"
echo "ig_partial_fold${FOLD}.npz. Read the per-batch/per-noise-sample lines above to"
echo "extrapolate real cost before deciding how to run the full 5-fold job."
