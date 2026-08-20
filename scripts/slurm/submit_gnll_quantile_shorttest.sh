#!/bin/bash
# Diagnostic-only: 3 epochs of fold0, full_gnll_quantile, to check
# train_nll_loss vs train_pinball_loss scales before committing to the real
# 1000-epoch/5-fold run (docs/narrative.md line 284 asks for exactly this
# check). Not part of the main array job — throwaway output_dir
# (*_shorttest), safe to delete after inspection. Online wandb confirmed
# working from Raven compute nodes (job 29403121, Aug 19 2026).
#
# Raven-specific (see submit_gnll_quantile_partition.sh for the full port
# notes from JUWELS).
#
# Usage:
#   export SBATCH_ACCOUNT=mmm_gpu   # --mail-user must be passed on the sbatch CLI, NOT via env var (SLURM does not expand ${VAR} in #SBATCH lines, and SBATCH_MAIL_USER is not a real sbatch env override -- only SBATCH_ACCOUNT is)
#   sbatch --mail-user=you@example.com --export=ALL scripts/slurm/submit_gnll_quantile_shorttest.sh

#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=00:30:00
#SBATCH --job-name=gnllq_shorttest
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
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

CFG="${REPO_DIR}/configs/partition/full_gnll_quantile/fold0_shorttest.yaml"

echo "Shorttest (3 epochs) fold0  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
