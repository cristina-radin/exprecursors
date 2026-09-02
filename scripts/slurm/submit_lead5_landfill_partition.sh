#!/bin/bash
# Item 3 (lead-time sweep), lead=5d, land_fill_mode=nearest (Aug 22
# 2026 decision: user chose nearest for everything, after weighing that
# the performance cost vs committed likely reflects the model losing a
# non-physical zero-fill-edge shortcut, not a worse model per se -- see
# docs/narrative.md). Same architecture/hyperparameters as
# full_gnll_quantile_v2_landfill, only lead_time differs.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_lead5_landfill_partition.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=lead5_landfill
#SBATCH --array=0-4
#SBATCH --output=slurm-partition-%x-%A_%a.out
#SBATCH --error=slurm-partition-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Pass --account on the CLI." >&2
    exit 1
fi

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/lead5_landfill/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "lead5_landfill Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
