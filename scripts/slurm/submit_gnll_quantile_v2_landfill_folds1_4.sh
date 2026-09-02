#!/bin/bash
# Folds 1-4 of full_gnll_quantile_v2_landfill (land_fill_mode=nearest),
# Aug 21 2026 -- fold0 already trained/evaluated separately (job
# 29438575). Decision to relaunch: the coastal-decay artifact reduction
# from land_fill_mode=nearest is confirmed by TWO independent methods
# (full retrain: 18.76x->12.99x; frozen-weight input swap, no retrain:
# 18.76x->13.19x) landing on the same ~13x number -- meets the user's
# pre-stated threshold ("si mejora un poco, lo vamos a hacer"), see
# docs/narrative.md's weight-swap-ablation entry. Performance cost stands
# (r 0.8657->0.8313 on fold0, worse on every metric vs the committed
# zero-fill model) -- accepted tradeoff, not reversed by this decision.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gnll_quantile_v2_landfill_folds1_4.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=gnllq_landfill
#SBATCH --array=1-4
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

CFG="${REPO_DIR}/configs/partition/full_gnll_quantile_v2_landfill/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "full_gnll_quantile_v2_landfill Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
