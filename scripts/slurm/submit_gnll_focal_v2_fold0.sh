#!/bin/bash
# Paso 4 (plan de ejecución por pasos, Aug 20 2026): single real fold (fold0)
# of full_gnll_focal_v2 as a GPU test run before committing the full 5-fold
# array (Paso 5, gated on the user confirming this run looks right).
#
# v2 = stratified_kfold split (fixes the val_years collision in the old
# kfold mode) + hobday_smooth_target (31-day-smoothed mean_clim) + cosine
# LR schedule with warmup (cosine_t_max_epochs=60) + reflect padding +
# focal-weighted GNLL. See docs/known_issues.md #40-44 and docs/narrative.md
# Aug 20 2026 entries for the full reasoning behind every one of these.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gnll_focal_v2_fold0.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=gnllf_v2_fold0
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
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

CFG="${REPO_DIR}/configs/partition/full_gnll_focal_v2/fold0.yaml"

echo "full_gnll_focal_v2 Fold: 0  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
