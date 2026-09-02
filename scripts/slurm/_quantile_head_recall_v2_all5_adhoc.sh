#!/bin/bash
# Paso 5 decisive analysis: full_gnll_quantile_v2, all 5 folds, quantile
# head recall/precision/FPR (forward_with_quantile, not the mean).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/_quantile_head_recall_v2_all5_adhoc.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=qhead_v2_all5
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start: $(date)"
python -u scripts/analysis/quantile_head_recall_v2_all5.py
echo "End: $(date)"
