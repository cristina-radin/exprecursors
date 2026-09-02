#!/bin/bash
# Time series plot: true target vs mean head vs quantile head vs p90
# threshold, year 2014 (most active MHW year, in fold0's test set).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:20:00
#SBATCH --job-name=plot_pred_vs_target
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
python -u scripts/analysis/plot_prediction_vs_target.py --year 2014
