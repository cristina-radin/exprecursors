#!/bin/bash
# Aug 24 2026: loss-comparison timeseries figure (meeting prep, no GPU).
# Loads 3 fold0 checkpoints (mse_v3, gnll_focal_v2, gnll_quantile_v2) and
# plots truth vs prediction per loss variant. CPU-only, small/fast.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_plot_loss_comparison.sh
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:55:00
#SBATCH --job-name=plot_loss_comparison
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
python -u scripts/analysis/plot_loss_comparison_timeseries.py
echo "End: $(date)"
