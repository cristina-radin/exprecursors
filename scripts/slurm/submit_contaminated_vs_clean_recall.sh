#!/bin/bash
# Aug 24 2026: contaminated-vs-clean-year recall check (meeting-prep, no GPU).
# Same cost profile as eval_recall_v2_partition.py's single-family run (~35min).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_contaminated_vs_clean_recall.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --job-name=contam_vs_clean
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
python -u scripts/analysis/contaminated_vs_clean_years_recall.py
echo "End: $(date)"
