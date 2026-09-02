#!/bin/bash
# Aug 23 2026, user's reframe: does q_pred add real information over
# pure persistence (state-based baseline)? y_true ~ persist + q_pred
# regression + partial correlation + trend skill, all 7 families.
# CPU only, no GPU, no model inference (only data-only persist_pred
# computation + reuse of already-saved eval_recall_v2_partition npz).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_incremental_value_regression.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --job-name=incremental_value_regression
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
python -u scripts/analysis/incremental_value_regression.py
echo "End: $(date)"
