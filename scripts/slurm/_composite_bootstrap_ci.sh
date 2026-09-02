#!/bin/bash
# Bootstrap CI for composite_precursor_analysis.py's NS-box curve, Aug 21
# 2026, user request. CPU only, no model, descriptive analysis, not an ML
# result -- same framing as the parent script.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --job-name=composite_bootstrap_ci
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
python -u scripts/composite_bootstrap_ci.py \
  --output experiments/figures/step7_persistence/composite_precursor
