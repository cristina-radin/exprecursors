#!/bin/bash
# Item 4 (composite precursor + Granger causality), Aug 21 2026. CPU only,
# no model, no train/test split -- descriptive analysis, not an ML result
# (user's own framing, confirmed before building).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=composite_precursor
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
python -u scripts/composite_precursor_analysis.py \
  --output experiments/figures/step7_persistence/composite_precursor
