#!/bin/bash
# Ad-hoc: pooled extreme-day recall analysis for full_gnll_quantile vs
# full_gnll_focal, comparing both against the narrative.md-documented plain
# GNLL baseline. Throwaway script, not part of the permanent pipeline.
#
# --mail-user must be passed on the sbatch CLI (not via env var/#SBATCH —
# SLURM does not expand ${VAR} in #SBATCH lines):
#   sbatch --account=mmm_gpu --mail-user=you@example.com scripts/slurm/eval_extreme_recall_adhoc.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=31000
#SBATCH --time=00:20:00
#SBATCH --job-name=eval_extreme_recall
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
python -u scripts/analysis/_adhoc_eval_extreme_recall.py
