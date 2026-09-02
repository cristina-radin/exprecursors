#!/bin/bash
# Occlusion sanity check, nearest (current committed) model, Aug 22
# 2026 -- rerun with the stratified_test_sample fix (known_issues #57
# P1) for a complete, consistent 4-method (IG/occlusion/GradCAM/
# GradientSHAP) triangulation, all on corrected sampling.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --job-name=occlusion_landfill
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
python -u scripts/occlusion_ptho_bot_sanity_check.py \
  --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml \
  --output experiments/figures/xai_integrated_gradients/occlusion_sanity_landfill_fold0 \
  --max_samples 300
