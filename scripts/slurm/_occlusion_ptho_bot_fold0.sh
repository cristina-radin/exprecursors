#!/bin/bash
# Occlusion-based sanity check for known_issues.md #52 (ptho_bot coastal
# IG artifact) -- independent, non-gradient attribution method, on the
# already-committed full_gnll_quantile_v2 fold0 checkpoint. CPU only
# (forward passes only, no backward needed).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --job-name=occlusion_ptho_bot
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
  --config configs/partition/full_gnll_quantile_v2/fold0.yaml \
  --output experiments/figures/xai_integrated_gradients/occlusion_sanity_fold0 \
  --max_samples 300
