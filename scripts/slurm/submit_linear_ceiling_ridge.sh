#!/bin/bash
# Aug 23 2026, user's idea 4: linear ceiling (ridge on NS-box-mean x
# 60d x 5 vars) for the main model family, to locate the CNN's r
# relative to linearly-extractable precursor information. CPU only.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_linear_ceiling_ridge.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --job-name=linear_ceiling_ridge
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
python -u scripts/analysis/linear_ceiling_ridge.py --config_dir full_gnll_quantile_v2_landfill --label full_lead7
echo "End: $(date)"
