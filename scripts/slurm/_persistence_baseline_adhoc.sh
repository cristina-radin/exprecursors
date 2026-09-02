#!/bin/bash
# Paso 7: lag-7 persistence baseline gate for the committed
# full_gnll_quantile_v2 model (stratified_kfold test years).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/_persistence_baseline_adhoc.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --job-name=persist_baseline
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start: $(date)"
python -u scripts/persistence_remote_sst.py --split_mode stratified_kfold
echo "End: $(date)"
