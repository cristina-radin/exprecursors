#!/bin/bash
# Re-verify onset skill on the committed full_gnll_quantile_v2 model
# (stratified_kfold) -- Paso 7 item 2, previously flagged not-yet-done.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=onset_skill_v2
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
python -u scripts/eval_onset_skill_quantile_v2.py
echo "End: $(date)"
