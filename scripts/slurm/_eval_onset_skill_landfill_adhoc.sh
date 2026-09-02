#!/bin/bash
# Aug 22 2026: onset/mid-event/no-MHW transition-day analysis for the
# CURRENT committed model (full_gnll_quantile_v2_landfill, nearest) --
# user's idea #2: persistence necessarily fails at transitions, this
# checks whether the model also fails there (the key negative result
# for the paper if so). Reuses eval_onset_skill_quantile_v2.py
# (generalized same day, already has the #53 per-year fix).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=onset_skill_landfill
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
python -u scripts/eval_onset_skill_quantile_v2.py --config_dir full_gnll_quantile_v2_landfill --label landfill
echo "End: $(date)"
