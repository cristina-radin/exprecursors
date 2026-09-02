#!/bin/bash
# Dry run: full_gnll_quantile_v2_landfill/fold0 (land_fill_mode=nearest,
# only change vs the committed model -- user request Aug 21 2026, for a
# clean IG without the coastal masking artifact, known_issues.md #52).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --job-name=dryrun_landfill
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true
export WANDB_MODE=disabled

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml --mode full --fast_dev_run 1
