#!/bin/bash
# Dry run: full_gnll_quantile_v2_lr2e5/fold0 (LR diagnostic, user request
# Aug 21 2026 -- wants a cleaner/more defensible val_loss curve for the
# paper even though the LR=5e-5 model already passed the recall/precision
# bar). Only learning_rate changed (5e-5 -> 2e-5), nothing else.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:30:00
#SBATCH --job-name=dryrun_lr2e5
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
python -u scripts/train_partition.py --config configs/partition/full_gnll_quantile_v2_lr2e5/fold0.yaml --mode full --fast_dev_run 1
