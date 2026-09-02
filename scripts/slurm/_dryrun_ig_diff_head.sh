#!/bin/bash
# Dry run: new --heads diff option in ig_partition_quantile.py (Aug 21
# 2026, user request) -- IG on q_pred - y_hat_mean directly, using the
# real full_gnll_quantile_v2 fold0 checkpoint + real data, but tiny
# n_steps/max_samples on CPU. Purpose: confirm the real LazyDataModule +
# checkpoint-loading + new diff_head_fn path runs end-to-end without
# error before spending any GPU time on the real n_steps=50/max_samples=300
# run. Synthetic-tensor unit tests (tests/test_ig_diff_head.py) already
# passed -- this is the real-data integration check.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:20:00
#SBATCH --job-name=dryrun_ig_diff
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
python -u scripts/ig_partition_quantile.py \
    --config configs/partition/full_gnll_quantile_v2/fold0.yaml \
    --output experiments/figures/xai_integrated_gradients/_dryrun_diff \
    --max_samples 2 \
    --n_steps 4 \
    --heads diff
echo "End: $(date)"
