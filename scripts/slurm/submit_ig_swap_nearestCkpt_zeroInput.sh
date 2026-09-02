#!/bin/bash
# Weight-swap ablation, direction B (Aug 21 2026, user request): the
# LAND_FILL (land_fill_mode=nearest-trained) checkpoint, fed inputs built
# with land_fill_mode=zero at inference time -- no retraining, weights
# frozen. Complements direction A (submit_ig_swap_zeroCkpt_nearestInput.
# sh) -- together they isolate the causal effect of land-pixel content
# on ocean-region IG for both trained models.
#
# Config: configs/partition/_adhoc_swap/nearest_ckpt_zero_input_fold0.yaml
# -- output_dir unchanged (points at the nearest-trained checkpoint, so
# best_ckpt() loads nearest-trained weights), land_fill_mode overridden
# to zero (so LazyDataModule builds zero-filled inputs).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_ig_swap_nearestCkpt_zeroInput.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=01:30:00
#SBATCH --job-name=ig_swap_nearestCkpt_zeroIn
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Pass --account on the CLI." >&2
    exit 1
fi

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start: $(date)"
python -u scripts/ig_partition_quantile.py \
    --config configs/partition/_adhoc_swap/nearest_ckpt_zero_input_fold0.yaml \
    --output experiments/figures/xai_integrated_gradients/ig_swap_nearestCkpt_zeroInput_fold0 \
    --max_samples 300 \
    --n_steps 50
echo "End: $(date)"
