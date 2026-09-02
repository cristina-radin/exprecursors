#!/bin/bash
# Weight-swap ablation, direction A (Aug 21 2026, user request): the
# COMMITTED (land_fill_mode=zero-trained) checkpoint, fed inputs built
# with land_fill_mode=nearest at inference time -- no retraining, weights
# frozen. Isolates the causal effect of land-pixel content on ocean-
# region IG, without the training-time confound of the model having
# learned different weights under each convention.
#
# Config: configs/partition/_adhoc_swap/zero_ckpt_nearest_input_fold0.yaml
# -- output_dir unchanged (points at the zero-trained checkpoint, so
# best_ckpt() loads zero-trained weights), land_fill_mode overridden to
# nearest (so LazyDataModule builds nearest-filled inputs).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_ig_swap_zeroCkpt_nearestInput.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=01:30:00
#SBATCH --job-name=ig_swap_zeroCkpt_nearestIn
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
    --config configs/partition/_adhoc_swap/zero_ckpt_nearest_input_fold0.yaml \
    --output experiments/figures/xai_integrated_gradients/ig_swap_zeroCkpt_nearestInput_fold0 \
    --max_samples 300 \
    --n_steps 50
echo "End: $(date)"
