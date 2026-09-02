#!/bin/bash
# GradientSHAP (Expected Gradients) for the committed full_gnll_quantile_v2
# (zero-fill), fold0, Aug 22 2026 -- third XAI method (user's original
# priority list: "GradCam, IG y Shapley"). Real training-window
# background distribution, not IG's single zero baseline.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gradientshap_quantile_v2_fold0.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=02:00:00
#SBATCH --job-name=gradientshap_v2
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
python -u scripts/gradientshap_quantile_partition.py \
    --config configs/partition/full_gnll_quantile_v2/fold0.yaml \
    --output experiments/figures/xai_gradientshap/gradientshap_quantile_v2_fold0 \
    --max_samples 300 --n_baseline 16 --n_samples 10
echo "End: $(date)"
