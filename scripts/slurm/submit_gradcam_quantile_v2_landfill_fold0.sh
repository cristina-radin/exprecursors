#!/bin/bash
# GradCAM for full_gnll_quantile_v2_landfill (nearest, the current
# committed model), fold0, Aug 22 2026 -- second XAI method, same
# fold/checkpoint as the IG comparison.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gradcam_quantile_v2_landfill_fold0.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=01:30:00
#SBATCH --job-name=gradcam_quantile_v2_landfill
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
python -u scripts/gradcam_quantile_partition.py \
    --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml \
    --output experiments/figures/xai_gradcam/gradcam_quantile_v2_landfill_fold0 \
    --max_samples 300
echo "End: $(date)"
