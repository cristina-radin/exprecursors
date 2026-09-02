#!/bin/bash
# Integrated Gradients for the committed full_gnll_quantile_v2, fold0
# (Aug 21 2026, user request). Computes IG for BOTH the mean head (point
# forecast) and the quantile head (q_pred, the actual precursor-detection
# signal per docs/narrative.md's DECISIVE finding) -- see
# scripts/ig_partition_quantile.py's docstring for why this replaces
# ig_simple.py (known_issues.md #49: that script was broken independent
# of the quantile-head gap).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_ig_quantile_v2_fold0.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=04:00:00
#SBATCH --job-name=ig_quantile_v2
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
    --config configs/partition/full_gnll_quantile_v2/fold0.yaml \
    --output experiments/figures/xai_integrated_gradients/ig_quantile_v2_fold0 \
    --max_samples 300 \
    --n_steps 50
echo "End: $(date)"
