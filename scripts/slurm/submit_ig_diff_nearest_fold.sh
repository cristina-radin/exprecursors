#!/bin/bash
# Differential IG (q_pred - y_hat_mean) on the ADOPTED nearest model,
# stratified sampling (fixed), fold given as $1 (0, 1, or 2) -- Aug 24
# 2026, user's request: "como precursor destaca algo mas" (beyond what
# already shows up for the to_anom point forecast, ptho_bot+NS box).
#
# Redo of job 29435465 (Aug 21), which used the `committed` (zero-fill)
# model AND the old test_indices[:max_samples] sampling (known_issues.md
# #57 -- 299/300 samples from a single year), both since superseded.
# Rigor before this launch: existing synthetic-tensor smoke tests
# (tests/test_ig_diff_head.py, unaffected by the sampling change) still
# pass; real-data dry run on `interactive` partition (job 29565643,
# nearest fold0, max_samples=2/n_steps=4, completed cleanly in 1:34).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_ig_diff_nearest_fold.sh <fold>
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=01:00:00
#SBATCH --job-name=ig_diff_nearest
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Pass --account on the CLI." >&2
    exit 1
fi
FOLD="${1:?usage: sbatch submit_ig_diff_nearest_fold.sh <fold 0|1|2>}"

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start fold${FOLD}: $(date)"
python -u scripts/ig_partition_quantile.py \
    --config configs/partition/full_gnll_quantile_v2_landfill/fold${FOLD}.yaml \
    --output experiments/figures/xai_integrated_gradients/ig_diff_nearest_fold${FOLD} \
    --max_samples 300 \
    --n_steps 50 \
    --heads diff
echo "End fold${FOLD}: $(date)"
