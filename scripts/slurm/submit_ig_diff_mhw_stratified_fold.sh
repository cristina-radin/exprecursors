#!/bin/bash
# Differential IG (q_pred - y_hat_mean), split by real MHW-day ground
# truth (def2) vs not, `nearest` model, fold given as $1 -- Aug 24 2026,
# user's request: "que pesa mas en los dias que hay MHW" (the pooled
# diff average answers "what matters on an average day", not this).
#
# Same GPU cost as the pooled diff run (job 29565673/74/75) -- same
# samples/n_steps, just accumulated into 2 buckets by outcome instead of
# 1, so no extra IG computation. Rigor before this launch: full test
# suite re-run clean (25 passed) after adding --stratify_mhw; real-data
# dry run on `interactive` (job 29566570, max_samples=20/n_steps=4,
# completed cleanly in 1:33, produced a sane 9/11 mhw/nonmhw split).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_ig_diff_mhw_stratified_fold.sh <fold>
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=01:00:00
#SBATCH --job-name=ig_diff_mhw_strat
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Pass --account on the CLI." >&2
    exit 1
fi
FOLD="${1:?usage: sbatch submit_ig_diff_mhw_stratified_fold.sh <fold 0|1|2>}"

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start fold${FOLD}: $(date)"
python -u scripts/ig_partition_quantile.py \
    --config configs/partition/full_gnll_quantile_v2_landfill/fold${FOLD}.yaml \
    --output experiments/figures/xai_integrated_gradients/ig_diff_mhw_stratified_fold${FOLD} \
    --max_samples 300 \
    --n_steps 50 \
    --heads diff \
    --stratify_mhw
echo "End fold${FOLD}: $(date)"
