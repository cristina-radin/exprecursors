#!/bin/bash
# Differential Integrated Gradients for full_gnll_quantile_v2 fold0 (Aug 21
# 2026, user request). Computes IG on q_pred - y_hat_mean directly (the
# 'diff' head added to scripts/ig_partition_quantile.py), isolating the
# gradient direction that differentiates the quantile head from the mean
# head -- their separately-computed, population-averaged IG maps turned
# out to be 0.998-0.999 spatially correlated and don't show this on their
# own (see known_issues.md #51). Same n_steps/max_samples/chunking/cudnn
# workaround as the original mean+quantile run (job 29433738) -- only one
# head instead of two, so roughly half the wall clock (~15 min expected).
# Rigor before this launch: synthetic-tensor smoke tests
# (tests/test_ig_diff_head.py, 3/3 passing incl. completeness axiom),
# full existing test suite re-run clean (19 passed), and a real-data CPU
# dry run (job 29435406, max_samples=2/n_steps=4, completed cleanly in 27s)
# using the actual LazyDataModule + checkpoint-loading path.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_ig_quantile_v2_fold0_diff.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=02:00:00
#SBATCH --job-name=ig_quantile_v2_diff
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
    --n_steps 50 \
    --heads diff
echo "End: $(date)"
