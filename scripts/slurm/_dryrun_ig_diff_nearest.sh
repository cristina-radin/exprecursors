#!/bin/bash
# Dry run: differential IG (--heads diff) combined with the stratified
# sampling fix, on the `nearest` (adopted) model -- this exact combination
# (diff head x stratified_test_sample) has never been run together before:
# the original diff run (job 29435465, Aug 21) used the OLD test_indices[
# :max_samples] sampling (same bug as known_issues.md #57, fixed in this
# script afterward by a separate session) and the `committed` model, not
# `nearest`. Confirm the real LazyDataModule + stratified sampling +
# diff_head_fn path runs end-to-end on the nearest fold0 checkpoint before
# spending GPU on the full 3-fold rerun.
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:20:00
#SBATCH --job-name=dryrun_ig_diff_nearest
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
    --config configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml \
    --output experiments/figures/xai_integrated_gradients/_dryrun_diff_nearest \
    --max_samples 2 \
    --n_steps 4 \
    --heads diff
echo "End: $(date)"
