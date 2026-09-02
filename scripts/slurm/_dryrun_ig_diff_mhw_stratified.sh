#!/bin/bash
# Dry run: new --stratify_mhw flag on ig_partition_quantile.py (Aug 24
# 2026, user request: "que pesa mas en los dias que hay MHW especificamente,
# no en promedio"). Splits the population average into "target day IS a
# real MHW day (def2)" vs "is NOT", instead of one pooled average --
# reuses the exact def2 ground truth (area_frac >= 0.05) from
# eval_recall_v2_partition.py. Same IG cost as without the flag (same
# samples/n_steps, just split into 2 accumulators by outcome instead of
# 1) -- this dry run just confirms the new sample_condition()/counts
# bookkeeping and the p90/area_frac loading work end-to-end before
# spending real GPU time on max_samples=300.
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=00:20:00
#SBATCH --job-name=dryrun_ig_mhw_strat
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
    --output experiments/figures/xai_integrated_gradients/_dryrun_diff_mhw \
    --max_samples 20 \
    --n_steps 4 \
    --heads diff \
    --stratify_mhw
echo "End: $(date)"
