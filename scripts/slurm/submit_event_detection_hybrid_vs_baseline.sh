#!/bin/bash
# Aug 24 2026: does the state_feature hybrid actually detect MORE MHW
# events, not just correlate better? test_corr improved a lot (fold0
# +0.0868, fold1 +0.0619, see all_results.csv hybrid_state_feature_retrain_result)
# but that was never checked against event-level POD/FAR/CSI -- and there's
# a real reason to doubt it transfers: r_persist is already ~0.94 at
# lead=7, so a model leaning on y(t) can look like a smoothed persistence
# (better day-to-day r) while doing WORSE at catching the onset jump
# itself, which is what the quantile head (not the mean head) is
# specifically for. Only 2/5 folds trained for the hybrid (folds 0,1) --
# eval_event_detection.py generalized (--folds) to pool just those, and
# the baseline is evaluated on the SAME fold subset (not the existing
# 5-fold full_lead7 summary) so the comparison isn't confounded by
# per-fold differences in event count/difficulty.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_event_detection_hybrid_vs_baseline.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --job-name=event_det_hybrid
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true
cd "${REPO_DIR}"

# index 0 = hybrid (use_state_feature=true), index 1 = baseline, SAME fold
# subset (0,1) for a fair like-for-like comparison against the hybrid's
# limited fold coverage.
CONFIG_DIRS=(full_gnll_quantile_v2_landfill_hybrid full_gnll_quantile_v2_landfill)
LABELS=(hybrid_folds01 baseline_folds01)

IDX=${SLURM_ARRAY_TASK_ID}
CFG_DIR=${CONFIG_DIRS[$IDX]}
LABEL=${LABELS[$IDX]}

echo "Family: ${LABEL}  config_dir=${CFG_DIR}  folds=0,1"
echo "Start: $(date)"
python -u scripts/eval_event_detection.py --config_dir "${CFG_DIR}" --mode full --label "${LABEL}" --folds 0,1
echo "End: $(date)"
