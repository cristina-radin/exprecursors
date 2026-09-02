#!/bin/bash
# Hybrid model (use_state_feature=true), folds 0-1, Aug 24 2026 --
# lowest-priority item tonight ("no me interesa en exceso"), launched only
# after the spatial-pipeline audit+launch and the event_detection
# lead-time-sweep launch, per the user's explicit ordering.
#
# Gives the model explicit access to y(t) (the target's own value at the
# last input-window day, same quantity lag-persistence uses), concatenated
# directly to the LSTM/attention context -- NOT mean-pooled like the
# existing x_temporal calendar features. Same TbotAtm/gnll_quantile_v2/
# land_fill_mode=nearest config as full_gnll_quantile_v2_landfill/fold{0,1}
# in every other respect, for a clean A/B comparison against that already-
# trained pair.
#
# Justification for spending GPU on this (already-established, see
# docs/narrative.md): the zero-cost post-hoc OLS hybrid (alpha*persist +
# beta*q_pred, no retrain) already computed the realistic ceiling on r
# gain (+0.001-0.013 depending on lead) -- this end-to-end retrain is
# justified only for joint uncertainty (GNLL with y(t) as an explicit
# input channel) or to check whether the post-hoc linear stack leaves
# nonlinear value on the table. Code path (state_feature mechanism) is
# implemented and tested: tests/test_state_feature.py (6/6 pass), full
# suite 25 passed/2 skipped, no regressions.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gnll_quantile_v2_landfill_hybrid_folds0_1.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=gnllq_landfill_hybrid
#SBATCH --array=0-1
#SBATCH --output=slurm-partition-%x-%A_%a.out
#SBATCH --error=slurm-partition-%x-%A_%a.err
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

CFG="${REPO_DIR}/configs/partition/full_gnll_quantile_v2_landfill_hybrid/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "full_gnll_quantile_v2_landfill_hybrid Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
