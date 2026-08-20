#!/bin/bash
# Full 5-fold array of full_gnll_quantile_v2 -- Paso 5, launched Aug 20-21
# 2026 after: (1) fold0 GPU test passed clean, (2) 3-agent code review of
# the shared v2 code (stratified_kfold, hobday_smooth_target, cosine
# scheduler) found nothing, (3) trainer.test() checkpoint bug fixed
# (known_issues.md #46), (4) decisive check that the quantile head's OWN
# prediction (tau=0.9, via forward_with_quantile()) beats both the mean
# head and full_gnll_focal on def2 (field-standard) recall+precision on
# the v1 checkpoints (44.7% recall / 91.8% precision vs focal's 12.8%
# recall) -- NOT spurious, precision far above base rate under both
# ground-truth definitions. See docs/narrative.md Aug 20-21 2026 entries.
#
# IMPORTANT for downstream analysis: evaluate this model via
# model.forward_with_quantile()'s q_pred, NOT model(xs, xt)'s mean --
# the mean head alone underperforms badly (see known_issues.md).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gnll_quantile_v2_partition.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=gnllq_v2
#SBATCH --array=0-4
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

CFG="${REPO_DIR}/configs/partition/full_gnll_quantile_v2/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "full_gnll_quantile_v2 Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
