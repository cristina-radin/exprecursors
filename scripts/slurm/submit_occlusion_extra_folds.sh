#!/bin/bash
# Aug 22 2026: occlusion sanity check on fold1/fold2, both committed and nearest.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=01:00:00
#SBATCH --job-name=occlusion_extra_folds
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true
cd "${REPO_DIR}"

CONFIG_DIRS=(full_gnll_quantile_v2 full_gnll_quantile_v2 full_gnll_quantile_v2_landfill full_gnll_quantile_v2_landfill)
FOLDS=(1 2 1 2)
LABELS=(committed_fold1 committed_fold2 nearest_fold1 nearest_fold2)

IDX=${SLURM_ARRAY_TASK_ID}
CFG_DIR=${CONFIG_DIRS[$IDX]}
FOLD=${FOLDS[$IDX]}
LABEL=${LABELS[$IDX]}

echo "Occlusion: ${LABEL}  config=configs/partition/${CFG_DIR}/fold${FOLD}.yaml"
echo "Start: $(date)"
python -u scripts/occlusion_ptho_bot_sanity_check.py \
  --config configs/partition/${CFG_DIR}/fold${FOLD}.yaml \
  --output experiments/figures/xai_integrated_gradients/occlusion_sanity_${LABEL} \
  --max_samples 300
echo "End: $(date)"
