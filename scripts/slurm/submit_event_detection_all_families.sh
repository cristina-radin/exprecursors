#!/bin/bash
# Aug 24 2026: run eval_event_detection.py (POD/FAR/CSI event-level onset
# detection, model vs persistence) across all 7 v2 experiment families --
# user called this "quiza el principal resultado del paper" and asked for
# "mas lead times y mas informacion" beyond the original lead=7-only run.
# Same family/label convention as submit_eval_recall_v2_all_families.sh.
# One array task per family, CPU-only, same cost as the original
# adhoc lead=7-only version (~1h30 walltime budget).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_event_detection_all_families.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=event_detection
#SBATCH --array=0-6
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true
cd "${REPO_DIR}"

# index -> (config_dir, mode, label) -- same masking fix as
# eval_recall_v2_partition.py: local/remote checkpoints need mask_local/
# mask_remote applied explicitly at eval time (not baked into the model).
CONFIG_DIRS=(full_gnll_quantile_v2_landfill local remote lead3_landfill lead5_landfill lead14_landfill lead30_landfill)
MODES=(full local_only remote_only full full full full)
LABELS=(full_lead7 local remote lead3 lead5 lead14 lead30)

IDX=${SLURM_ARRAY_TASK_ID}
CFG_DIR=${CONFIG_DIRS[$IDX]}
MODE=${MODES[$IDX]}
LABEL=${LABELS[$IDX]}

echo "Family: ${LABEL}  config_dir=${CFG_DIR}  mode=${MODE}"
echo "Start: $(date)"
python -u scripts/eval_event_detection.py --config_dir "${CFG_DIR}" --mode "${MODE}" --label "${LABEL}"
echo "End: $(date)"
