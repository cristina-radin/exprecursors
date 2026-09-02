#!/bin/bash
# Aug 24 2026: sustained_lead_time.py for the 3 remaining lead families
# (3, 5, 30 -- 7 and 14 already done) needed for a complete 5-lead
# sustained_full POD comparison figure. Same config_dir/mode/label
# mapping as submit_event_detection_all_families.sh. Array so the 3 run
# in parallel instead of serially.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_sustained_lead_time_remaining.sh
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:00:00
#SBATCH --job-name=sustained_lead_remaining
#SBATCH --array=0-2
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true
cd "${REPO_DIR}"

CONFIG_DIRS=(lead3_landfill lead5_landfill lead30_landfill)
LABELS=(lead3 lead5 lead30)

IDX=${SLURM_ARRAY_TASK_ID}
CFG_DIR=${CONFIG_DIRS[$IDX]}
LABEL=${LABELS[$IDX]}

echo "Start: $(date)  label=${LABEL}  config_dir=${CFG_DIR}"
python -u scripts/analysis/sustained_lead_time.py --config_dir "${CFG_DIR}" --mode full --label "${LABEL}"
echo "End: $(date)"
