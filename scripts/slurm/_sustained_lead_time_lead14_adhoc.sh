#!/bin/bash
# Aug 24 2026: sustained_lead_time.py for the lead14 family, same
# config_dir/mode/label mapping as submit_event_detection_all_families.sh
# (index 5: config_dir=lead14_landfill, mode=full, label=lead14).
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=sustained_lead_time_lead14
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start: $(date)"
python -u scripts/analysis/sustained_lead_time.py --config_dir lead14_landfill --mode full --label lead14
echo "End: $(date)"
