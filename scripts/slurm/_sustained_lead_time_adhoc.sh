#!/bin/bash
# Aug 24 2026: sustained_lead_time.py, full_lead7 family only, ad-hoc
# (user's methodological question re: eval_event_detection.py's
# score_alarms allowing a single isolated alarm day to count as a hit --
# see scripts/analysis/sustained_lead_time.py docstring). Killed twice on
# the raven01 login node (load average ~49, 185 users) before finishing
# even fold 0 -- runs as a real SLURM job instead, same partition/budget
# as the existing event_detection adhoc job.
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=01:30:00
#SBATCH --job-name=sustained_lead_time
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
python -u scripts/analysis/sustained_lead_time.py
echo "End: $(date)"
