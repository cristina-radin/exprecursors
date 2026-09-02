#!/bin/bash
# Aug 22 2026, user's idea: persistence recall/precision/FPR baseline for
# def1/def2, one per distinct lead time (3,5,7,14,30) -- makes the
# model's def1/def2 recall numbers interpretable (good vs bad vs a
# trivial baseline). CPU only, no GPU, no checkpoint loading.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_persistence_recall_baseline.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=02:00:00
#SBATCH --job-name=persist_recall_baseline
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
python -u scripts/analysis/persistence_recall_baseline.py
echo "End: $(date)"
