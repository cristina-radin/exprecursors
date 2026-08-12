#!/bin/bash
# Submit onset skill evaluation for one partition mode.
#
# Usage:
#   export SLURM_ACCOUNT=your_account
#   export SLURM_MAIL=your@email.com
#   source .env
#   sbatch --export=ALL,MODE=remote_only scripts/slurm/submit_eval.sh
#   sbatch --export=ALL,MODE=local_only  scripts/slurm/submit_eval.sh
#   sbatch --export=ALL,MODE=full        scripts/slurm/submit_eval.sh

#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=onset_skill
#SBATCH --output=slurm-onset_skill-%x-%j.out
#SBATCH --error=slurm-onset_skill-%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${SLURM_MAIL:-}

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"

echo "Start: $(date)  MODE=${MODE}"
python -u scripts/eval_onset_skill.py --mode "${MODE}"
echo "End:   $(date)"
