#!/bin/bash
#SBATCH --account=hai_1127
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=onset_skill
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/experiments/partition/onset_skill/onset_skill_%x-%j.out
#SBATCH --error=/p/project1/hai_1127/radin1/exprecursors/experiments/partition/onset_skill/onset_skill_%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

# Usage:
#   sbatch --export=MODE=remote_only submit_onset_skill.sh
#   sbatch --export=MODE=local_only  submit_onset_skill.sh
#   sbatch --export=MODE=full        submit_onset_skill.sh

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
mkdir -p /p/project1/hai_1127/radin1/exprecursors/experiments/partition/onset_skill

cd /p/project1/hai_1127/radin1/exprecursors

echo "Start: $(date)  MODE=${MODE}"
python -u partition/eval_onset_skill.py --mode ${MODE}
echo "End:   $(date)"
