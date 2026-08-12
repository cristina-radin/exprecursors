#!/bin/bash
#SBATCH --account=hai_1127
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=local_only_tr
#SBATCH --array=0-4
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/local_only/local_tr-f%a-%j.out
#SBATCH --error=/p/project1/hai_1127/radin1/exprecursors/local_only/local_tr-f%a-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate

CFG="/p/project1/hai_1127/radin1/exprecursors/local_only/configs/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "Task ${SLURM_ARRAY_TASK_ID} — config: ${CFG}"
echo "Start: $(date)"

cd /p/project1/hai_1127/radin1/exprecursors

python -u local_only/train_local_only.py --config "${CFG}"

echo "End: $(date)"
