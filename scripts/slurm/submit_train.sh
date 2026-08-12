#!/bin/bash
#SBATCH --account=hai_1127
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=partition
#SBATCH --array=0-4
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/partition/partition-${MODE}-f%a-%j.out
#SBATCH --error=/p/project1/hai_1127/radin1/exprecursors/partition/partition-${MODE}-f%a-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

# Submit with:
#   sbatch --export=MODE=remote_only partition/submit_partition.sh
#   sbatch --export=MODE=local_only  partition/submit_partition.sh

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate

CFG="/p/project1/hai_1127/radin1/exprecursors/partition/configs/${MODE//_only/}/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "Mode: ${MODE}  Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd /p/project1/hai_1127/radin1/exprecursors

python -u partition/train_partition.py --config "${CFG}" --mode "${MODE}"

echo "End: $(date)"
