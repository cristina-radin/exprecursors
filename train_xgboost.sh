#!/bin/bash
#SBATCH -A hai_1127
#SBATCH -p develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --output=slurmlog/slurm-%j.out

# dont specify this, otherwise training gets stuck
# #SBATCH --ntasks-per-node=4
# #SBATCH --cpus-per-task=12
# #SBATCH --threads-per-core=1

module --force purge
module load Stages/2025
module load GCCcore/.13.3.0
module load Python/3.12.3

# Load virtual environment
source venv/bin/activate

which python

# Submit SLURM job
python -u reduced_dimensionality.py