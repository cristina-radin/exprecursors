#!/bin/bash
#SBATCH -J variable_scatter
#SBATCH -A hai_1127
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/eval/variable_scatter_%j.log
#SBATCH --mail-type=END,FAIL --mail-user=cristina.radin@uni-hamburg.de

export WANDB_MODE=offline

cd /p/project1/hai_1127/radin1/exprecursors
source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate

python eval/plot_variable_scatter.py \
    --output_dir eval_results/variable_scatter
