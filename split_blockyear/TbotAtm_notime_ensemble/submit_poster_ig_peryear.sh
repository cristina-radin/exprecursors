#!/bin/bash
#SBATCH -J mhw-ig-peryear-notime
#SBATCH -A hai_1127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/split_blockyear/TbotAtm_notime_ensemble/ig_peryear_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

python eval/poster_ig_peryear.py \
    --exp_dirs   split_blockyear/TbotAtm_notime_seed100 \
                 split_blockyear/TbotAtm_notime_seed200 \
                 split_blockyear/TbotAtm_notime_seed300 \
                 split_blockyear/TbotAtm_notime_seed400 \
    --npz        split_blockyear/TbotAtm_notime_ensemble/eval_results/predictions.npz \
    --output     poster_figures/fig_ig_peryear_notime.png \
    --n_per_year 5
