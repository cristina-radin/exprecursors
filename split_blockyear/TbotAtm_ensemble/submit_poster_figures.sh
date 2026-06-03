#!/bin/bash
#SBATCH -J mhw-poster-figs
#SBATCH -A hai_1127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/split_blockyear/TbotAtm_ensemble/poster_figures_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

python eval/make_poster_figures.py \
    --output_dir poster_figures \
    --seeds 100 200 300 400 500
