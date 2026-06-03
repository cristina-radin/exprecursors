#!/bin/bash
#SBATCH -J mhw-ens-xai-TbotAtm
#SBATCH -A hai_1127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/split_blockyear/TbotAtm_ensemble/ensemble_xai_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

# Periods comparison (for poster: 1985-2004 vs 2005-2014 vs 2015-2024)
python xai/run_xai_ensemble.py \
    --exp_dirs   split_blockyear/TbotAtm_seed100 \
                 split_blockyear/TbotAtm_seed200 \
                 split_blockyear/TbotAtm_seed300 \
                 split_blockyear/TbotAtm_seed400 \
    --output_dir split_blockyear/TbotAtm_ensemble/xai_periods \
    --periods    1985-2004,2005-2014,2015-2024 \
    --n_ig       30

# Seasonal comparison
python xai/run_xai_ensemble.py \
    --exp_dirs   split_blockyear/TbotAtm_seed100 \
                 split_blockyear/TbotAtm_seed200 \
                 split_blockyear/TbotAtm_seed300 \
                 split_blockyear/TbotAtm_seed400 \
    --output_dir split_blockyear/TbotAtm_ensemble/xai_seasonal \
    --seasons    DJF,MAM,JJA,SON \
    --season_years 1985-2024 \
    --n_ig       30
