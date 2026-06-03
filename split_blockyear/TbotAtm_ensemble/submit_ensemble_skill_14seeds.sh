#!/bin/bash
#SBATCH -J mhw-ens-skill-14s
#SBATCH -A hai_1127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/split_blockyear/TbotAtm_ensemble/ensemble_skill_14s_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

python eval/ensemble_skill.py \
    --exp_dirs \
        split_blockyear/TbotAtm_seed100 \
        split_blockyear/TbotAtm_seed200 \
        split_blockyear/TbotAtm_seed300 \
        split_blockyear/TbotAtm_seed400 \
        split_blockyear/TbotAtm_seed500 \
        split_blockyear/TbotAtm_seed600 \
        split_blockyear/TbotAtm_seed700 \
        split_blockyear/TbotAtm_seed800 \
        split_blockyear/TbotAtm_seed900 \
        split_blockyear/TbotAtm_seed1000 \
        split_blockyear/TbotAtm_seed1100 \
        split_blockyear/TbotAtm_seed1200 \
        split_blockyear/TbotAtm_seed1300 \
        split_blockyear/TbotAtm_seed1400 \
    --output_dir split_blockyear/TbotAtm_ensemble/eval_results_14s \
    --threshold  0.0
