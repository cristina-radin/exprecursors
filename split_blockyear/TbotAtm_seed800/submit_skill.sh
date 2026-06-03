#!/bin/bash
#SBATCH -J mhw-skill-TbotAtm-s800
#SBATCH -A hai_1127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/split_blockyear/TbotAtm_seed800/skill_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

BEST_CKPT=$(ls split_blockyear/TbotAtm_seed800/checkpoints/*.ckpt 2>/dev/null | \
    sed 's/.*val_loss=//;s/\.ckpt//' | paste - <(ls split_blockyear/TbotAtm_seed800/checkpoints/*.ckpt) | \
    sort -n | head -1 | awk '{print $2}')
echo "Using checkpoint: ${BEST_CKPT}"

python eval/skill_scores.py \
    --checkpoint "${BEST_CKPT}" \
    --config     split_blockyear/TbotAtm_seed800/config.yaml \
    --output_dir split_blockyear/TbotAtm_seed800/eval_results \
    --threshold  0.0
