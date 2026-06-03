#!/bin/bash
#SBATCH -J mhw-xai-TbotAtm
#SBATCH -A hai_1127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/split_blockyear/TbotAtm/xai_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cristina.radin@uni-hamburg.de

source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

BEST_CKPT="split_blockyear/TbotAtm/checkpoints/cnn-lstm-epoch=26-val_loss=0.5301.ckpt"
echo "Using checkpoint: ${BEST_CKPT}"

python xai/run_xai.py \
    --checkpoint "${BEST_CKPT}" \
    --config     split_blockyear/TbotAtm/config.yaml \
    --output_dir split_blockyear/TbotAtm/xai_results \
    --n_events   10 \
    --n_ig       50 \
    --periods    "1985-2004,2005-2014,2015-2024"
