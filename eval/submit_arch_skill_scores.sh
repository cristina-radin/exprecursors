#!/bin/bash
#SBATCH -J arch_skill
#SBATCH -A hai_1127
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00
#SBATCH --output=/p/project1/hai_1127/radin1/exprecursors/eval/arch_skill_%j.log
#SBATCH --mail-type=END,FAIL --mail-user=cristina.radin@uni-hamburg.de

export WANDB_MODE=offline
source /p/project1/hai_1127/radin1/exprecursors/venv/bin/activate
cd /p/project1/hai_1127/radin1/exprecursors

BASE=split_blockyear

declare -A CKPTS=(
  [deepSST_layers2]="checkpoints/cnn-lstm-epoch=07-val_loss=0.5455.ckpt|config_blockyear_deepSST.yaml"
  [deepSST_layers2_cnn512]="checkpoints/cnn-lstm-epoch=48-val_loss=0.4780.ckpt|config_blockyear_deepSST_wide.yaml"
  [deepSST_layers2_drop05]="checkpoints/cnn-lstm-epoch=09-val_loss=0.5161.ckpt|config_blockyear_deepSST_reg.yaml"
  [deepSST_layers2_neurons128]="checkpoints/cnn-lstm-epoch=06-val_loss=0.5111.ckpt|config_blockyear_deepSST_small.yaml"
  [deepSST_layers3]="checkpoints/cnn-lstm-epoch=05-val_loss=0.5232.ckpt|config_blockyear_deepSST_deep.yaml"
  [deepSST_layers3_drop05]="checkpoints/cnn-lstm-epoch=22-val_loss=0.5069.ckpt|config_blockyear_deepSST_deep_reg.yaml"
  [deepSST_layers3_win90]="checkpoints/cnn-lstm-epoch=34-val_loss=0.5364.ckpt|config_blockyear_deepSST_win90.yaml"
  [deepSST_layers4_drop01]="checkpoints/cnn-lstm-epoch=42-val_loss=0.4553.ckpt|config_blockyear_deepSST_deeper_drop01.yaml"
  [deepSST_layers4_drop02]="checkpoints/cnn-lstm-epoch=29-val_loss=0.4918.ckpt|config_blockyear_deepSST_deeper_drop02.yaml"
  [deepSST_layers4_drop04]="checkpoints/cnn-lstm-epoch=06-val_loss=0.4916.ckpt|config_blockyear_deepSST_deeper_drop04.yaml"
  [deepSST_layers4_neurons256]="checkpoints/cnn-lstm-epoch=07-val_loss=0.4645.ckpt|config_blockyear_deepSST_deeper_lstm256.yaml"
  [deepSST_layers4_neurons768]="checkpoints/cnn-lstm-epoch=33-val_loss=0.5253.ckpt|config_blockyear_deepSST_deeper_lstm768.yaml"
  [deepSST_layers5]="checkpoints/cnn-lstm-epoch=03-val_loss=0.5250.ckpt|config_blockyear_deepSST_deepest.yaml"
)

for model in "${!CKPTS[@]}"; do
  ckpt=$(echo ${CKPTS[$model]} | cut -d'|' -f1)
  cfg=$(echo  ${CKPTS[$model]} | cut -d'|' -f2)
  echo "=== $model ==="
  python eval/skill_scores.py \
    --checkpoint $BASE/$model/$ckpt \
    --config     $BASE/$model/$cfg \
    --output_dir $BASE/$model/eval_results \
    --threshold  0.0
done
