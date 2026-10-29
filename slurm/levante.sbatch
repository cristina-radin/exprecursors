#!/bin/bash
#SBATCH --job-name=train_partition
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#
# Levante sbatch (no internet connection)
#
# Usage:
#   One fold:
#     sbatch --account=bg1446_gpu --mail-user=___@__.com --mail-type=END,FAIL \
#       slurm/levante.sbatch configs/gnll_quantile.yaml 0
#
#   All 5 folds (array):
#     sbatch --account=bg1446_gpu --mail-user=___@__.com --mail-type=END,FAIL \
#       --array=0-4 --output=logs/%x_%A_%a.out slurm/levante.sbatch configs/gnll_quantile.yaml
#
#   Quick test (5 batches):
#     LIMIT_BATCHES=5 sbatch --account=bg1446_gpu \
#       slurm/levante.sbatch configs/gnll_quantile.yaml 0


set -euo pipefail   # stop at the first error or undefined variable

CONFIG_YAML="$1"                     # 1st argument: the config yaml

FOLD="${SLURM_ARRAY_TASK_ID:-$2}"    # fold: from the array (--array=0-4), or the 2nd argument for a single fold
LIMIT_BATCHES="${LIMIT_BATCHES:-0}"  # 0 = full run; otherwise limit the number of batches (for debugging)


# The job inherits the loaded modules of the submitting shell!
if ! command -v git >/dev/null 2>&1; then
    set +u
    module load git/2.43.7-gcc-11.2.0
    set -u
fi
echo "git: $(git --version)"

cd "$SLURM_SUBMIT_DIR"
# Must be launched from the repo root
if [ "$PWD" != "$(git rev-parse --show-toplevel)" ]; then
    echo "ERROR: submit sbatch from the repo root" >&2
    exit 1
fi

# To avoid losing track: Refuse to launch with uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: uncommitted changes, commit first:" >&2
    git status --porcelain >&2
    exit 1
fi

# env
if [ ! -f .env ]; then
    echo "ERROR: .env not found in $PWD (copy .env.example and fill it in)" >&2
    exit 1
fi
set -a        # export every variable in .env to python
source .env
set +a


 # compute nodes have no internet: log locally, sync later with wandb sync
export WANDB_MODE=offline  

# activate project  venv
unset PYTHONPATH 
set +u
source /work/bg1446/u241379/venv/bin/activate
set -u

# INFO for the log 

echo "commit: $(git rev-parse HEAD)"
echo "config: $CONFIG_YAML"
echo "fold: $FOLD"
echo "limit_batches: $LIMIT_BATCHES"
echo "MHW_DATA_FILE: $MHW_DATA_FILE ($(du -h "$MHW_DATA_FILE" | cut -f1))"
echo "node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)"
echo "python: $(which python)"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"


# Check the GPU is visible 
python -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"




python scripts/train_partition.py --config "$CONFIG_YAML" --fold "$FOLD" \
    --limit_batches "$LIMIT_BATCHES"
