#!/bin/bash
#SBATCH --job-name=train_partition
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#
# Levante-specific. See slurm/raven.sbatch for the Raven equivalent (pending).
#
# #SBATCH lines are read literally by sbatch; ${VAR} is never expanded there
# (known_issues.md #43) -- --account and --mail-user go on the sbatch command
# line, not in this file. Create logs/ first (sbatch does not create the
# --output directory). Launch with:
#
#   mkdir -p logs
#   sbatch --account=bg1446_gpu --mail-user=you@example.com --mail-type=END,FAIL \
#     slurm/levante.sbatch configs/gnll_quantile.yaml 0
#
# For a 5-fold array (one task per fold, SLURM_ARRAY_TASK_ID becomes the
# fold -- the second argument is then unused and may be omitted):
#
#   mkdir -p logs
#   sbatch --account=bg1446_gpu --mail-user=you@example.com --mail-type=END,FAIL \
#     --array=0-4 slurm/levante.sbatch configs/gnll_quantile.yaml

set -euo pipefail

CONFIG_YAML="${1:?Usage: sbatch slurm/levante.sbatch <config.yaml> [fold]}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="$SLURM_ARRAY_TASK_ID"
else
    FOLD="${2:?fold is required: pass it as the 2nd argument, or submit with --array so SLURM_ARRAY_TASK_ID supplies it}"
fi

cd "$SLURM_SUBMIT_DIR"
REPO_ROOT=$(git rev-parse --show-toplevel)
if [ "$PWD" != "$REPO_ROOT" ]; then
    echo "ERROR: \$SLURM_SUBMIT_DIR ($PWD) is not the repo root ($REPO_ROOT)." \
         "Submit sbatch from the repo root." >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "ERROR: working tree has uncommitted changes -- refusing to launch an" \
         "unreproducible run. Commit first, or set ALLOW_DIRTY=1 to override." >&2
    git status --porcelain >&2
    exit 1
fi

if [ ! -f .env ]; then
    echo "ERROR: .env not found in $PWD. Copy .env.example to .env and fill it in." >&2
    exit 1
fi
set -a
source .env
set +a
for var in MHW_DATA_FILE MHW_CLIM_FILE WANDB_ENTITY WANDB_PROJECT; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set (check .env)." >&2
        exit 1
    fi
done
# Levante compute nodes have no internet access (Part 3/i): sync later with
# `wandb sync` from a login node. Overrides whatever .env says.
export WANDB_MODE=offline

module load git/2.43.7-gcc-11.2.0
source /work/bg1446/u241379/exprecursors/venv/bin/activate

echo "commit: $(git rev-parse HEAD)"
echo "config: $CONFIG_YAML"
echo "fold: $FOLD"
echo "MHW_DATA_FILE: $MHW_DATA_FILE ($(du -h "$MHW_DATA_FILE" | cut -f1))"
echo "node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)"

python scripts/train_partition.py --config "$CONFIG_YAML" --fold "$FOLD"
