#!/bin/bash
#SBATCH --job-name=train_partition
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
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
# fold -- the second argument is then unused and may be omitted). %j alone
# does not distinguish array tasks, so override --output on the command
# line for this case:
#
#   mkdir -p logs
#   sbatch --account=bg1446_gpu --mail-user=you@example.com --mail-type=END,FAIL \
#     --array=0-4 --output=logs/%x_%A_%a.out slurm/levante.sbatch configs/gnll_quantile.yaml
#
# Short real-GPU smoke test before a full run: set LIMIT_BATCHES (forwarded
# as --limit_batches; train_partition.py then suffixes run_name with
# "_limit<N>" so it can never collide with a real run's output_dir):
#
#   mkdir -p logs
#   LIMIT_BATCHES=5 sbatch --account=bg1446_gpu \
#     slurm/levante.sbatch configs/gnll_quantile.yaml 0

set -euo pipefail

CONFIG_YAML="${1:?Usage: sbatch slurm/levante.sbatch <config.yaml> [fold]}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD="$SLURM_ARRAY_TASK_ID"
else
    FOLD="${2:?fold is required: pass it as the 2nd argument, or submit with --array so SLURM_ARRAY_TASK_ID supplies it}"
fi
LIMIT_BATCHES="${LIMIT_BATCHES:-0}"

# sbatch inherits the submitting shell's environment, including whatever
# modules it already has loaded -- module load git here can conflict with a
# git module the submitter loaded themselves (seen in practice: job
# 27948693, "git/2.43.7-gcc-11.2.0 cannot be loaded due to a conflict").
# Only load it if no git is already on PATH.
if ! command -v git >/dev/null 2>&1; then
    set +u
    module load git/2.43.7-gcc-11.2.0
    set -u
fi
echo "git: $(git --version)"

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
for var in MHW_DATA_FILE MHW_CLIM_FILE MHW_EXPERIMENTS_DIR WANDB_ENTITY WANDB_PROJECT; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set (check .env)." >&2
        exit 1
    fi
done
# Levante compute nodes have no internet access (Part 3/i): sync later with
# `wandb sync` from a login node. Overrides whatever .env says.
export WANDB_MODE=offline

# Always activate this venv, regardless of whatever venv/conda env (or
# none) the submitting shell already had active: `source activate` prepends
# this venv's bin to PATH unconditionally, so it wins over anything
# inherited. PYTHONPATH is not touched by activate, so an inherited one
# could still shadow this venv's packages on import -- cleared explicitly.
unset PYTHONPATH
set +u
source /work/bg1446/u241379/exprecursors/venv/bin/activate
set -u

echo "commit: $(git rev-parse HEAD)"
echo "config: $CONFIG_YAML"
echo "fold: $FOLD"
echo "limit_batches: $LIMIT_BATCHES"
echo "MHW_DATA_FILE: $MHW_DATA_FILE ($(du -h "$MHW_DATA_FILE" | cut -f1))"
echo "node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader)"
echo "python: $(which python)"
python -c "import torch; print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')"
# Not touched (unlike PYTHONPATH): torch bundles its own CUDA libs, but an
# inherited LD_LIBRARY_PATH from a module loaded in the submitter's shell
# could still shadow them -- printed so that's visible if it ever happens.
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"

python scripts/train_partition.py --config "$CONFIG_YAML" --fold "$FOLD" \
    --limit_batches "$LIMIT_BATCHES"
