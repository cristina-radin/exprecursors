#!/bin/bash
# Submit TbotAtm Full focal-weighted-GNLL partition training, all 5 folds.
#
# Alternative to full_gnll_quantile (auxiliary quantile head): reweights
# the per-sample GaussianNLLLoss toward exceedance days (truth > Hobday
# p90(DOY)) via focal_weight/focal_alpha in the config, instead of a
# separate pinball-loss head. Keeps mean=E[Y|X] and var=conditional
# variance statistically unperturbed — only the sample WEIGHTING changes.
# See src/models/cnn_lstm.py::CNNLightningModule._focal_weighted_loss.
#
# Raven-specific (see submit_gnll_quantile_partition.sh for the full port
# notes from JUWELS — same partition/module/account choices apply here).
#
# Requires MHW_CLIM_FILE (sst_climatology_doy.nc) — train_partition.py loads
# it via src.utils.hobday.load_ns_p90() when focal_weight: true.
#
# Usage:
#   source .env   # must have WANDB_ENTITY and WANDB_PROJECT set
#   export SBATCH_ACCOUNT=mmm_gpu SBATCH_MAIL_USER=you@example.com
#   sbatch --export=ALL scripts/slurm/submit_gnll_focal_partition.sh

#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=partition_gnllf
#SBATCH --array=0-4
#SBATCH --output=slurm-partition-%x-%A_%a.out
#SBATCH --error=slurm-partition-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Export SBATCH_ACCOUNT or pass --account on the CLI." >&2
    exit 1
fi

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/full_gnll_focal/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "Focal-NLL Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
