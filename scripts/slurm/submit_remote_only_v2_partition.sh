#!/bin/bash
# Item 2 (Aug 21 2026): remote-only masking, full 5-fold array, same
# architecture/hyperparameters as the committed full_gnll_quantile_v2
# (stratified_kfold, hobday_smooth_target, gaussian_nll+quantile_head,
# land_fill_mode=zero -- kept after land_fill_mode=nearest was found to
# regress performance, see narrative.md Aug 21 2026). Masking is applied
# via --mode remote_only (zeros everything inside the NS box), NOT via
# the yaml.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_remote_only_v2_partition.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=remote_v2
#SBATCH --array=0-4
#SBATCH --output=slurm-partition-%x-%A_%a.out
#SBATCH --error=slurm-partition-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

if [ -z "${SBATCH_ACCOUNT:-}" ] && [ -z "${SLURM_JOB_ACCOUNT:-}" ]; then
    echo "ERROR: no SLURM account set. Pass --account on the CLI." >&2
    exit 1
fi

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

CFG="${REPO_DIR}/configs/partition/remote/fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "remote_only_v2 Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode remote_only

echo "End: $(date)"
