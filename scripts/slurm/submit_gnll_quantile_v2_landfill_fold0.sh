#!/bin/bash
# fold0 GPU test of full_gnll_quantile_v2_landfill (land_fill_mode=nearest,
# only change vs the committed full_gnll_quantile_v2 -- user wants a
# clean IG without the coastal masking artifact, known_issues.md #52).
# If this fold0 looks good (clean training + clean IG), remaining folds
# 1-4 AND the lead-time sweep configs will also use land_fill_mode=
# nearest (user decision Aug 21 2026).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_gnll_quantile_v2_landfill_fold0.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=gnllq_landfill_f0
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
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

CFG="${REPO_DIR}/configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml"

echo "full_gnll_quantile_v2_landfill Fold: 0  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
