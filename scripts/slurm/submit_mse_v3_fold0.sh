#!/bin/bash
# fold0 GPU test run of full_mse_v3 (plain MSE baseline, no GNLL/quantile/
# focal) -- launched alongside full_gnll_focal_v2 and full_gnll_quantile_v2's
# fold0 for a 3-way comparison, user request Aug 20 2026. "v3" because
# "full_mse_v2" already exists (an old JUWELS-path config, never run on
# Raven, unrelated to this session's stratified_kfold/cosine/reflect v2
# treatment -- kept as-is to avoid confusion, not overwritten). Same v2
# treatment as the other two: stratified_kfold, hobday_smooth_target,
# reflect padding, cosine LR w/ warmup. See docs/known_issues.md #40-44
# and docs/narrative.md Aug 20 2026 entries.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_mse_v3_fold0.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=mse_v3_fold0
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

CFG="${REPO_DIR}/configs/partition/full_mse_v3/fold0.yaml"

echo "full_mse_v3 Fold: 0  Config: ${CFG}"
echo "Start: $(date)"

cd "${REPO_DIR}"
python -u scripts/train_partition.py --config "${CFG}" --mode full

echo "End: $(date)"
