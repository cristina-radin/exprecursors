#!/bin/bash
# Fair re-evaluation of committed full_gnll_quantile_v2 fold0's actual
# best checkpoint (Aug 21 2026, needed to answer whether the land_fill
# retrain really improved the loss -- job 29417405's test metrics predate
# the ckpt_path="best" fix, known_issues.md #46).
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/_eval_committed_fold0_best_ckpt.sh
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_committed_fold0_bestckpt
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true

cd "${REPO_DIR}"
echo "Start: $(date)"
python -u scripts/eval_test_metrics_from_best_ckpt.py \
  --config configs/partition/full_gnll_quantile_v2/fold0.yaml
echo "End: $(date)"
