#!/bin/bash
# Aug 24 2026: first-ever Raven launch of the spatial (2D ConvLSTM) pipeline
# -- previously only run on JUWELS. Migration required a 3-agent audit
# (data/split, model/eval, Raven-migration readiness) before launch, per
# the user's explicit instruction ("revisar a conciencia el espacial ...
# Solo entonces lanzar experimento"). Fixes applied before this launch:
#
#   1. src_spatial/dataset_spatial.py: removed the hardcoded JUWELS
#      DATA_FILE constant; now reads config["data_dir"] directly inside
#      SpatialDataset.__init__ -- matches the scalar pipeline's actual
#      convention (LazyDataModule/LazyDataset also read data_dir straight
#      from the resolved YAML, not an env var at dataset-load time).
#   2. configs/spatial/TbotAtm_fold{0,1}.yaml: data_dir and output_dir
#      repointed to Raven paths (/raven/u/cradin/data/merged_daily.nc,
#      /raven/u/cradin/exprecursors/experiments/spatial/runs/...).
#      TbotAtm_fold1.yaml did not exist before tonight -- created as a
#      copy of fold0 with fold: 1 and a distinct output_dir.
#   3. scripts_spatial/train_spatial.py build_splits(): val-year shuffle
#      seed changed from `seed` to `seed + fold` -- with the same seed,
#      fold0/fold1's val_years sets measured 83-100% pairwise overlap on
#      real data (same bug class as known_issues.md #1/#42's kfold
#      val_years collision), undermining independence between the two
#      folds' early-stopping. Also added year-list printing +
#      resolved_config.yaml dump (previously only counts were printed,
#      violating the standing "always output resolved splits/config"
#      rule).
#   4. src_spatial/dataset_spatial.py compute_stats(): was masking every
#      variable to the SST ocean mask uniformly, but __getitem__ only
#      masks ptho_bot (to land_mask_tbottom) and never spatially masks
#      the ERA5 variables at all (u10/v10/msl/ssr, not in ocean_variables)
#      -- computing their normalization stats over ocean-only pixels was
#      out of scope with what the model actually sees (measured ~16-17%
#      std inflation on real data for the wind/pressure/radiation
#      channels). Fixed to mirror __getitem__'s per-variable masking
#      exactly.
#
# Deliberately NOT done tonight (documented, not silently skipped):
#   - NF-S-5 (scripts_spatial/eval/mhw_onset_skill.py's tgt>0 proxy
#     instead of proper per-pixel Hobday p90+persistence) -- a fix is
#     fully designed (known_issues.md) but not implemented; that script
#     also hardcodes N_FOLDS=5 and will not run meaningfully against only
#     2 trained folds regardless. Do not run it until all 5 folds exist
#     AND the fix lands.
#   - MLD-weights prerequisite / *_phys configs -- irrelevant to the
#     plain TbotAtm model this job trains.
#
#   sbatch --account=mmm_gpu --mail-user=cristina.radin@uni-hamburg.de scripts/slurm/submit_spatial_tbotatm_folds0_1.sh
#SBATCH --partition=gpu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125000
#SBATCH --time=12:00:00
#SBATCH --job-name=spatial_tbotatm
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

module purge
module load python-waterboa/2024.06

REPO_DIR="${SLURM_SUBMIT_DIR}"
source "${REPO_DIR}/venv/bin/activate"
source "${REPO_DIR}/.env" 2>/dev/null || true
cd "${REPO_DIR}"

CFG="configs/spatial/TbotAtm_fold${SLURM_ARRAY_TASK_ID}.yaml"

echo "spatial TbotAtm  Fold: ${SLURM_ARRAY_TASK_ID}  Config: ${CFG}"
echo "Start: $(date)"
python -u scripts_spatial/train_spatial.py --config "${CFG}"
echo "End: $(date)"
