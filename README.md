# MHW Precursors — North Sea Marine Heatwave Prediction

CNN-LSTM model for predicting North Sea marine heatwaves (MHWs) 7 days ahead,
with explainability analysis (Integrated Gradients, Grad-CAM) and
local/remote predictability partitioning.

## Project structure

```
src/              model code — data loading, CNN-LSTM, XAI, utilities
scripts/          training, evaluation, XAI runs; scripts/slurm/ for SLURM
scripts/analysis/ exploratory analysis (Granger, thermal inertia, tau)
configs/          experiment configs (kfold/, partition/)
tests/            pytest suite
docs/             narrative.md (decisions), data.md (pipeline + variables)
archive/          EGU 2026 poster scripts
results/          gitignored; paper numbers in all_results.csv
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# Train TbotAtm model, fold 0
python scripts/train.py --config configs/kfold/TbotAtm.yaml --fold 0

# Submit all 5 folds to SLURM
sbatch scripts/slurm/submit_train.sh

# Evaluate onset skill for remote_only partition
python scripts/eval_onset_skill.py --mode remote_only
```

## Data

Set `MHW_DATA_FILE` to the path of `merged_daily.nc` (see `.env.example`).

See [CONTRIBUTING.md](CONTRIBUTING.md) for full pipeline and conventions.
