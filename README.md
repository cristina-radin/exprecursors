# MHW Precursors — North Sea Marine Heatwave Prediction

CNN-LSTM model for predicting North Sea marine heatwaves (MHWs) 7 days ahead,
with explainability analysis (Integrated Gradients, Grad-CAM) and
local/remote predictability partitioning.

## Project structure

```
src/
├── data/
│   ├── dataset.py          # LazyDataset — sliding window loader
│   ├── datamodule.py       # PyTorch Lightning DataModule (k-fold splits)
│   └── masking.py          # NS-box masking: mask_remote(), mask_local()
├── models/
│   └── cnn_lstm.py         # CNN-LSTM architecture + Lightning wrapper
├── xai/
│   ├── integrated_gradients.py
│   ├── grad_cam.py
│   └── utils.py
└── utils/
    ├── checkpoints.py      # best_ckpt() — lowest val_loss selection
    ├── hobday.py           # Hobday 2016 MHW classification
    └── metrics.py          # skill_by_phase() — onset/mid_event/no_mhw

scripts/
├── train.py                # Standard k-fold training
├── train_partition.py      # Partition training (remote_only / local_only)
├── eval_ig.py              # Integrated Gradients composite maps
├── eval_onset_skill.py     # Onset-phase skill by partition mode
├── ensemble_skill.py       # Multi-seed ensemble evaluation
└── slurm/                  # SLURM submission scripts

configs/
├── kfold/TbotAtm.yaml      # TbotAtm (ptho_bot + ERA5) base config
├── kfold/SSTAtm.yaml       # SSTAtm (to_anom + ERA5) base config
├── partition/{remote,local}.yaml
└── partition/{local,remote}/fold0-4.yaml

scripts/analysis/           # Granger causality, thermal inertia, tau checks
tests/                      # pytest: splits, masking, checkpoints
archive/poster_egu2026/     # EGU 2026 poster figure scripts
docs/narrative.md           # Scientific decisions log
results/                    # gitignored; paper numbers in all_results.csv
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

Input: set `MHW_DATA_FILE` to the path of `merged_daily.nc` (see `.env.example`).

Target: North Sea basin-mean `to_anom` (SST − climatological mean, ref 1985–2014),
averaged at 0.1° resolution before regridding to 0.5°.
MHW threshold: Hobday 2016 P90, 31-day circular smooth, applied at runtime.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full pipeline and conventions.
