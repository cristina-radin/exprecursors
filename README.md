# MHW precursors — North Sea marine heatwave prediction

CNN-LSTM that predicts the North Sea basin-mean SST anomaly 7 days ahead. It has
a Gaussian-NLL head (mean, log-variance) and an independent quantile head
(tau = 0.9); the quantile head is the one used to flag marine-heatwave days.

This branch (`rebuild`) holds only what is needed to train the committed model,
`full_gnll_quantile_v2_landfill`, and to check it. The exploratory code lives in
`develop` and in the tag `v1-exploracion`. `docs/history/` is the read-only
record of past decisions and bugs; `docs/open_issues.md` lists what is still
open in this code.

## Layout

```
scripts/train_partition.py   training entry point
src/data/                    dataset, datamodule (stratified k-fold), NS-box masks
src/models/cnn_lstm.py       CNN-LSTM model and Lightning module
src/utils/                   Hobday utilities, checkpoint helpers, env paths
configs/gnll_quantile.yaml, configs/gnll.yaml, configs/mse.yaml
tests/                       pytest suite (synthetic data, no real files needed)
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # edit the paths, then: source .env
```

Environment variables: `MHW_DATA_FILE` (`merged_daily.nc`), `MHW_CLIM_FILE`
(`sst_climatology_doy.nc`), `WANDB_ENTITY`, `WANDB_PROJECT` (`WANDB_MODE=offline`
without internet), optional `MHW_EXPERIMENTS_DIR` (default `./experiments`).
Runs are written to `$MHW_EXPERIMENTS_DIR/partition/<run_name>`.

## Train

```bash
python scripts/train_partition.py \
    --config configs/gnll_quantile.yaml --fold 0
```

`--fold` (0 to `n_folds - 1`) selects the fold of the stratified k-fold
split; it is a CLI argument, not a yaml key (the yaml must not have a
`fold` key). `mode: full|local_only|remote_only` in the yaml selects
whether CNNLSTMModel masks the North Sea box (see src/data/masking.py).

## Test

```bash
pytest tests/
```
