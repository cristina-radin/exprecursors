# Open issues that affect the code in this branch

Numbers are the original `docs/history/known_issues.md` numbers. Nothing here is
fixed yet; fixing any of them changes results and needs its own step.

## #65 — no purge at Dec 31 / Jan 1 boundaries (`src/data/datamodule.py`)
`stratified_kfold` assigns whole calendar years to train/val/test. A sample
covers 66 days (60 input + lead 7), so windows and targets of adjacent years in
different splits overlap. Proposed fix: purge train samples within 66 samples
(+ optional embargo) of any val/test sample. Estimated cost on the real record:
11-16 % of the train samples per fold (embargo 0). Not applied.

## #57 P2-1 — three definitions of the North Sea box
- `src/utils/hobday.py`: lat 50-63, lon -5-13 on the 0.1° climatology (p90, mean_clim).
- `src/data/masking.py`: index slices lat[100:127], lon[150:187] on the 0.5° grid
  (same box as hobday.py).
- the `target` in `merged_daily.nc`: lat 51.0-62.5, lon -5.2-13.2 at 0.1°.
The hobday.py box does not match the target box (rms 0.047 °C vs 0.026 °C for
the box that does). Effect ~0.02 °C on the p90 threshold and on the #40 delta.

## #57 P2-2 — leak in `compute_stats` (`src/data/dataset.py`)
Input mean/std are computed on the contiguous slice `[min(train), max(train)+60]`.
With `stratified_kfold` that slice contains val and test days: 32-36 % of its
days are not inputs of any train sample (fold 0: 4,520 of 13,929 days, 2,291 of
them in test years). Only the input normalisation constants are affected; the
target statistics use the exact train indices. Reported effect: std -0.91 %,
mean +0.0014.

## #54 — events that span Dec 31 / Jan 1
Per-year Hobday processing splits real events at the year boundary (3 of 52
events, 2006-07, 2015-16, 2022-23). The year ranking in `datamodule.py` runs
`apply_hobday` on the full contiguous series, so it is not affected; the issue
applies to any evaluation that reprocesses the series year by year.

## #40 — `mean_clim` is not 31-day smoothed
Hobday et al. 2016 smooths both the climatological mean and the p90; the mean
used to build `target` never was (RMS 0.046 °C, max 0.131 °C). The committed
model corrects it at load time with `hobday_smooth_target: true`
(`load_ns_mean_clim_smooth_delta`). Regenerating `mean_clim` at the source is not
done.

## #66 — `window_size: 60` has no justification
It was never ablated and no rationale is documented. Do not state that 60 days
was chosen empirically.

## #2 — `merged_daily_v2.nc` parked
`merged_daily.nc` has a `land_mask` shifted by about one pixel on the coasts
(`land_mask_tbottom` is correct). `dataset.py` uses `land_mask_tbottom` for
`ptho_bot`, which is the only masked variable. `merged_daily_v2.nc` (corrected
`land_mask`) exists but the canonical file remains `merged_daily.nc`.

## #30 / #16 — silent architecture mismatch when loading checkpoints
`CNNLSTM_DEFAULTS` in `src/utils/checkpoints.py` (cnn_features 128, lstm_hidden
256, dropout 0.3, ...) differs from the defaults in `scripts/train_partition.py`
(256, 512, 0.2, ...). A config that omits a key gets a different architecture
depending on which code path reads it, and loading with `strict=False` hides the
resulting shape mismatch. Always rebuild the model from `model_config.json`.

## Training hyperparameters fixed in code, not in the yaml
`scripts/train_partition.py` hardcodes `EarlyStopping(patience=30)` and
`ModelCheckpoint(save_top_k=3)` — neither is a config key, so changing them
means editing the script, not the yaml. Not currently exposed as options.

## `kfold`-mode results are only reproducible from `v1-exploracion`
`src/data/datamodule.py` now supports `split_mode: stratified_kfold` only. The
partition table in `docs/history/narrative.md` (job 14199778, `split_mode:
kfold`) is only reproducible from tag `v1-exploracion`; `rebuild` keeps
`stratified_kfold` only.

## Attention architectures, focal-weighted NLL, and the hybrid model were
## removed from `rebuild` (owner's decision, reversible)
`CNNLSTMModel` is CNN -> LSTM -> last timestep -> FC head only now:
`TemporalAttention`, the `attention_only`/`lstm_attention` branches, and
`forward_with_attention()` are gone (block A), along with `state_feature` /
`use_state_feature` (the hybrid model, block B) and `focal_weight` /
`focal_alpha` / `p90_by_doy` (block C). All three live in tag `v1-exploracion`.
- Focal-weighted NLL: fold-level results from Aug 2026 only, known gaps
  known_issues.md #37-39.
- Hybrid (state_feature): trained on 2/5 folds only, no significant
  event-detection gain over the baseline (narrative.md, Aug 24 2026).
- Attention (`attention_only`/`lstm_attention`): known_issues #29 found both
  were structurally identical to each other before an Aug 18 2026 fix: no
  genuine CNN+attention-without-LSTM checkpoint was ever trained.

## Plan: retrain everything from `rebuild` once the code is final and reviewed
Round A: same science, results must match the documented ones within
fold-to-fold spread. Round B: scientific changes one at a time (#65, #57 P2-1,
#66). Existing Raven checkpoints (Aug 22 2026) are the comparison baseline,
not the final results.

## #65, continued: how the leak actually happens in `datamodule.py`
#65 lives in `datamodule.py`: samples are assigned by target year with no gap
at Dec 31/Jan 1, and test years are non-contiguous, so almost every test year
is adjacent to a train year.

## Bucket assignment is not balanced
Bucket assignment is plain round-robin (`i % n_folds`), so bucket 0 always
gets the most extreme year of each group of five: test MHW days per fold
decrease systematically (264, 226, 212, 196, 163). A serpentine assignment
would balance better.

## #57 P2-2, precise statement
Input normalisation stats use a contiguous time slice from the first to the
last train sample; with non-contiguous train years this includes val and
test years. Target stats are train-only. Fix changes numbers.

## Round B hypothesis: the global average pool erases location
The CNN encoder ends in a global average pool, so the network has almost no
information about WHERE a pattern is. This could explain local-only >
full (masking tells it where to look), the linear ridge on the NS-box mean
beating the CNN, and zero-fill scoring higher than nearest (coastlines at
zero act as landmarks). Test: add lat/lon coordinate channels, or avoid the
global average pool, and compare full vs local.

## Round B: the quantile head is never evaluated
`test_step` only evaluates the mean head; `q_pred` is neither scored nor
saved, and test predictions are saved only as a PNG.

## Round B: more hyperparameters fixed in code
`weight_decay` 1e-4, `log_var` clamp +-10, head hidden size 64,
`EarlyStopping` patience 30, `save_top_k` 3.

## Round B: nearest fill covers all land, not just the coast
Nearest fill covers ALL land pixels, not only the coast: continents become
large patches of repeated coastal values.

## Round B: one mean/std per variable for the whole map
`ptho_bot` variability differs ~15x between coast and open ocean.
