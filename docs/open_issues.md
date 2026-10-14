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

## Triage of `docs/history/known_issues.md` #1-#66 against the rebuild code (14 Oct 2026)
Checked against the code at `0c5eeb4`. `known_issues.md` itself is the
historical record and is not edited. States: RESUELTO (fixed; how or where) /
OBSOLETO (the code it refers to no longer exists in `rebuild`) / ABIERTO
(still applies; "Ronda A" marks the ones that must be settled before Round A)
/ RONDA B (deferred to Round B by plan) / DUDA (cannot be triaged against the
code; reason given).

| # | Short title | State | Where | Note |
|---|---|---|---|---|
| 1 | Target is `SST - mean_clim`, not `SST - P90` | RESUELTO | `dataset.py:118` | Target read as-is from `merged_daily.nc`; p90 is used only for MHW labels. |
| 2 | `land_mask` vs `land_mask_tbottom` | RESUELTO | `dataset.py:47-64` | `ptho_bot` (the only masked variable) uses `land_mask_tbottom`. `merged_daily_v2.nc` stays parked (#2 above). |
| 3 | Checkpoint regex with trailing dot | RESUELTO | `checkpoints.py:79-85` | Regex plus `rstrip(".")`; covered by `test_checkpoints.py`. |
| 4 | Do not compare GNLL vs MSE skill | RESUELTO | `CLAUDE.md` | Reporting rule, kept in CLAUDE.md ("Losses"). |
| 5 | State mean-of-folds vs pooled r | RESUELTO | `CLAUDE.md` | Reporting rule, kept in CLAUDE.md ("Reporting skill"). |
| 6 | \|IG\| vs signed IG | DUDA | none | Interpretation rule for XAI; no XAI code in `rebuild`. Decide whether it becomes a CLAUDE.md rule when XAI comes back. |
| 7 | `p90_thresh` stored without 31-day smooth | RESUELTO | `hobday.py:34` | Smooth applied in `load_ns_p90`. The threshold itself is under review (see the 14 Oct section). |
| 8 | No unbatched per-sample IG loops | OBSOLETO | none | No IG/inference scripts in `rebuild`. |
| 9 | MHW needs >=5 days + gap <=2 | ABIERTO (Ronda A) | `hobday.py:64-102` | Implemented, but gaps are filled before the 5-day rule (reverse of Oliver). Drives the split. See the 14 Oct section. |
| 10 | CCM unreliable on daily data | DUDA | none | Paper caveat, no CCM code in `rebuild`. |
| 11 | No silent fallbacks | RESUELTO | `CLAUDE.md` | Rule in CLAUDE.md; no `except` in `src/`, `scripts/`, `tests/`, `tools/`. |
| 12 | Old experiment names | OBSOLETO | none | `rebuild` configs are `gnll`, `gnll_quantile`, `mse`. |
| 13 | Forbidden `merged_daily_deepSST.nc` | RESUELTO | `paths.py`, `CLAUDE.md` | Data file only from `$MHW_DATA_FILE`; rule in CLAUDE.md. |
| 14 | Gulf Stream narrative superseded | DUDA | none | Claim about the paper text, not about code. |
| 15 | `LazyDataset.is_land` attribute name | RESUELTO | `dataset.py:50` | Attribute exists; the scripts that misused it are gone. |
| 16 | Hardcoded architecture at checkpoint load | RESUELTO | `checkpoints.py:12-55`, `cnn_lstm.py:113-125` | No constructor defaults, keys read from the signature, `load_model_config` raises on missing/extra keys, no `strict=False`. The "#30 / #16" section above is stale (`CNNLSTM_DEFAULTS` no longer exists). |
| 17 | W&B identifiers hardcoded | RESUELTO | `train_partition.py:265-271` | From `WANDB_ENTITY`/`WANDB_PROJECT`, raises if unset. |
| 18 | "Vectorised" comment on a loop | OBSOLETO | none | `_compute_trend` no longer exists. |
| 19 | `to_anom > 0` as MHW label | OBSOLETO | none | No evaluation code in `rebuild`. |
| 20 | Dead XAI wrappers | OBSOLETO | none | `src/xai/` not in `rebuild`. |
| 21 | Pre-June-2026 TbotAtm on absolute `ptho_bot` | OBSOLETO | none | Old results; `rebuild` retrains on `merged_daily.nc`. |
| 22 | Silent routing to the `_OLD` file | OBSOLETO | none | `thermal_inertia_test.py` not in `rebuild`. |
| 23 | NumPy RNG API in split code | OBSOLETO | `datamodule.py:111-123` | `stratified_kfold` uses no RNG (deterministic ranking). Rule kept in CLAUDE.md. |
| 24 | Stale spatial checkpoints | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 25 | `gaussian_nll` was a dead parameter | RESUELTO | `cnn_lstm.py:154`, `:287-299`, `:329-334` | Real 2-output head and `GaussianNLLLoss`. |
| 26 | 8 px MaxPool grid artifact in IG | OBSOLETO | none | No IG code; the configs use `pooling: avg` (#27). |
| 27 | `pooling` option, use `avg` | RESUELTO | `cnn_lstm.py:54-60`, yamls | Explicit key, no default, saved in `model_config.json`. |
| 28 | Double climatology subtraction | RESUELTO | `dataset.py:7-8` | No climatology code in `dataset.py`. |
| 29 | `attention_only` == `lstm_attention` | OBSOLETO | none | Attention removed (see "Attention architectures..." above). |
| 30 | `cnn_features` default drift | RESUELTO | `checkpoints.py`, `train_partition.py:214-225` | Same as #16: no defaults anywhere, one construction path. |
| 31 | `loss_fn` dead under `gaussian_nll` | RESUELTO | `cnn_lstm.py:287-299` | Mismatch raises. |
| 32 | Test lists not reset | RESUELTO | `cnn_lstm.py:370-376` | `on_test_epoch_start` resets them. |
| 33 | `num_sanity_val_steps=0` | RESUELTO | `train_partition.py:291` | Set to 2. |
| 34 | Reflect padding | RESUELTO | `cnn_lstm.py:56-65`, yamls | Explicit key, no default. |
| 35 | `best_ckpt()` `inf` fallback | RESUELTO | `checkpoints.py:79-85` | Raises on a non-matching name. |
| 36 | Duplicate of #30 | RESUELTO | see #30 | Same issue, same state. |
| 37 | `quantile_head` vs `focal_weight` | OBSOLETO | none | Focal loss removed. |
| 38 | Focal params not persisted | OBSOLETO | none | Focal loss removed. |
| 39 | `p90_by_doy` not restored | OBSOLETO | none | Focal loss removed. |
| 40 | `mean_clim` not 31-day smoothed | RESUELTO | `dataset.py:122-146`, yamls | `hobday_smooth_target: true` in all 3 yamls. Not regenerated at source. Leap-year doy shift affects this delta (see the 14 Oct section). |
| 41 | Basin-mean-then-threshold vs per-pixel MHW | DUDA | `hobday.py` | Paper-framing decision, not a code bug. The 14 Oct decision (Oliver on the box-mean series) keeps basin-mean-first, so it does not settle it. |
| 42 | `kfold` `val_years` collision | RESUELTO | `datamodule.py:79-83`, `test_splits.py` | Only `stratified_kfold`; val rotates per fold, tested. |
| 43 | SLURM mail never sent | RESUELTO | `levante.sbatch:12-19`, `CLAUDE.md` | `--account`/`--mail-user` on the command line. |
| 44 | Cosine `T_max` from `max_epochs` | RESUELTO | `cnn_lstm.py:426-445`, yamls | `cosine_t_max_epochs` explicit (60). |
| 45 | `configs/partition/` sprawl | RESUELTO | `configs/` | Three yamls. |
| 46 | `trainer.test()` on last-epoch weights | RESUELTO | `train_partition.py:301` | `ckpt_path="best"`. |
| 47 | XAI explains only the mean head | OBSOLETO | none | No XAI code. Applies again when XAI returns. |
| 48 | `persistence_remote_sst.py` import path | OBSOLETO | none | Script not in `rebuild`. |
| 49 | `ig_simple.py` broken | OBSOLETO | none | Script not in `rebuild`. |
| 50 | cuDNN LSTM backward in eval mode | OBSOLETO | none | No IG code. Remember it when XAI returns. |
| 51 | Mean vs quantile IG maps identical | OBSOLETO | none | XAI result, no code in `rebuild`. |
| 52 | `ptho_bot` coastal IG is a masking artifact | OBSOLETO | `dataset.py:66-95` | XAI part gone; `land_fill_mode: nearest` is in the yamls. Related Round B item: "nearest fill covers all land". |
| 53 | Concatenated test years bridged by Hobday | OBSOLETO | none | No evaluation code; the split runs Hobday on the full contiguous series. Applies to any future evaluation. |
| 54 | Per-year Hobday cuts Dec 31/Jan 1 events | ABIERTO | none yet | Applies only to evaluation that reprocesses year by year; none exists in `rebuild` yet. |
| 55 | Land pixels in `ptho_bot` stats (nearest) | RESUELTO | `dataset.py:189-195` | Land excluded for ocean variables under both fill modes. |
| 56 | def2 climatology, onset script, v1 configs | OBSOLETO | none | The three scripts/configs are not in `rebuild`. |
| 57 P1 | IG sampled only 1985 | OBSOLETO | none | No XAI code. |
| 57 P2-1 | Three NS box definitions | RONDA B | `hobday.py:18-19`, `masking.py:17-18` | Listed for Round B in the plan above. |
| 57 P2-2 | `compute_stats` contiguous-slice leak | ABIERTO | `dataset.py:173-187` | Documented in code and above; not in the Round B list of the plan. Changes numbers if fixed. |
| 57 P2-3 | Clim 0.1° vs inputs 0.5° | ABIERTO (Ronda A) | `hobday.py:25-34` | The current threshold is a box mean of 0.1° per-pixel p90s; settled by the threshold decision in the 14 Oct section. The #40 delta still comes from the 0.1° clim. |
| 58 | Relaunch into a dirty `output_dir` | RESUELTO | `train_partition.py:42-53`, `checkpoints.py:66-73` | Raises on existing checkpoints and on `-vN.ckpt` files. |
| 59 | GradCAM on `(batch, 2)` output | OBSOLETO | none | No XAI code. |
| 60 | Eval scripts bypass local/remote masking | RESUELTO | `cnn_lstm.py:174-189` | Masking now lives in `_encode`, shared by every forward path. |
| 61 | Spatial pipeline paths | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 62 | Spatial val-year shuffle | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 63 | Spatial `compute_stats` masking | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 64 | Spatial onset uses `tgt>0` | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 65 | No purge at Dec 31/Jan 1 | RONDA B | `datamodule.py:117-126` | See #65 above. |
| 66 | `window_size: 60` unjustified | RONDA B | yamls | See #66 above. |

Stale text found in this file while triaging (not edited here): the "#30 / #16"
section (`CNNLSTM_DEFAULTS` no longer exists), "Training hyperparameters fixed in
code" and "Round B: more hyperparameters fixed in code" (`early_stopping_patience`,
`save_top_k` and `weight_decay` are yaml keys now; the `log_var` clamp and the
head size 64 are still in code), "the quantile head is never evaluated"
(`q_pred` is now saved in `test_predictions.npz`, though still not scored), and
the test MHW days in "Bucket assignment is not balanced" (264, 226, 212, 196, 163;
the current split gives 261, 224, 207, 184, 171).
