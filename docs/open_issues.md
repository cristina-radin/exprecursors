# Open issues that affect the code in this branch

Numbers are the original `docs/history/known_issues.md` numbers. Fixed items
are marked FIXED / RESUELTO with their commit.

## #65 — no purge at Dec 31 / Jan 1 boundaries (`src/data/datamodule.py`)
`stratified_kfold` assigns whole calendar years to train/val/test. A sample
covers 66 days (60 input + lead 7), so windows and targets of adjacent years in
different splits overlap. Proposed fix: purge train samples within 66 samples
(+ optional embargo) of any val/test sample. Estimated cost on the real record:
11-16 % of the train samples per fold (embargo 0). FIXED (`bd94997`):
`_purge` in `datamodule.py` (test > val > train, gap = window_size - 1 +
lead_time = 66, embargo 0). Fold-0 test run 28205723 (W&B `uks7ze7s`)
completed with the purge: r 0.8678, MAE 0.298 °C (fold-0 test set only).
Zigzag split, real data: train -9.1 to -14.3 %, val -4.5 to -11.6 %
per fold; with embargo 30 (gap 96, not implemented) train -13.3 to -20.8 %,
val -6.6 to -16.8 %.

## #57 P2-1 — three definitions of the North Sea box
- `src/utils/hobday.py`: lat 50-63, lon -5-13 on the 0.1° climatology
  (`mean_clim`, for the #40 delta only; the MHW threshold no longer uses this
  box, see #57 P2-3).
- `src/data/masking.py`: index slices lat[100:127], lon[150:187] on the 0.5° grid
  (same box as hobday.py).
- the `target` in `merged_daily.nc`: lat 51.0-62.5, lon -5.2-13.2 at 0.1°.
The hobday.py box does not match the target box (rms 0.047 °C vs 0.026 °C for
the box that does). Remaining effect (moved here from #57 P2-3): the #40 delta
still comes from the 0.1° climatology averaged over the hobday.py box; computed
over the target box instead it changes by RMS 0.0010 °C, max 0.0029 °C.

## #57 P2-2 — leak in `compute_stats` (`src/data/dataset.py`)
Input mean/std are computed on the contiguous slice `[min(train), max(train)+60]`.
With `stratified_kfold` that slice contains val and test days: 32-36 % of its
days are not inputs of any train sample (fold 0: 4,520 of 13,929 days, 2,291 of
them in test years). Only the input normalisation constants are affected; the
target statistics use the exact train indices. Reported effect: std -0.91 %,
mean +0.0014.
FIXED (21 Oct 2026, `e13001d`): input stats now use only the days covered by
train input windows (days i .. i + window_size - 1 for every train index i).
Fold 0 (zigzag + purge): 14,593 -> 8,096 days, none in val/test years; std
change ptho_bot -0.75 %, u10 -1.11 %, v10 -0.73 %, msl -1.38 %, ssr +1.99 %;
mean shifts <= 0.0014 std.

## #54 — events that span Dec 31 / Jan 1
Per-year Hobday processing splits real events at the year boundary. With the
current detection (Oliver `677c888`, training target after `3161023`) 7 of 78
events span Dec 31 / Jan 1 (2006-07, 2015-16, 2016-17, 2018-19, 2019-20,
2020-21, 2022-23). The year ranking in `datamodule.py` runs `threshold` +
`detect` once on the full contiguous series and only then counts days per
year, so it is not affected. No code in `src/`, `scripts/` or `tools/` runs
the detection year by year or computes event metrics (POD/FAR/F1) yet.
Design requirement for the Round A event-metrics code: event metrics
(POD/FAR/F1, detected / onset anticipated) must run `detect()` on the pooled
40-year predicted series (each year predicted by the fold where it is test),
not per fold or per year, so the 7 of 78 events that cross Dec 31 / Jan 1 are
not cut. Threshold: the one from the true target series.

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
`CNNLSTMModel` has no constructor defaults, `src/utils/checkpoints.py` reads
its keys from the constructor signature, and `load_model_config` raises on a
missing or extra key; nothing loads with `strict=False`. Always rebuild the
model from `model_config.json`.

## Training hyperparameters in the yaml
`EarlyStopping` patience and `ModelCheckpoint(save_top_k)` are read from the
yaml keys `early_stopping_patience` and `save_top_k`
(`scripts/train_partition.py:248-261`).

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
fold-to-fold spread. Round B: scientific changes one at a time (#57 P2-1,
#66; #65 fixed before Round A in `bd94997`). Existing Raven checkpoints
(Aug 22 2026) are the comparison baseline, not the final results.

## #65, continued: how the leak actually happens in `datamodule.py`
#65 lived in `datamodule.py`: samples are assigned by target year with no gap
at Dec 31/Jan 1, and test years are non-contiguous, so almost every test year
is adjacent to a train year. Fixed in `bd94997` (see #65 above).

## Bucket assignment is not balanced (resolved 14 Oct 2026)
Bucket assignment was plain round-robin (`i % n_folds`), so bucket 0 always
got the most extreme year of each group of five: test MHW days per fold
decreased systematically (261, 224, 207, 184, 171). Resolved: serpentine
assignment with the Oliver detection, test MHW days 404, 361, 349, 350, 349
(max-min 1.9 pp; after the leap-year delta fix, `3161023`). See the 14 Oct section.

## #57 P2-2, precise statement
Input normalisation stats used a contiguous time slice from the first to the
last train sample; with non-contiguous train years this included val and
test years. Target stats are train-only. Fixed in `e13001d` (see above).

## Round B hypothesis: the global average pool erases location
The CNN encoder ends in a global average pool, so the network has almost no
information about WHERE a pattern is. This could explain local-only >
full (masking tells it where to look), the linear ridge on the NS-box mean
beating the CNN, and zero-fill scoring higher than nearest (coastlines at
zero act as landmarks). Test: add lat/lon coordinate channels, or avoid the
global average pool, and compare full vs local.

## Round B: the quantile head is never evaluated
`test_step` only evaluates the mean head; `q_pred` is saved in
`test_predictions.npz` but not scored.

## Round B: more hyperparameters fixed in code
`log_var` clamp +-10, head hidden size 64.

## `quantile_weight` 0.3 and `quantile_tau` 0.9 were set by hand
Both were fixed on 19 Aug 2026 (`f35cb6e`, `docs/history/narrative.md:283-295`)
and never compared with other values: every quantile-head run in git and in
W&B uses 0.9 / 0.3. tau 0.9 matches the Hobday p90. `mse_quantile` keeps the
same 0.3 (owner's decision); the effective share of the pinball term is to be
reported per model.

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
| 7 | `p90_thresh` stored without 31-day smooth | OBSOLETO | none | `load_ns_p90` and `p90_thresh` are no longer used (the threshold is computed on the series, `677c888`). |
| 8 | No unbatched per-sample IG loops | OBSOLETO | none | No IG/inference scripts in `rebuild`. |
| 9 | MHW needs >=5 days + gap <=2 | RESUELTO | `hobday.py:97-123` | Fixed in `677c888`: `detect` drops runs < 5 days first, then joins gaps <= 2 (Oliver). |
| 10 | CCM unreliable on daily data | DUDA | none | Paper caveat, no CCM code in `rebuild`. |
| 11 | No silent fallbacks | RESUELTO | `CLAUDE.md` | Rule in CLAUDE.md; no `except` in `src/`, `scripts/`, `tests/`, `tools/`. |
| 12 | Old experiment names | OBSOLETO | none | `rebuild` configs are `gnll`, `gnll_quantile`, `mse`. |
| 13 | Forbidden `merged_daily_deepSST.nc` | RESUELTO | `paths.py`, `CLAUDE.md` | Data file only from `$MHW_DATA_FILE`; rule in CLAUDE.md. |
| 14 | Gulf Stream narrative superseded | DUDA | none | Claim about the paper text, not about code. |
| 15 | `LazyDataset.is_land` attribute name | RESUELTO | `dataset.py:50` | Attribute exists; the scripts that misused it are gone. |
| 16 | Hardcoded architecture at checkpoint load | RESUELTO | `checkpoints.py:12-55`, `cnn_lstm.py:113-125` | No constructor defaults, keys read from the signature, `load_model_config` raises on missing/extra keys, no `strict=False`. |
| 17 | W&B identifiers hardcoded | RESUELTO | `train_partition.py:265-271` | From `WANDB_ENTITY`/`WANDB_PROJECT`, raises if unset. |
| 18 | "Vectorised" comment on a loop | OBSOLETO | none | `_compute_trend` no longer exists. |
| 19 | `to_anom > 0` as MHW label | OBSOLETO | none | No evaluation code in `rebuild`. |
| 20 | Dead XAI wrappers | OBSOLETO | none | `src/xai/` not in `rebuild`. |
| 21 | Pre-June-2026 TbotAtm on absolute `ptho_bot` | OBSOLETO | none | Old results; `rebuild` retrains on `merged_daily.nc`. |
| 22 | Silent routing to the `_OLD` file | OBSOLETO | none | `thermal_inertia_test.py` not in `rebuild`. |
| 23 | NumPy RNG API in split code | OBSOLETO | `datamodule.py:65-77`, `:189` | `stratified_kfold` uses no RNG (deterministic ranking). Rule kept in CLAUDE.md. |
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
| 40 | `mean_clim` not 31-day smoothed | RESUELTO | `dataset.py:125-141`, yamls | `hobday_smooth_target: true` in all 3 yamls. Not regenerated at source. |
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
| 54 | Per-year Hobday cuts Dec 31/Jan 1 events | ABIERTO | none yet | Design requirement for the Round A event-metrics code: `detect()` on the pooled 40-year predicted series, threshold from the true target. See #54 above. |
| 55 | Land pixels in `ptho_bot` stats (nearest) | RESUELTO | `dataset.py:189-195` | Land excluded for ocean variables under both fill modes. |
| 56 | def2 climatology, onset script, v1 configs | OBSOLETO | none | The three scripts/configs are not in `rebuild`. |
| 57 P1 | IG sampled only 1985 | OBSOLETO | none | No XAI code. |
| 57 P2-1 | Three NS box definitions | RONDA B | `hobday.py:30-31`, `masking.py:17-18` | Listed for Round B in the plan above. Includes the delta part moved from #57 P2-3. |
| 57 P2-2 | `compute_stats` contiguous-slice leak | RESUELTO | `dataset.py:155-212` | Fixed in `e13001d`: input stats from train-window days only. |
| 57 P2-3 | Clim 0.1° vs inputs 0.5° | RESUELTO | `hobday.py:62-94` | For the threshold, fixed in `677c888` (computed on the box-mean series). The delta part moved to #57 P2-1. |
| 58 | Relaunch into a dirty `output_dir` | RESUELTO | `train_partition.py:42-53`, `checkpoints.py:66-73` | Raises on existing checkpoints and on `-vN.ckpt` files. |
| 59 | GradCAM on `(batch, 2)` output | OBSOLETO | none | No XAI code. |
| 60 | Eval scripts bypass local/remote masking | RESUELTO | `cnn_lstm.py:174-189` | Masking now lives in `_encode`, shared by every forward path. |
| 61 | Spatial pipeline paths | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 62 | Spatial val-year shuffle | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 63 | Spatial `compute_stats` masking | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 64 | Spatial onset uses `tgt>0` | OBSOLETO | none | Spatial pipeline not in `rebuild`. |
| 65 | No purge at Dec 31/Jan 1 | RESUELTO | `datamodule.py:79-99`, `:203-207` | Fixed in `bd94997` (`_purge`); fold-0 test run 28205723. See #65 above. |
| 66 | `window_size: 60` unjustified | RONDA B | yamls | See #66 above. |

Stale text found in this file while triaging was corrected in a follow-up commit.

Counts (69 rows): RESUELTO 33, OBSOLETO 29, ABIERTO 1, RONDA B 2, DUDA 4
(updated 22 Oct 2026: #57 P2-2, #65, #9 and #57 P2-3 fixed; #7 obsolete).

## Revisión línea a línea (Cris), 14 oct 2026

### Plan
- Line by line: `dataset.py`, `hobday.py`, `cnn_lstm.py` (model and loss),
  `datamodule.py`.
- By blocks: `train_partition.py`, `checkpoints.py`, `levante.sbatch`.
- Not read: `equivalence.py`, `smoke.py`, tests.
- Round A (5 folds) is launched ONLY when this review is finished.

### Reviewed
- `masking.py` + `test_masking.py` (`ef4258b`, `302f4ff`). Do NOT rename
  `mask_remote`/`mask_local`. Round B: masking with 0 is debatable; check
  robustness with cropping and permutation.
- `paths.py` (`0d79159`). `MHW_EXPERIMENTS_DIR` is required.
  `MHW_REFERENCE_CKPT_DIR` (used by `tools/equivalence.py`) was missing from
  `.env.example`; added 15 Oct 2026.
- `checkpoints.py` (`0c5eeb4`, by blocks, docstrings only).

### `hobday.py`: REVIEWED
Findings 1-4 resolved in `677c888` (Oliver detection, docstrings); validated
against `marineHeatWaves.detect` (see "MHW detection on anomalies"). Finding 5
is #57 P2-3 / #57 P2-1.
Findings:
1. `load_ns_p90` averages the per-pixel p90 of the 0.1° climatology over the
   box and compares it with the box-mean anomaly. Mean of percentiles is not the
   percentile of the mean: the threshold is biased by about +0.16 °C (+0.04 to
   +0.30 °C depending on doy), and only 6.2 % of reference-period days exceed
   it (Oliver threshold on the box-mean series: 10.1 %).
2. `apply_hobday` fills gaps <=2 days BEFORE requiring 5 days; Oliver requires
   >=5 days first and then joins events across gaps <=2. Example: `XXX.XX` is
   an MHW with the current code and is not one with Oliver.
3. The docstring says +-15 days. Typo: the climatology uses
   `ydrunpctl,90,11` and `ydrunmean,11` (+-5).
4. The docstring of `load_ns_mean_clim_smooth_delta` refers to `mean_clim.nc`
   (the file is `sst_climatology_doy.nc`) and to `known_issues.md` (only in
   `docs/history/`; the numbers are in #40 above).
5. Climatology at 0.1° (700 x 1000), inputs at 0.5° (141 x 201) (#57 P2-3).

DECISION: MHW detection follows Oliver's reference implementation of Hobday et
al. (2016) (`marineHeatWaves.detect`, github.com/ecjoliver/marineHeatWaves,
commit `d7292bf`), reimplemented in our own words (the package is GPL-3) and
cited. Threshold computed on the series being analysed (the box-mean target),
window +-5 days by index, linear `nanpercentile` p90, periodic 31-day smoothing,
reference 1985-2014, first >=5 days then join gaps <=2, doy for a 365-day
calendar without Feb 29. This is a DELIBERATE change of definition: on the
training target (with the #40 delta), MHW days 1047 -> 1822 and events
55 -> 80, 1985-2024 (1813 days and 78 events after the leap-year delta fix,
`3161023`). The narrative says 52 events (current code on the target
without the delta): update it.

### Detection decides the split
`datamodule._mhw_days_per_year` ranks years by MHW days, so the detection
decides which years are train/val/test. Whether to freeze the current
years or recompute the assignment was pending; decided below.

Split: DECIDED (14 Oct 2026). Freezing the current years was not justified on
its own (they came from the faulty detection), even though they are balanced
under the corrected definition (2.0 pp). Adopted instead: serpentine (zigzag)
assignment of the years ranked by Oliver MHW days, ties by increasing year
(`datamodule._rank_years`, `_serpentine_buckets`). Updated 21 Oct 2026 after
the leap-year delta fix (`3161023`), which changed 2004 (23 -> 18 MHW days) and
2024 (107 -> 102) and swapped 1990, 2004 <-> 2005, 2024 between buckets 2 and
3 (folds 0 and 4 unchanged; fold 1 val, folds 2 and 3 test and val changed).
Test MHW days per fold 404, 361, 349, 350, 349 (13.8, 12.4, 12.0, 12.0, 12.0 %,
max-min 1.9 pp; events 17, 17, 15, 17, 18); val is the same list shifted by
one fold (1.9 pp); train max-min 0.8 pp. 13 of the 40 years have 0 MHW days,
so their order comes from the tie-break alone. Before the fix: 404, 361, 352,
357, 348.

| Fold | Test years | Val years |
|---|---|---|
| 0 | 1987, 1988, 1989, 1997, 2013, 2014, 2019, 2022 | 1986, 1991, 2002, 2009, 2011, 2012, 2016, 2023 |
| 1 | 1986, 1991, 2002, 2009, 2011, 2012, 2016, 2023 | 1985, 1993, 2005, 2007, 2010, 2015, 2018, 2024 |
| 2 | 1985, 1993, 2005, 2007, 2010, 2015, 2018, 2024 | 1990, 1994, 2000, 2001, 2004, 2006, 2008, 2017 |
| 3 | 1990, 1994, 2000, 2001, 2004, 2006, 2008, 2017 | 1992, 1995, 1996, 1998, 1999, 2003, 2020, 2021 |
| 4 | 1992, 1995, 1996, 1998, 1999, 2003, 2020, 2021 | 1987, 1988, 1989, 1997, 2013, 2014, 2019, 2022 |

Consequences:
- Equivalence check: EXCEPTION approved by the owner (14 Oct 2026).
  `tools/equivalence.py --check` fails by design after this change, because
  the split (and with it the test samples and the normalisation statistics)
  changes; the model code does not. Order: commit the code, smoke test, the
  owner retrains fold 0, point `MHW_REFERENCE_CKPT_DIR` at the new checkpoint,
  `--generate`, `--check`, commit the new reference.
  CLOSED (15 Oct 2026): reference regenerated (`--generate --force`) from job
  28138143 (code at `725217a`, zigzag split), best checkpoint epoch 10
  (`cnn-lstm-epoch=10-val_loss=0.0212.ckpt`), copied to
  `/work/bg1446/u241379/reference_ckpt_zigzag_725217a`. `--check`: "OK:
  bit-for-bit identical". Fold 0 zigzag: r 0.8715, MAE 0.323 °C (test set of
  that fold only). The check is back to being required before every commit.
- The fold-0 retrain of job 27965937 (r 0.827) used the old split and is no
  longer comparable fold by fold. Round A is compared with the pooled 5-fold
  August result, not fold by fold.
- August reference r (0.814) is POOLED: all test predictions of the 5 folds
  together, in °C, mean head (`scripts/eval_recall_v2_partition.py` in tag
  `v1-exploracion`, job 29463377), not the mean of the per-fold r. The table
  column `r_mean` means "mean head", not "mean of folds". Round A reports it the
  same way (pooled) as the main number, plus mean +- standard deviation of the
  per-fold r.
- To fix in `docs/history/narrative.md`: line 1587 calls a pooled r (0.850)
  "5-fold average".
- `docs/history/narrative.md` must update the 52 events: with the Oliver
  detection there are 78 events (1813 MHW days) on the training target (with
  the #40 delta, after the leap-year fix `3161023`), 75 events (1823 days) on the
  target without the delta.
- `load_ns_mean_clim_smooth_delta` is unchanged. It keeps its default
  arguments (`smooth_days=31`, `clim_file=CLIM_FILE`) TEMPORARILY, against the
  no-defaults rule. The leap-day of the delta is now fixed (`3161023`), so the
  defaults can go; removing them is a code change pending the owner's OK.

Balance, MHW days in the 8 test years of each fold (calendar days; an event is
counted in every split where it has at least one day). Computed with a temporary
script outside git on the training target; the first row reproduces the split
of job 27965937 exactly.

| Split | Detection | Test MHW days f0-f4 | Test events f0-f4 | Test MHW % f0-f4 | max-min |
|---|---|---|---|---|---|
| current years | current | 261, 224, 207, 184, 171 | 11, 10, 13, 12, 11 | 8.9, 7.7, 7.1, 6.3, 5.9 | 3.1 pp |
| current years | Oliver | 388, 379, 367, 357, 331 | 18, 17, 18, 18, 15 | 13.3, 13.0, 12.6, 12.2, 11.3 | 2.0 pp |
| recomputed with Oliver | current | 209, 203, 212, 223, 200 | 12, 13, 14, 9, 9 | 7.2, 7.0, 7.3, 7.6, 6.8 | 0.8 pp |
| recomputed with Oliver | Oliver | 454, 389, 352, 329, 298 | 20, 19, 15, 17, 15 | 15.5, 13.3, 12.1, 11.3, 10.2 | 5.3 pp |

With the current years and the Oliver detection no fold is clearly unbalanced
(11.3-13.3 % test MHW days, 15-18 events). Recomputing the round-robin with
Oliver gives a WORSE test balance (5.3 pp, steadily decreasing from fold 0 to
fold 4), the known round-robin bias.

### Other open findings
- (a) Leap-year doy shift: the calendar has no Feb 29 (14,600 days) but
  `dayofyear` goes up to 366, so in leap years every day from Mar 1 was indexed
  one doy ahead, and the `>= 365 -> 365` clamp merged Dec 30 and Dec 31.
  FIXED (21 Oct 2026, `3161023`): `LazyDataset.doys` now comes from `noleap_doy`
  and the delta is indexed with it. Effect on the training target: 3050 of
  14600 days change (all in leap years from Mar 1), max 0.0173 °C; MHW 1822
  days / 80 events -> 1813 / 78. The old threshold (`load_ns_p90`) no longer
  exists. How CDO indexed `sst_climatology_doy.nc` is still not verified.
- (b) The target box (51-62.5 N, 5.2 W-13.2 E at 0.1°) differs from the
  masking/hobday box (50-63 N, 5 W-13 E) (#57 P2-1).
- (c) The absolute box-mean SST is not stored in any file; only the anomaly
  `target` is.

## Data provenance
Found 16 Oct 2026; not stated in the attributes of `merged_daily.nc` nor in
`docs/history/data.md`. The SST (`target`, `to_anom`) and `ptho_bot` come from
an ocean-only ICON-O + HAMOCC hindcast (ICON-OES-MM-COAST, MPI-M/DWD),
experiment `hamocc_era5_244_cerosinc_pco2_cAR9` by Moritz Mathis (g260161),
directory `/work/gg1426/g260161/icon-oes-mm-coast_relaunch2/experiments/hamocc_era5_244_cerosinc_pco2_cAR9/`
(named in the `history` attribute of `ptho_bot_all_data.nc`).
- Forcing: hourly ERA5 fluxes and wind stress read from file, no coupled
  atmosphere (`NAMELIST_hamocc_era5_244_cerosinc_pco2_cAR9`, `&ocean_forcing_nml`:
  `iforc_oce = 12`, `forcing_fluxes_type = 1`, `forcing_windstress_u/v_type = 1`,
  `forcing_frequency = 3600`; the experiment's `ocean-flux.nc` links to
  `/work/gg1426/g260161/forcing/era5/icon_244_0158km60/era5_icon_202501_1h.nc`).
- Run from 1940 (chunks 1940-1959 ... 2020-2024).
- No temperature relaxation (`type_surfRelax_Temp = 0`); salinity relaxed to
  WOA13 with a 3-month timescale (`type_surfRelax_Salt = 4`,
  `para_surfRelax_Salt = 3.0`).
- The run script lives under `/work/bg1184/` (no access), so the per-year
  forcing file handling was not checked; `ocean-flux.nc` currently points to the
  January 2025 file.
- `preprocess_all.py`, which built `merged_daily.nc` from the ICON output, lives
  outside this repo (see `docs/history/data.md`); not read yet. PENDING.

## MHW detection on anomalies (decision, 16 Oct 2026)
DECISION: MHW days and events are detected on the box-mean SST ANOMALY (the
training target with the #40 delta), definition A. The pipeline code does not
change. Hobday et al. (2016) detect on absolute SST (definition B); the two are
not equivalent (numbers updated 21 Oct 2026 after the leap-year delta fix,
`3161023`, which `sandbox/abs_vs_anom.py` now also applies):
- A gives 78 events (1813 days), B gives 64 events (1623 days), 1985-2024;
  61 of the 78 A events overlap a B event, 62 of the 64 B events overlap an A
  event. Cause: on absolute SST the +-5-day window pools days with different
  seasonal means (the climatology changes up to 0.083 °C/day), which widens the
  distribution. On the anomaly scale the B threshold is higher than the A
  threshold in winter/spring (+0.04 to +0.11 °C, Dec-May) and lower in Aug-Sep
  (-0.07, -0.05 °C); max 0.137 °C.
- `hobday.py` matches `marineHeatWaves.detect`
  (github.com/ecjoliver/marineHeatWaves, commit d7292bf) on 14,599 of 14,600
  days of the absolute series (64 events in both; 1623 vs 1624 MHW days). The
  one differing day (1998-03-05) comes from calendar handling: Oliver smooths
  a 366-day threshold with an interpolated Feb 29, we use a 365-day year, so
  thresholds differ by at most 0.0037 °C within +-15 days of Feb 29 (1e-14
  elsewhere); the detection logic is identical. On 1998-03-05 the SST
  (6.7587 °C) is 0.0004 °C above Oliver's threshold and 0.0026 °C below ours.
  Before the leap-year fix the two matched on all days.
- Against NOAA OISST v2.1 on the same box (below), daily F1 is 0.72 for A
  (model A vs OISST A) and 0.70 for B (model B vs OISST B). A is chosen: it is
  marginally closer to the observations and it is the scale of the training
  target. The paper must state that detection is on anomalies, not absolute SST.
Absolute SST was reconstructed as target (with delta) + box mean (target box,
`skipna`) of the 31-day-smoothed `mean_clim`; the exact land mask of the target
is not documented. Scripts: `sandbox/abs_vs_anom.py`, `sandbox/oliver_check.py`.

Documented North Sea MHWs (Mohamed et al. 2023, Frontiers in Marine Science;
Mohamed et al., EGU24-8900; Jacobs et al. 2024, Frontiers in Marine Science):
all 7 appear in both A and B, several fragmented. Overlap = days of our event
inside the documented window.
| Documented event | A (anomaly) | B (absolute SST) |
|---|---|---|
| Jan-May 2007 (S North Sea + Channel, weak) | 24 Nov 2006-9 Feb (78 d) + 16 Feb-14 May (88 d); overlap 40 + 88 d | 25 Nov 2006-8 Feb (76 d) + 16 Feb-13 May (87 d); 39 + 87 d |
| Jul 2009 (N North Sea, ~3 weeks) | 30 Jun-10 Jul (11 d); 10 d | 1-10 Jul (10 d); 10 d |
| 31 Mar-9 May 2014 (S North Sea, ~40 d) | 3 pieces: 29 Mar-14 Apr, 20 Apr-2 May, 8-12 May; 15 + 13 + 2 d | 2 pieces: 31 Mar-13 Apr, 22 Apr-1 May; 14 + 10 d |
| Jun-Aug 2018 | 31 May-12 Jun (13 d) + 17 Jul-16 Aug (31 d); 12 + 31 d | 1-12 Jun (12 d) + 19 Jul-17 Aug (30 d); 12 + 30 d |
| Aug-Oct 2022 (peak Sep) | 1-15 Sep (15 d) + 24 Oct-14 Dec (52 d); 15 + 8 d | 25 Aug-16 Sep (23 d) + 24 Oct-12 Dec (50 d); 23 + 8 d |
| 13 Jun-4 Jul 2023 (NW North Sea) | 17 Jun-19 Jul (33 d); 18 d | same as A |
| Sep 2023 | 5 Sep-19 Oct (45 d); 26 d | same as A |
Max intensity of the matched events 0.95-1.79 °C. Several documented events
are regional (southern, northern or north-western North Sea) while our series
is the mean of the whole box, so they show up shorter or split into pieces
(2014 most clearly). The documented dates were taken as given, not re-checked
against the papers. Script: `sandbox/documented_events.py`.

## Validation against NOAA OISST v2.1 (16 Oct 2026)
OISST v2.1 daily, 0.25°, box 51.125-62.375 N, -5.125-13.125 E (46 x 74 cells,
1923 ocean), 1985-2024 without Feb 29, cos(lat)-weighted box mean ignoring land
(r 0.9377 vs 0.9369 unweighted). Data in `/work/bg1446/u241379/data/oisst/`
(outside the repo, 212 MB): the CoastWatch ERDDAP subset (`ncdcOisst21Agg_LonPM180`)
misses 1193 days (mostly 1992-1998 and 2021-2024), filled with the original NCEI
daily files (same product; max |NCEI - ERDDAP| 1.9e-6 °C on common days).
Same detection (`hobday.py`, same parameters), OISST A = anomaly vs its own
1985-2014 +-5-day climatology, 31-day smoothed.
- Daily r of anomalies, model vs OISST: 0.938. Absolute SST: r 0.991 (inflated
  by the shared seasonal cycle), model bias -0.27 °C.
- MHW days: model A 1813, OISST A 2303 (OISST B 2081). The model detects fewer
  MHW days than the satellite.
- Events (overlap >= 1 day), A vs A: 71/78 model events paired, 73/108 OISST
  events paired. B vs B: 60/64 and 64/106.
- Day level, model A vs OISST A (POD / FAR / F1): all 0.64 / 0.19 / 0.72;
  DEF 0.62 / 0.17 / 0.71; MAM 0.55 / 0.30 / 0.61; JJA 0.65 / 0.25 / 0.70;
  SON 0.73 / 0.05 / 0.83. Best SON, worst MAM. B vs B: all 0.62 / 0.20 / 0.70.
- Numbers updated 21 Oct 2026 after the leap-year delta fix (`3161023`); the
  model bias rounds to -0.27 °C (-0.266) and r is unchanged.
Scripts: `sandbox/oisst_fill_ncei.py`, `sandbox/oisst_compare.py`.

## W&B: wandb 0.25.1 silently fails to sync (19 Oct 2026)
With the current W&B server, `wandb sync` from wandb 0.25.1 cannot create the
run: `upsert_run` -> `server_settings_introspection()` gets `None`
(`CommError: 'NoneType' object has no attribute 'get'`, only in
`/tmp/debug-cli.<user>.log`), yet the CLI prints "done" and writes the
`.synced` marker. Job 28205723 (`uks7ze7s`) was lost this way and re-synced
with 0.31.0 (`--include-synced`) into `mhw-precursors-rebuild`. The venv now
has wandb 0.31.0 (`requirements.txt`). Any run synced with 0.25.1 must be
checked with `wandb.Api().run("<entity>/<project>/<id>")`; if missing,
re-sync it with `--include-synced`.

## `test_predictions.npz` keys renamed (28 Oct 2026)
Renamed in the commit "rename test_predictions.npz keys (pred, degC,
quantile_pred)" so the names say what they are (all values are SST
anomalies; `_norm` normalised units, `_degC` °C):

| Old | New |
|---|---|
| `mean_norm` | `pred_norm` |
| `mean_physical` | `pred_degC` |
| `y_physical` | `y_degC` |
| `q_pred_norm` | `quantile_pred_norm` |
| `q_pred_physical` | `quantile_pred_degC` |
| `log_var_physical` | `log_var_degC2` |

`y_norm` and `log_var_norm` are unchanged. npz files written before this
commit (`experiments_rebuild` jobs 28138143 and 28205723, and all August runs)
use the old names; scripts reading them must map the keys.
