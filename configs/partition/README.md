# configs/partition/ — structure index (Aug 22 2026)

Not physically reorganized yet — `lead30_landfill/fold{3,4}.yaml` are
about to be read by still-PENDING SLURM array tasks (job 29457756), so
nothing in this directory gets moved/renamed until all of Aug 22 2026's
34 launched jobs finish (see `experiments/partition/README.md` for their
status). This file is the navigation aid in the meantime. Full cleanup
inventory/history: `docs/known_issues.md` #45.

## Current pipeline (`land_fill_mode=nearest`, decided Aug 22 2026)

| Directory | Purpose |
|---|---|
| `full_gnll_quantile_v2_landfill/` | Main model, no masking, lead=7d (also the sweep's lead=7 point) |
| `local/` | `--mode local_only` — masks everything outside the NS box |
| `remote/` | `--mode remote_only` — masks everything inside the NS box |
| `lead3_landfill/`, `lead5_landfill/`, `lead14_landfill/`, `lead30_landfill/` | Lead-time sweep points (3/5/14/30d); lead=7 is `full_gnll_quantile_v2_landfill/` above |

All identical hyperparameters (lr=5e-5, cosine, `gaussian_nll`+
`quantile_head` tau=0.9, `hobday_smooth_target`, `stratified_kfold`,
`land_fill_mode: nearest`) — only masking mode / `lead_time` /
`output_dir` differ between them.

## Superseded / historical — kept as record, do not launch

| Directory | What it is |
|---|---|
| `full_gnll_quantile_v2/` | The **zero**-fill committed model — kept, still the fair-comparison baseline (docs/narrative.md) |
| `full_gnll_quantile_v2_lr2e5/` | LR diagnostic, NOT adopted |
| `full_gnll_focal_v2/`, `full_mse_v3/` | Paso 5 alternative-loss variants, fold0-only, quantile head won |
| `full/`, `full_gnll/`, `full_gnll_quantile/`, `full_gnll_focal/`, `full_mse_v2/` | Pre-stratified_kfold or JUWELS-path v1 configs — see known_issues.md #45 for the full per-directory breakdown |
| `_deprecated_v1/` | `local.yaml`/`remote.yaml`, moved here Aug 21 2026 (known_issues.md #56.3) — `split_mode: kfold` (buggy), do not run |
| `_adhoc_swap/` | One-off weight-swap ablation configs (Aug 21 2026), not part of the regular pipeline |

## Planned cleanup (deferred, known_issues.md #45)

Consolidate the 10 `full*` variants into one consistent naming scheme
once this session's active jobs finish and there's time to do it
carefully (renaming `output_dir` in lockstep, re-verifying `best_ckpt()`
still resolves every checkpoint).
