# CLEANUP_INVENTORY.md — scalar pipeline (Fase 1, solo lectura)

Rama `cleanup/scalar-pipeline`, base tag `pre-cleanup-scalar-2026-09-25` (14f4b25). **Nada se ha borrado.**  
Alcance: `src/`, `scripts/` (incl. `analysis/`, `slurm/`), `tests/`, `configs/partition/`, `configs/kfold/`. Fuera de alcance y no tocado: `src_spatial/`, `scripts_spatial/`, `configs/spatial/`, `scripts/slurm/submit_spatial_tbotatm_folds0_1.sh`, `archive/` (ver notas).

Método: grafo de imports por AST (`src`, `scripts`, `tests`, `archive`, `src_spatial`, `scripts_spatial`); referencias por nombre de archivo en `.sh/.py/.yaml/.md`; artefactos de salida (`*.png/.npz/.npy/.json`) buscados literalmente en known_issues/narrative/audit_plan. Puntos de entrada: `train_partition.py`, `eval_event_detection.py`, `eval_recall_v2_partition.py`, `ig_partition_quantile.py`, `gradcam_quantile_partition.py`, `gradientshap_quantile_partition.py`, `occlusion_ptho_bot_sanity_check.py`. Alcanzables por import: 16 archivos (7 scripts + 9 `src/`).

## Resumen

| categoría | archivos | líneas |
|---|--:|--:|
| KEEP-CORE | 79 | 7340 |
| KEEP-PROVENANCE | 110 | 9228 |
| KEEP-PENDING | 9 | 1244 |
| DELETE | 31 | 6477 |
| DOUBT | 33 | 4896 |
| OUT-OF-SCOPE | 5 | 193 |
| **total clasificado** | **267** | **29378** |

Antes: 262 archivos / 29185 líneas en alcance (excluye OUT-OF-SCOPE).  
Después (si solo se borra DELETE): 231 archivos / 22708 líneas.  
Después (si además se borra todo DOUBT, cota inferior): 198 archivos / 17812 líneas.

Desglose de DELETE por tipo: .py: 24, .sh: 7, .yaml: 0.

## DOUBT — preguntas concretas

- `configs/kfold/SSTAtm.yaml` (41L) — Va con scripts/train.py.
- `configs/kfold/TbotAtm.yaml` (41L) — Va con scripts/train.py (mismo destino). Nota: test_checkpoints.py (obligatorio) lee EXPERIMENTS_DIR/kfold/TbotAtm_lstmonly_fold*; ¿siguen existiendo esos checkpoints en Raven?
- `configs/partition/_deprecated_v1/` (2 archivos) — #56.3/#45 los dejan a propósito como registro histórico; nunca produjeron checkpoints. Borrarlos contradice known_issues: ¿confirmas DELETE (y actualizo #56.3 en Fase 2/log) o KEEP?
- `configs/partition/full_mse_v2/` (5 archivos) — Config JUWELS nunca lanzada en Raven (#45 lo marca 'rename o delete si no se usa'); data.md lo cita para la migración de land_mask v2. ¿Se lanzó alguna vez en JUWELS? Si no, DELETE los 5 yaml + submit_mse_v2_partition.sh.
- `scripts/analysis/plot_combined_test_timeseries.py` (178L) — Figura de reunión (truth vs pred, 5 folds); no citada. ¿Se usa en el paper? Si no, DELETE con plot_loss_comparison_timeseries.py + sus 2 sbatch.
- `scripts/analysis/plot_loss_comparison_timeseries.py` (274L) — Va con plot_combined_test_timeseries.py.
- `scripts/analysis/plot_sustained_lead_histogram.py` (111L) — Va con sustained_lead_time.py.
- `scripts/analysis/plot_two_regimes_of_skill.py` (165L) — Va con sustained_lead_time.py (Figura 1 de la slide de persistencia).
- `scripts/analysis/sustained_lead_time.py` (392L) — Análisis 'sustained lead' de la reunión del 24-ago (3 scripts + 3 sbatch); ningún doc cita sus números. ¿Va al paper/slides finales (KEEP-PROVENANCE) o DELETE?
- `scripts/analysis/thermal_inertia_test.py` (636L) — Regla: lee merged_daily_deepSST_OLD (silencioso, #22) => DELETE. Pero su Step 1 (tau) puede hacer falta para verificar el tau antes de la ablación de window_size (#66) y CLAUDE.md lo lista. ¿DELETE (check_tau_methodology.py cubre el tau) o KEEP-PENDING?
- `scripts/composite_precursor_analysis.py` (270L) — Contiene el análisis composite (citado: ptho_bot 14/14 k) y el bloque Granger (a borrar). ¿Lo dejo entero como provenance (y anoto el bloque Granger como 'no citar', propuesta de refactor) o prefieres borrarlo entero y confiar en composite_bootstrap_ci.py? Ojo: composite_bootstrap_ci comprueba consistencia contra composite_ns_box_curve.npz que produce este script.
- `scripts/eval_partition_timeseries.py` (274L) — Pipeline de la figura de referencia pre-sesión (mse_v2, test_predictions.npz por fold). ¿Sigue vigente o lo sustituye plot_combined_test_timeseries.py? Si está sustituido, DELETE los 2 scripts + submit_eval_partition_timeseries.sh.
- `scripts/ig_masked_batched.py` (249L) — audit_plan lo declara CONFIRMED ('resultados masked-IG citables') pero #25 borró los checkpoints SSTAtm gnll masked. ¿Se cita algún resultado masked-IG en el paper? Si no, DELETE los 3 (batched, merge, casestudy_2023; ig_masked_model ya va a DELETE).
- `scripts/ig_masked_casestudy_2023.py` (371L) — Caso Jun-Jul 2023 con checkpoints masked fold2 ya borrados; ¿lo quieres para el paper (reentrenando) o DELETE?
- `scripts/ig_masked_merge.py` (184L) — Va con ig_masked_batched.py (mismo destino).
- `scripts/mhw_hobday_stats.py` (219L) — ¿Alguna figura/conteo de eventos Hobday (p.ej. '52 eventos') salió de aquí? Solo lo citan docs como patrón de código. Si no, DELETE.
- `scripts/permutation_importance.py` (244L) — ¿Se usó la importancia por permutación en algún póster/figura/número del paper? No hay cita en docs. Si no, DELETE (+ submit_permutation_importance.sh).
- `scripts/plot_test_timeseries.py` (473L) — Va con eval_partition_timeseries.py.
- `scripts/slurm/_composite_precursor_analysis.sh` — follows composite_precursor_analysis.py (Granger question)
- `scripts/slurm/_plot_combined_test_timeseries_adhoc.sh` — follows plot_combined/plot_loss group
- `scripts/slurm/_sustained_lead_time_adhoc.sh` — follows sustained_lead_time group
- `scripts/slurm/_sustained_lead_time_lead14_adhoc.sh` — follows sustained_lead_time group
- `scripts/slurm/submit_eval_partition_timeseries.sh` — follows eval_partition_timeseries group
- `scripts/slurm/submit_mse_v2_partition.sh` — follows full_mse_v2 config (never run on Raven, known_issues #45)
- `scripts/slurm/submit_permutation_importance.sh` — follows permutation_importance.py
- `scripts/slurm/submit_plot_loss_comparison.sh` — follows plot_combined/plot_loss group
- `scripts/slurm/submit_sustained_lead_time_remaining.sh` — follows sustained_lead_time group
- `scripts/train.py` (150L) — Entrenador kfold documentado en README (quick-start) y CONTRIBUTING. train_partition.py --mode full lo cubre. ¿Lo borro (y en Fase 2 actualizo README/CONTRIBUTING + configs/kfold) o lo mantengo como reproducción de los resultados kfold?

## Tabla completa

| archivo | categoría | motivo | evidencia | líneas |
|---|---|---|---|--:|
| `configs/kfold/SSTAtm.yaml` | DOUBT | kfold-era configs (buggy split_mode kfold); referenced by README quick-start via scripts/train.py |  | 41 |
| `configs/kfold/TbotAtm.yaml` | DOUBT | kfold-era configs (buggy split_mode kfold); referenced by README quick-start via scripts/train.py |  | 41 |
| `configs/partition/README.md` | KEEP-CORE | structure index of the config tree |  | 41 |
| `configs/partition/_adhoc_swap/nearest_ckpt_zero_input_fold0.yaml` | KEEP-PROVENANCE | weight-swap ablation configs (narrative Aug 21) |  | 48 |
| `configs/partition/_adhoc_swap/zero_ckpt_nearest_input_fold0.yaml` | KEEP-PROVENANCE | weight-swap ablation configs (narrative Aug 21) |  | 48 |
| `configs/partition/_deprecated_v1/local.yaml` | DOUBT | never produced checkpoints (output_dir ''), split_mode kfold; but #56.3/#45 say keep as historical record -> contradicts DELETE rule |  | 52 |
| `configs/partition/_deprecated_v1/remote.yaml` | DOUBT | never produced checkpoints (output_dir ''), split_mode kfold; but #56.3/#45 say keep as historical record -> contradicts DELETE rule |  | 52 |
| `configs/partition/full/fold0.yaml` | KEEP-PROVENANCE | v1 MSE partition, job 14199778 numbers (narrative Partition table) |  | 37 |
| `configs/partition/full/fold1.yaml` | KEEP-PROVENANCE | v1 MSE partition, job 14199778 numbers (narrative Partition table) |  | 37 |
| `configs/partition/full/fold2.yaml` | KEEP-PROVENANCE | v1 MSE partition, job 14199778 numbers (narrative Partition table) |  | 37 |
| `configs/partition/full/fold3.yaml` | KEEP-PROVENANCE | v1 MSE partition, job 14199778 numbers (narrative Partition table) |  | 37 |
| `configs/partition/full/fold4.yaml` | KEEP-PROVENANCE | v1 MSE partition, job 14199778 numbers (narrative Partition table) |  | 37 |
| `configs/partition/full_gnll/fold0.yaml` | KEEP-PROVENANCE | v1 GNLL '15.2%' baseline (known_issues #45: do not delete/rename) |  | 38 |
| `configs/partition/full_gnll/fold1.yaml` | KEEP-PROVENANCE | v1 GNLL '15.2%' baseline (known_issues #45: do not delete/rename) |  | 38 |
| `configs/partition/full_gnll/fold2.yaml` | KEEP-PROVENANCE | v1 GNLL '15.2%' baseline (known_issues #45: do not delete/rename) |  | 38 |
| `configs/partition/full_gnll/fold3.yaml` | KEEP-PROVENANCE | v1 GNLL '15.2%' baseline (known_issues #45: do not delete/rename) |  | 38 |
| `configs/partition/full_gnll/fold4.yaml` | KEEP-PROVENANCE | v1 GNLL '15.2%' baseline (known_issues #45: do not delete/rename) |  | 38 |
| `configs/partition/full_gnll_focal/fold0.yaml` | KEEP-PROVENANCE | v1 focal 5-fold cited (#45) |  | 43 |
| `configs/partition/full_gnll_focal/fold0_shorttest.yaml` | KEEP-PROVENANCE | v1 focal 5-fold cited (#45) |  | 43 |
| `configs/partition/full_gnll_focal/fold1.yaml` | KEEP-PROVENANCE | v1 focal 5-fold cited (#45) |  | 43 |
| `configs/partition/full_gnll_focal/fold2.yaml` | KEEP-PROVENANCE | v1 focal 5-fold cited (#45) |  | 43 |
| `configs/partition/full_gnll_focal/fold3.yaml` | KEEP-PROVENANCE | v1 focal 5-fold cited (#45) |  | 43 |
| `configs/partition/full_gnll_focal/fold4.yaml` | KEEP-PROVENANCE | v1 focal 5-fold cited (#45) |  | 43 |
| `configs/partition/full_gnll_focal_v2/fold0.yaml` | KEEP-PROVENANCE | Paso 4/5 fold0 variant comparison (narrative) |  | 47 |
| `configs/partition/full_gnll_focal_v2/fold0_shorttest.yaml` | KEEP-PROVENANCE | Paso 4/5 fold0 variant comparison (narrative) |  | 47 |
| `configs/partition/full_gnll_focal_v2/fold1.yaml` | KEEP-PROVENANCE | Paso 4/5 fold0 variant comparison (narrative) |  | 47 |
| `configs/partition/full_gnll_focal_v2/fold2.yaml` | KEEP-PROVENANCE | Paso 4/5 fold0 variant comparison (narrative) |  | 47 |
| `configs/partition/full_gnll_focal_v2/fold3.yaml` | KEEP-PROVENANCE | Paso 4/5 fold0 variant comparison (narrative) |  | 47 |
| `configs/partition/full_gnll_focal_v2/fold4.yaml` | KEEP-PROVENANCE | Paso 4/5 fold0 variant comparison (narrative) |  | 47 |
| `configs/partition/full_gnll_quantile/fold0.yaml` | KEEP-PROVENANCE | v1 quantile 5-fold cited (#45: do not delete/rename; reload ckpts) |  | 42 |
| `configs/partition/full_gnll_quantile/fold0_shorttest.yaml` | KEEP-PROVENANCE | v1 quantile 5-fold cited (#45: do not delete/rename; reload ckpts) |  | 42 |
| `configs/partition/full_gnll_quantile/fold1.yaml` | KEEP-PROVENANCE | v1 quantile 5-fold cited (#45: do not delete/rename; reload ckpts) |  | 42 |
| `configs/partition/full_gnll_quantile/fold2.yaml` | KEEP-PROVENANCE | v1 quantile 5-fold cited (#45: do not delete/rename; reload ckpts) |  | 42 |
| `configs/partition/full_gnll_quantile/fold3.yaml` | KEEP-PROVENANCE | v1 quantile 5-fold cited (#45: do not delete/rename; reload ckpts) |  | 42 |
| `configs/partition/full_gnll_quantile/fold4.yaml` | KEEP-PROVENANCE | v1 quantile 5-fold cited (#45: do not delete/rename; reload ckpts) |  | 42 |
| `configs/partition/full_gnll_quantile_v2/fold0.yaml` | KEEP-PROVENANCE | zero-fill committed model; fair-comparison baseline, ckpt reload for XAI/eval triangulation |  | 47 |
| `configs/partition/full_gnll_quantile_v2/fold1.yaml` | KEEP-PROVENANCE | zero-fill committed model; fair-comparison baseline, ckpt reload for XAI/eval triangulation |  | 47 |
| `configs/partition/full_gnll_quantile_v2/fold2.yaml` | KEEP-PROVENANCE | zero-fill committed model; fair-comparison baseline, ckpt reload for XAI/eval triangulation |  | 47 |
| `configs/partition/full_gnll_quantile_v2/fold3.yaml` | KEEP-PROVENANCE | zero-fill committed model; fair-comparison baseline, ckpt reload for XAI/eval triangulation |  | 47 |
| `configs/partition/full_gnll_quantile_v2/fold4.yaml` | KEEP-PROVENANCE | zero-fill committed model; fair-comparison baseline, ckpt reload for XAI/eval triangulation |  | 47 |
| `configs/partition/full_gnll_quantile_v2_landfill/fold0.yaml` | KEEP-CORE | canonical model family |  | 48 |
| `configs/partition/full_gnll_quantile_v2_landfill/fold1.yaml` | KEEP-CORE | canonical model family |  | 48 |
| `configs/partition/full_gnll_quantile_v2_landfill/fold2.yaml` | KEEP-CORE | canonical model family |  | 48 |
| `configs/partition/full_gnll_quantile_v2_landfill/fold3.yaml` | KEEP-CORE | canonical model family |  | 48 |
| `configs/partition/full_gnll_quantile_v2_landfill/fold4.yaml` | KEEP-CORE | canonical model family |  | 48 |
| `configs/partition/full_gnll_quantile_v2_landfill_hybrid/fold0.yaml` | KEEP-PENDING | hybrid state_feature model (open decision) |  | 54 |
| `configs/partition/full_gnll_quantile_v2_landfill_hybrid/fold1.yaml` | KEEP-PENDING | hybrid state_feature model (open decision) |  | 50 |
| `configs/partition/full_gnll_quantile_v2_lr2e5/fold0.yaml` | KEEP-PROVENANCE | LR diagnostic, cited (narrative Aug 21) |  | 47 |
| `configs/partition/full_mse_v2/fold0.yaml` | DOUBT | legacy JUWELS-path MSE (merged_daily_v2.nc), never run on Raven per #45; data.md cites it for the land-mask migration |  | 38 |
| `configs/partition/full_mse_v2/fold1.yaml` | DOUBT | legacy JUWELS-path MSE (merged_daily_v2.nc), never run on Raven per #45; data.md cites it for the land-mask migration |  | 38 |
| `configs/partition/full_mse_v2/fold2.yaml` | DOUBT | legacy JUWELS-path MSE (merged_daily_v2.nc), never run on Raven per #45; data.md cites it for the land-mask migration |  | 38 |
| `configs/partition/full_mse_v2/fold3.yaml` | DOUBT | legacy JUWELS-path MSE (merged_daily_v2.nc), never run on Raven per #45; data.md cites it for the land-mask migration |  | 38 |
| `configs/partition/full_mse_v2/fold4.yaml` | DOUBT | legacy JUWELS-path MSE (merged_daily_v2.nc), never run on Raven per #45; data.md cites it for the land-mask migration |  | 38 |
| `configs/partition/full_mse_v3/fold0.yaml` | KEEP-PROVENANCE | Paso 4 fold0 MSE baseline (narrative) |  | 45 |
| `configs/partition/lead14_landfill/fold0.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead14_landfill/fold1.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead14_landfill/fold2.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead14_landfill/fold3.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead14_landfill/fold4.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead30_landfill/fold0.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead30_landfill/fold1.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead30_landfill/fold2.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead30_landfill/fold3.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead30_landfill/fold4.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead3_landfill/fold0.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead3_landfill/fold1.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead3_landfill/fold2.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead3_landfill/fold3.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead3_landfill/fold4.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead5_landfill/fold0.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead5_landfill/fold1.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead5_landfill/fold2.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead5_landfill/fold3.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/lead5_landfill/fold4.yaml` | KEEP-CORE | current pipeline lead sweep |  | 48 |
| `configs/partition/local/fold0.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/local/fold1.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/local/fold2.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/local/fold3.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/local/fold4.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/remote/fold0.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/remote/fold1.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/remote/fold2.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/remote/fold3.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/partition/remote/fold4.yaml` | KEEP-CORE | current pipeline (README) |  | 48 |
| `configs/spatial/SSTAtm_fold0.yaml` | OUT-OF-SCOPE | spatial pipeline; not touched |  | 29 |
| `configs/spatial/SSTAtm_phys_fold0.yaml` | OUT-OF-SCOPE | spatial pipeline; not touched |  | 31 |
| `configs/spatial/TbotAtm_fold0.yaml` | OUT-OF-SCOPE | spatial pipeline; not touched |  | 29 |
| `configs/spatial/TbotAtm_fold1.yaml` | OUT-OF-SCOPE | spatial pipeline; not touched |  | 29 |
| `scripts/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 1 |
| `scripts/analysis/_adhoc_eval_extreme_recall.py` | KEEP-PROVENANCE | 15.2%/19.0%/34.9% extreme-day recall numbers (narrative, known_issues #41) | sbatch: eval_extreme_recall_adhoc.sh; cited in known_issues.md, narrative.md | 146 |
| `scripts/analysis/calibrate_mhw_area_threshold.py` | KEEP-CORE | produces area_frac_timeseries.npy = def2 ground truth read by eval_recall_v2/ig_partition_quantile/eval_event_detection lineage (known_issues #41/#56.1) | cited in audit_plan.md, known_issues.md, narrative.md | 164 |
| `scripts/analysis/causal_triangulation.py` | DELETE | Granger causality + CCM: Granger explicitly discarded (p=0.0000 false positives, narrative Item 4), CCM saturates (#10). Deleted by explicit instruction | cited in CLAUDE.md, audit_plan.md | 289 |
| `scripts/analysis/check_tau_methodology.py` | KEEP-PENDING | tau/ACF from raw to_anom; needed to verify tau claim before any window_size ablation (known_issues #66); tau_check2_boxplot.png must be regenerated (audit NF-P4-E) | cited in CLAUDE.md, audit_plan.md | 445 |
| `scripts/analysis/contaminated_vs_clean_years_recall.py` | KEEP-PENDING | decisive seam-distance check for the Dec31/Jan1 leakage question (known_issues #65 deferred); +14.8pp result | sbatch: submit_contaminated_vs_clean_recall.sh; cited in known_issues.md | 332 |
| `scripts/analysis/event_detection_lead_time_comparison.py` | KEEP-PROVENANCE | headline event_detection lead-time sweep table/figure (narrative Aug 24) | cited in narrative.md | 214 |
| `scripts/analysis/ig_coastal_decay_check.py` | KEEP-PROVENANCE | coastal-decay ratios 18.76x/12.99x (known_issues #52/#55, narrative) | cited in narrative.md | 90 |
| `scripts/analysis/incremental_value_regression.py` | KEEP-PROVENANCE | OLS incremental-value table (narrative Aug 23) | sbatch: submit_incremental_value_regression.sh; cited in narrative.md | 163 |
| `scripts/analysis/lead_time_sweep_model_vs_persistence.py` | KEEP-PROVENANCE | lead-time sweep model vs persistence figure (narrative Aug 22) | cited in narrative.md | 155 |
| `scripts/analysis/linear_ceiling_ridge.py` | KEEP-PROVENANCE | linear ridge ceiling r=0.8797 (narrative Aug 23) | sbatch: submit_linear_ceiling_ridge.sh; cited in narrative.md | 133 |
| `scripts/analysis/mhw_definition_agreement_and_recall.py` | KEEP-PROVENANCE | def1/def2 confusion matrix + recall v1 (narrative Paso 2) | cited in known_issues.md, narrative.md | 268 |
| `scripts/analysis/persistence_lag_sweep.py` | KEEP-PROVENANCE | persistence lag 1-14 table (narrative Item 2) | sbatch: _persistence_lag_sweep_adhoc.sh; cited in narrative.md | 95 |
| `scripts/analysis/persistence_recall_baseline.py` | KEEP-PROVENANCE | persistence recall/precision/FPR table (narrative Aug 22) | sbatch: submit_persistence_recall_baseline.sh; cited in narrative.md | 148 |
| `scripts/analysis/plot_combined_test_timeseries.py` | DOUBT | Aug 24 meeting-prep figures (truth-vs-pred 5 folds; loss comparison slide); no doc cites outputs | sbatch: _plot_combined_test_timeseries_adhoc.sh | 178 |
| `scripts/analysis/plot_fold_year_assignment.py` | KEEP-PROVENANCE | stratified_kfold vs kfold verification figure (known_issues #1/#42) | no importer / sbatch / doc reference | 165 |
| `scripts/analysis/plot_land_fill_comparison.py` | KEEP-PROVENANCE | land_fill verification (known_issues #52: ocean bit-identical, land std=0.34) | sbatch: _plot_land_fill_comparison_adhoc.sh; cited in known_issues.md | 118 |
| `scripts/analysis/plot_loss_comparison_timeseries.py` | DOUBT | Aug 24 meeting-prep figures (truth-vs-pred 5 folds; loss comparison slide); no doc cites outputs | sbatch: submit_plot_loss_comparison.sh | 274 |
| `scripts/analysis/plot_mean_clim_smooth_correction.py` | KEEP-PROVENANCE | verification figure for known_issues #40 (RMS 0.046C / max 0.131C) | no importer / sbatch / doc reference | 112 |
| `scripts/analysis/plot_prediction_vs_target.py` | KEEP-PROVENANCE | prediction_vs_target_2014.png (narrative Aug 21 XAI entry) | sbatch: _plot_prediction_vs_target_adhoc.sh | 183 |
| `scripts/analysis/plot_sustained_lead_histogram.py` | DOUBT | Aug 24 meeting-prep 'sustained lead' analysis (answers question about eval_event_detection lead-time definition); no doc cites its output; 3 sbatch | no importer / sbatch / doc reference | 111 |
| `scripts/analysis/plot_two_regimes_of_skill.py` | DOUBT | Aug 24 meeting-prep 'sustained lead' analysis (answers question about eval_event_detection lead-time definition); no doc cites its output; 3 sbatch | no importer / sbatch / doc reference | 165 |
| `scripts/analysis/quantile_calibration_check.py` | KEEP-PENDING | quantile coverage collapse table (narrative Aug 22); open item: conditional recalibration | cited in narrative.md | 84 |
| `scripts/analysis/quantile_head_recall.py` | KEEP-PROVENANCE | v1 quantile-head recall 81.8%/44.7% (narrative Aug 20-21) | sbatch: _quantile_head_recall_adhoc.sh; cited in narrative.md | 207 |
| `scripts/analysis/quantile_head_recall_v2_all5.py` | KEEP-PROVENANCE | pooled 5-fold quantile-head recall 80.7%/48.7% + #56.1 corrected rerun | sbatch: _quantile_head_recall_v2_all5_adhoc.sh, submit_eval_recall_v2_all_families.sh; cited in audit_plan.md, known_issues.md, narrative.md | 200 |
| `scripts/analysis/quantile_head_recall_v2_fold0.py` | KEEP-PROVENANCE | v2 fold0 early read 57.5%/87.0% (narrative Paso 5) | sbatch: _quantile_head_recall_v2_fold0_adhoc.sh | 206 |
| `scripts/analysis/quantile_pr_curve_analysis.py` | KEEP-PROVENANCE | PR-curve/AUPRC + 'never dominates persistence' result (narrative Aug 22) | cited in narrative.md | 173 |
| `scripts/analysis/raw_ptho_bot_coastal_check.py` | KEEP-PROVENANCE | raw-data ground truth 0.87x decay / 1.39x NS-box enrichment (known_issues #52) | cited in known_issues.md, narrative.md | 113 |
| `scripts/analysis/sustained_lead_time.py` | DOUBT | Aug 24 meeting-prep 'sustained lead' analysis (answers question about eval_event_detection lead-time definition); no doc cites its output; 3 sbatch | sbatch: _sustained_lead_time_adhoc.sh, _sustained_lead_time_lead14_adhoc.sh, submit_sustained_lead_time_remaining.sh | 392 |
| `scripts/analysis/thermal_inertia_test.py` | DOUBT | Rule says DELETE (Step 2 silently routes to merged_daily_deepSST_OLD, #22; never completed). But its Step 1 tau code may be needed for the pending tau/window_size question | cited in CLAUDE.md, audit_plan.md, known_issues.md | 636 |
| `scripts/analysis/xai_diff_ig_summary_plot.py` | KEEP-PROVENANCE | corrected differential-IG summary figure (narrative Aug 24) | no importer / sbatch / doc reference | 161 |
| `scripts/analysis/xai_triangulation_summary_plot.py` | KEEP-PROVENANCE | 4-method XAI triangulation figure (narrative Aug 22/24) | cited in narrative.md | 173 |
| `scripts/composite_bootstrap_ci.py` | KEEP-PROVENANCE | ptho_bot composite bootstrap CI table (14/14 k, narrative Item 4) | sbatch: _composite_bootstrap_ci.sh; cited in narrative.md | 203 |
| `scripts/composite_ig.py` | DELETE | IG/MHW composite with to_anom>0 label (known_issues #19, audit NEEDS_RERUN, results invalid); superseded by ig_partition_quantile --stratify_mhw | cited in audit_plan.md, known_issues.md | 297 |
| `scripts/composite_ig_signed.py` | DELETE | IG/MHW composite with to_anom>0 label (known_issues #19, audit NEEDS_RERUN, results invalid); superseded by ig_partition_quantile --stratify_mhw | cited in audit_plan.md, known_issues.md | 262 |
| `scripts/composite_precursor_analysis.py` | DOUBT | Mixes composite (cited) with Granger causality (documented broken, to be deleted). Deleting only the Granger block is an edit, not a delete | sbatch: _composite_bootstrap_ci.sh, _composite_precursor_analysis.sh; cited in narrative.md | 270 |
| `scripts/diag_attention.py` | DELETE | attention diagnostic; audit NEEDS_RERUN; #29 notes it would crash on attention_only checkpoints; imports archived run_xai.py | cited in audit_plan.md, known_issues.md | 213 |
| `scripts/ensemble_skill.py` | DELETE | old ensemble skill w/ to_anom>0 label (#19), pre-fix data (#21), broad except (#11); audit NEEDS_RERUN | cited in audit_plan.md, known_issues.md | 435 |
| `scripts/eval_event_detection.py` | KEEP-CORE | canonical eval: POD/FAR/CSI vs persistence + bootstrap | sbatch: _eval_event_detection_adhoc.sh, _sustained_lead_time_adhoc.sh, submit_event_detection_all_families.sh +1; cited in known_issues.md, narrative.md | 501 |
| `scripts/eval_ig.py` | DELETE | kfold-era signed-IG Hobday composite on old dataclass; superseded by ig_partition_quantile (stratified, quantile head, --stratify_mhw) | cited in audit_plan.md, known_issues.md | 344 |
| `scripts/eval_onset_persistence.py` | KEEP-PROVENANCE | onset table n=84 still printed in narrative.md ('Onset skill', Source: eval_onset_persistence.py); documented in README/CONTRIBUTING; superseded by eval_onset_skill_quantile_v2 but numbers remain cited | cited in audit_plan.md | 290 |
| `scripts/eval_onset_skill.py` | KEEP-PROVENANCE | onset table n=84 still printed in narrative.md ('Onset skill', Source: eval_onset_persistence.py); documented in README/CONTRIBUTING; superseded by eval_onset_skill_quantile_v2 but numbers remain cited | sbatch: submit_eval.sh; cited in CONTRIBUTING.md, README.md, audit_plan.md, data.md, known_issues.md | 313 |
| `scripts/eval_onset_skill_curve.py` | KEEP-PROVENANCE | skill-recovery curve table (narrative 'skill-recovery curve'; known_issues #53) | sbatch: _eval_onset_skill_curve_adhoc.sh; cited in known_issues.md, narrative.md | 277 |
| `scripts/eval_onset_skill_quantile_v2.py` | KEEP-PROVENANCE | onset/mid-event/no-MHW tables (narrative Aug 21-22, known_issues #53/#56.2) | sbatch: _eval_onset_skill_landfill_adhoc.sh, _eval_onset_skill_quantile_v2_adhoc.sh; cited in audit_plan.md, known_issues.md, narrative.md | 320 |
| `scripts/eval_partition_timeseries.py` | DOUBT | pre-session reference figure pipeline for mse_v2 predictions; no doc cites its outputs; possibly superseded by analysis/plot_combined_test_timeseries.py | sbatch: submit_eval_partition_timeseries.sh | 274 |
| `scripts/eval_recall_v2_partition.py` | KEEP-CORE | canonical eval: def1/def2 recall/precision/FPR per family | imported by ig_partition_quantile.py, contaminated_vs_clean_years_recall.py; sbatch: _dryrun_ig_diff_mhw_stratified.sh, submit_contaminated_vs_clean_recall.sh, submit_eval_recall_v2_all_families.sh +1; cited in audit_plan.md, known_issues.md, narrative.md | 243 |
| `scripts/eval_test_metrics_from_best_ckpt.py` | KEEP-PROVENANCE | fair-comparison r=0.8657/0.8237 table (narrative land_fill retrain) | sbatch: _eval_committed_fold0_best_ckpt.sh; cited in audit_plan.md, narrative.md | 72 |
| `scripts/gradcam_partition.py` | DELETE | MSE-era GradCAM pilot; superseded by gradcam_quantile_partition.py (head selection, stratified sampling); no numbers cited | sbatch: submit_gradcam_pilot.sh; cited in known_issues.md, narrative.md | 257 |
| `scripts/gradcam_quantile_partition.py` | KEEP-CORE | canonical XAI (GradCAM) | sbatch: submit_gradcam_extra_folds.sh, submit_gradcam_quantile_v2_fold0.sh, submit_gradcam_quantile_v2_landfill_fold0.sh; cited in audit_plan.md, known_issues.md, narrative.md | 170 |
| `scripts/gradientshap_quantile_partition.py` | KEEP-CORE | canonical XAI (GradientSHAP) | sbatch: submit_gradientshap_extra_folds.sh, submit_gradientshap_quantile_v2_fold0.sh, submit_gradientshap_quantile_v2_landfill_fold0.sh; cited in audit_plan.md, narrative.md | 234 |
| `scripts/ig_masked_batched.py` | DOUBT | masked-SSTAtm IG (batched). audit_plan says 'existing masked-IG results are citable' but #25 deleted the gnll masked experiment dirs | cited in audit_plan.md, known_issues.md | 249 |
| `scripts/ig_masked_casestudy_2023.py` | DOUBT | 2023 case study on masked SSTAtm checkpoints (fold2); cited only as code-pattern in known_issues | cited in known_issues.md | 371 |
| `scripts/ig_masked_merge.py` | DOUBT | masked-SSTAtm IG (batched). audit_plan says 'existing masked-IG results are citable' but #25 deleted the gnll masked experiment dirs | cited in audit_plan.md | 184 |
| `scripts/ig_masked_model.py` | DELETE | superseded by ig_masked_batched.py (its own docstring: same logic, batched); masked-SSTAtm era | cited in known_issues.md | 320 |
| `scripts/ig_partition_quantile.py` | KEEP-CORE | canonical XAI (IG); imports sampling.py | imported by test_ig_diff_head.py; sbatch: _dryrun_ig_diff_head.sh, _dryrun_ig_diff_mhw_stratified.sh, _dryrun_ig_diff_nearest.sh +8; cited in audit_plan.md, known_issues.md, narrative.md | 363 |
| `scripts/ig_signed_partition.py` | KEEP-PROVENANCE | SmoothGrad/8px-FFT diagnostic numbers (known_issues #26: 15.2%->20.4%); launched by submit_ig_partition*.sh | sbatch: submit_ig_partition.sh, submit_ig_partition_smoothgrad.sh, submit_ig_partition_smoothgrad_pilot.sh; cited in known_issues.md | 659 |
| `scripts/ig_simple.py` | DELETE | broken (known_issues #49: imports non-existent MHWDataset, wrong RNG, averages mean+log_var); replaced by ig_partition_quantile.py | sbatch: submit_ig_quantile_v2_fold0.sh; cited in known_issues.md, narrative.md | 173 |
| `scripts/local_only_skill.py` | DELETE | SSTAtm-era local/remote split on non-partition model; superseded by train_partition local_only/remote_only; #25 deleted its experiments | cited in known_issues.md | 208 |
| `scripts/mhw_ensemble_hobday.py` | KEEP-PROVENANCE | narrative: 23/52 events, recall 19.2%->30.8% (MC Hobday on GNLL) | cited in narrative.md | 176 |
| `scripts/mhw_hobday_stats.py` | DOUBT | audit_plan Phase 3 CONFIRMED; Hobday stats/monthly timeseries; no output cited by name; cited only as code-pattern reference | cited in audit_plan.md, known_issues.md, narrative.md | 219 |
| `scripts/occlusion_ptho_bot_sanity_check.py` | KEEP-CORE | canonical XAI (occlusion cross-check) | sbatch: _occlusion_ptho_bot_fold0.sh, _occlusion_ptho_bot_landfill_fold0.sh, submit_occlusion_extra_folds.sh; cited in audit_plan.md, known_issues.md | 251 |
| `scripts/permutation_importance.py` | DOUBT | TbotAtm variable importance by permutation; no numbers cited in docs (only known_issues #30 cnn_features list); sbatch exists | sbatch: submit_permutation_importance.sh; cited in known_issues.md | 244 |
| `scripts/persistence_baseline.py` | DELETE | lag-tau persistence vs old lead_sweep ensemble r; superseded by analysis/persistence_lag_sweep.py + lead_time_sweep_model_vs_persistence.py | cited in audit_plan.md | 219 |
| `scripts/persistence_remote_sst.py` | KEEP-PROVENANCE | Paso 7 persistence gate (r=0.9309, recall 54.6%/16.5%) narrative + known_issues #48 | sbatch: _persistence_baseline_adhoc.sh; cited in audit_plan.md, known_issues.md, narrative.md | 297 |
| `scripts/plot_experiments_summary.py` | DELETE | no importer/sbatch/doc reference; targets old experiments/ layouts (multiseed/lead_sweep, deleted in #25) | no importer / sbatch / doc reference | 434 |
| `scripts/plot_lead_sweep.py` | DELETE | no importer/sbatch/doc reference; targets old experiments/ layouts (multiseed/lead_sweep, deleted in #25) | no importer / sbatch / doc reference | 162 |
| `scripts/plot_split_scatter.py` | DELETE | old naming noSST/SST/deepSST (#12), audit ARCHIVE; plot_variable_* read merged_daily_deepSST_OLD (forbidden, #13) | cited in audit_plan.md, known_issues.md | 151 |
| `scripts/plot_test_timeseries.py` | DOUBT | pre-session reference figure pipeline for mse_v2 predictions; no doc cites its outputs; possibly superseded by analysis/plot_combined_test_timeseries.py | no importer / sbatch / doc reference | 473 |
| `scripts/plot_variable_scatter.py` | DELETE | old naming noSST/SST/deepSST (#12), audit ARCHIVE; plot_variable_* read merged_daily_deepSST_OLD (forbidden, #13) | cited in audit_plan.md, known_issues.md | 195 |
| `scripts/plot_variable_xai_panel.py` | DELETE | old naming noSST/SST/deepSST (#12), audit ARCHIVE; plot_variable_* read merged_daily_deepSST_OLD (forbidden, #13) | cited in audit_plan.md | 155 |
| `scripts/run_xai.py` | DELETE | audit ARCHIVE (Phase 1/2); crash bug #15/#16; imported only by DELETE scripts | imported by diag_attention.py, run_xai_ensemble.py; cited in audit_plan.md, known_issues.md, narrative.md | 589 |
| `scripts/run_xai_ensemble.py` | DELETE | audit ARCHIVE (Phase 1/2); crash bug #15/#16; imported only by DELETE scripts | cited in audit_plan.md, known_issues.md | 386 |
| `scripts/skill_scores.py` | DELETE | old ensemble skill w/ to_anom>0 label (#19), pre-fix data (#21), broad except (#11); audit NEEDS_RERUN | cited in known_issues.md | 463 |
| `scripts/slurm/_composite_bootstrap_ci.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 26 |
| `scripts/slurm/_composite_precursor_analysis.sh` | DOUBT | follows composite_precursor_analysis.py (Granger question) |  | 26 |
| `scripts/slurm/_dryrun_ig_diff_head.sh` | DELETE | one-off launch dry-runs (fast_dev_run) of already-launched configs/flags; bugs found are documented |  | 37 |
| `scripts/slurm/_dryrun_ig_diff_mhw_stratified.sh` | DELETE | one-off launch dry-runs (fast_dev_run) of already-launched configs/flags; bugs found are documented |  | 40 |
| `scripts/slurm/_dryrun_ig_diff_nearest.sh` | DELETE | one-off launch dry-runs (fast_dev_run) of already-launched configs/flags; bugs found are documented |  | 38 |
| `scripts/slurm/_dryrun_landfill_fold0.sh` | DELETE | one-off launch dry-runs (fast_dev_run) of already-launched configs/flags; bugs found are documented |  | 26 |
| `scripts/slurm/_dryrun_lr2e5.sh` | DELETE | one-off launch dry-runs (fast_dev_run) of already-launched configs/flags; bugs found are documented |  | 27 |
| `scripts/slurm/_eval_committed_fold0_best_ckpt.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 31 |
| `scripts/slurm/_eval_event_detection_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 25 |
| `scripts/slurm/_eval_onset_skill_curve_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 25 |
| `scripts/slurm/_eval_onset_skill_landfill_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 30 |
| `scripts/slurm/_eval_onset_skill_quantile_v2_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 26 |
| `scripts/slurm/_occlusion_ptho_bot_fold0.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 29 |
| `scripts/slurm/_occlusion_ptho_bot_landfill_fold0.sh` | KEEP-CORE | canonical XAI launchers (nearest fold0) |  | 29 |
| `scripts/slurm/_persistence_baseline_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 28 |
| `scripts/slurm/_persistence_lag_sweep_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 23 |
| `scripts/slurm/_plot_combined_test_timeseries_adhoc.sh` | DOUBT | follows plot_combined/plot_loss group |  | 26 |
| `scripts/slurm/_plot_land_fill_comparison_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 23 |
| `scripts/slurm/_plot_prediction_vs_target_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 24 |
| `scripts/slurm/_quantile_head_recall_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 29 |
| `scripts/slurm/_quantile_head_recall_v2_all5_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 28 |
| `scripts/slurm/_quantile_head_recall_v2_fold0_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 28 |
| `scripts/slurm/_sustained_lead_time_adhoc.sh` | DOUBT | follows sustained_lead_time group |  | 31 |
| `scripts/slurm/_sustained_lead_time_lead14_adhoc.sh` | DOUBT | follows sustained_lead_time group |  | 27 |
| `scripts/slurm/eval_extreme_recall_adhoc.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 30 |
| `scripts/slurm/submit_contaminated_vs_clean_recall.sh` | KEEP-PENDING | contaminated-vs-clean / hybrid (open decisions) |  | 28 |
| `scripts/slurm/submit_eval.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 39 |
| `scripts/slurm/submit_eval_partition_timeseries.sh` | DOUBT | follows eval_partition_timeseries group |  | 54 |
| `scripts/slurm/submit_eval_recall_v2_all_families.sh` | KEEP-CORE | canonical eval launchers |  | 42 |
| `scripts/slurm/submit_event_detection_all_families.sh` | KEEP-CORE | canonical eval launchers |  | 47 |
| `scripts/slurm/submit_event_detection_hybrid_vs_baseline.sh` | KEEP-PENDING | contaminated-vs-clean / hybrid (open decisions) |  | 51 |
| `scripts/slurm/submit_gnll_focal_partition.sh` | KEEP-PROVENANCE | launched cited v1 5-fold/diagnostic runs (jobs in narrative) |  | 56 |
| `scripts/slurm/submit_gnll_focal_shorttest.sh` | KEEP-PROVENANCE | launched cited v1 5-fold/diagnostic runs (jobs in narrative) |  | 53 |
| `scripts/slurm/submit_gnll_focal_v2_fold0.sh` | KEEP-PROVENANCE | launched cited v2 runs (jobs 29417248/405/406/29426208, lr2e5 29433645) |  | 46 |
| `scripts/slurm/submit_gnll_partition.sh` | KEEP-PROVENANCE | launched cited v1 5-fold/diagnostic runs (jobs in narrative) |  | 50 |
| `scripts/slurm/submit_gnll_quantile_partition.sh` | KEEP-PROVENANCE | launched cited v1 5-fold/diagnostic runs (jobs in narrative) |  | 65 |
| `scripts/slurm/submit_gnll_quantile_shorttest.sh` | KEEP-PROVENANCE | launched cited v1 5-fold/diagnostic runs (jobs in narrative) |  | 49 |
| `scripts/slurm/submit_gnll_quantile_v2_fold0.sh` | KEEP-PROVENANCE | launched cited v2 runs (jobs 29417248/405/406/29426208, lr2e5 29433645) |  | 43 |
| `scripts/slurm/submit_gnll_quantile_v2_landfill_fold0.sh` | KEEP-CORE | canonical model training (fold0 / folds1-4) |  | 43 |
| `scripts/slurm/submit_gnll_quantile_v2_landfill_folds1_4.sh` | KEEP-CORE | canonical model training (fold0 / folds1-4) |  | 48 |
| `scripts/slurm/submit_gnll_quantile_v2_landfill_hybrid_folds0_1.sh` | KEEP-PENDING | contaminated-vs-clean / hybrid (open decisions) |  | 60 |
| `scripts/slurm/submit_gnll_quantile_v2_lr2e5_fold0.sh` | KEEP-PROVENANCE | launched cited v2 runs (jobs 29417248/405/406/29426208, lr2e5 29433645) |  | 42 |
| `scripts/slurm/submit_gnll_quantile_v2_partition.sh` | KEEP-PROVENANCE | launched cited v2 runs (jobs 29417248/405/406/29426208, lr2e5 29433645) |  | 52 |
| `scripts/slurm/submit_gradcam_extra_folds.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 45 |
| `scripts/slurm/submit_gradcam_pilot.sh` | DELETE | launches gradcam_partition.py (DELETE) |  | 68 |
| `scripts/slurm/submit_gradcam_quantile_v2_fold0.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 38 |
| `scripts/slurm/submit_gradcam_quantile_v2_landfill_fold0.sh` | KEEP-CORE | canonical XAI launchers (nearest fold0) |  | 38 |
| `scripts/slurm/submit_gradientshap_extra_folds.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 45 |
| `scripts/slurm/submit_gradientshap_quantile_v2_fold0.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 39 |
| `scripts/slurm/submit_gradientshap_quantile_v2_landfill_fold0.sh` | KEEP-CORE | canonical XAI launchers (nearest fold0) |  | 38 |
| `scripts/slurm/submit_ig_diff_mhw_stratified_fold.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 50 |
| `scripts/slurm/submit_ig_diff_nearest_fold.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 50 |
| `scripts/slurm/submit_ig_extra_folds.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 48 |
| `scripts/slurm/submit_ig_partition.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): SmoothGrad/IG partition (#26) |  | 60 |
| `scripts/slurm/submit_ig_partition_smoothgrad.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): SmoothGrad/IG partition (#26) |  | 82 |
| `scripts/slurm/submit_ig_partition_smoothgrad_pilot.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): SmoothGrad/IG partition (#26) |  | 88 |
| `scripts/slurm/submit_ig_quantile_v2_fold0.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 43 |
| `scripts/slurm/submit_ig_quantile_v2_fold0_diff.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 51 |
| `scripts/slurm/submit_ig_quantile_v2_landfill_fold0.sh` | KEEP-CORE | canonical XAI launchers (nearest fold0) |  | 41 |
| `scripts/slurm/submit_ig_swap_nearestCkpt_zeroInput.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 47 |
| `scripts/slurm/submit_ig_swap_zeroCkpt_nearestInput.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 47 |
| `scripts/slurm/submit_incremental_value_regression.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 31 |
| `scripts/slurm/submit_lead14_landfill_partition.sh` | KEEP-CORE | current-pipeline launchers (configs README): local/remote/lead sweep |  | 44 |
| `scripts/slurm/submit_lead30_landfill_partition.sh` | KEEP-CORE | current-pipeline launchers (configs README): local/remote/lead sweep |  | 44 |
| `scripts/slurm/submit_lead3_landfill_partition.sh` | KEEP-CORE | current-pipeline launchers (configs README): local/remote/lead sweep |  | 44 |
| `scripts/slurm/submit_lead5_landfill_partition.sh` | KEEP-CORE | current-pipeline launchers (configs README): local/remote/lead sweep |  | 44 |
| `scripts/slurm/submit_linear_ceiling_ridge.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 29 |
| `scripts/slurm/submit_local_only_train.sh` | DELETE | launches train_local_only.py with configs/partition/local.yaml (both DELETE/nonexistent) |  | 42 |
| `scripts/slurm/submit_local_only_v2_partition.sh` | KEEP-CORE | current-pipeline launchers (configs README): local/remote/lead sweep |  | 45 |
| `scripts/slurm/submit_mse_v2_partition.sh` | DOUBT | follows full_mse_v2 config (never run on Raven, known_issues #45) |  | 54 |
| `scripts/slurm/submit_mse_v3_fold0.sh` | KEEP-PROVENANCE | launched cited v2 runs (jobs 29417248/405/406/29426208, lr2e5 29433645) |  | 46 |
| `scripts/slurm/submit_occlusion_extra_folds.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): XAI zero-fill/extra folds/swap/diff |  | 39 |
| `scripts/slurm/submit_permutation_importance.sh` | DOUBT | follows permutation_importance.py |  | 44 |
| `scripts/slurm/submit_persistence_recall_baseline.sh` | KEEP-PROVENANCE | reproduces a cited run/number (see script row): eval/analysis ad-hoc |  | 30 |
| `scripts/slurm/submit_plot_loss_comparison.sh` | DOUBT | follows plot_combined/plot_loss group |  | 29 |
| `scripts/slurm/submit_remote_only_v2_partition.sh` | KEEP-CORE | current-pipeline launchers (configs README): local/remote/lead sweep |  | 45 |
| `scripts/slurm/submit_spatial_tbotatm_folds0_1.sh` | OUT-OF-SCOPE | spatial pipeline launcher; not touched |  | 75 |
| `scripts/slurm/submit_sustained_lead_time_remaining.sh` | DOUBT | follows sustained_lead_time group |  | 39 |
| `scripts/slurm/submit_train.sh` | KEEP-CORE | README-documented launcher -> train_partition (remote/local) |  | 43 |
| `scripts/temporal_shuffle_eval.py` | DELETE | reads model.gaussian_nll -> AttributeError (known_issues #25); masked-SSTAtm gnll era | cited in known_issues.md | 175 |
| `scripts/train.py` | DOUBT | documented quick-start (README/CONTRIBUTING); kfold-era trainer superseded by train_partition --mode full; tied to configs/kfold | cited in CLAUDE.md, CONTRIBUTING.md, README.md, audit_plan.md, known_issues.md | 150 |
| `scripts/train_local_only.py` | DELETE | superseded by train_partition.py --mode local_only; its config configs/partition/local.yaml no longer exists (moved to _deprecated_v1) | sbatch: submit_local_only_train.sh; cited in known_issues.md | 174 |
| `scripts/train_partition.py` | KEEP-CORE | canonical training entry point | imported by eval_onset_skill.py, test_masking.py; sbatch: _dryrun_landfill_fold0.sh, _dryrun_lr2e5.sh, submit_gnll_focal_partition.sh +20; cited in CLAUDE.md, CONTRIBUTING.md, audit_plan.md, known_issues.md, narrative.md | 270 |
| `src/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 1 |
| `src/data/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 1 |
| `src/data/datamodule.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by composite_ig.py, gradientshap_quantile_partition.py, train_local_only.py, eval_ig.py +41; cited in CLAUDE.md, audit_plan.md, known_issues.md, narrative.md | 329 |
| `src/data/dataset.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by datamodule.py, ig_simple.py, plot_land_fill_comparison.py; cited in CLAUDE.md, audit_plan.md, data.md, known_issues.md, narrative.md | 509 |
| `src/data/masking.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by eval_event_detection.py, gradcam_partition.py, train_partition.py, ig_signed_partition.py +3; cited in CLAUDE.md, CONTRIBUTING.md, known_issues.md, narrative.md | 30 |
| `src/models/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 1 |
| `src/models/cnn_lstm.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by ig_simple.py, composite_ig.py, gradientshap_quantile_partition.py, train_local_only.py +41; sbatch: submit_gnll_focal_partition.sh; cited in CLAUDE.md, audit_plan.md, known_issues.md, narrative.md | 725 |
| `src/utils/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 1 |
| `src/utils/checkpoints.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by gradientshap_quantile_partition.py, eval_event_detection.py, eval_onset_skill_curve.py, eval_test_metrics_from_best_ckpt.py +19; cited in CLAUDE.md, CONTRIBUTING.md, audit_plan.md, known_issues.md | 95 |
| `src/utils/hobday.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by metrics.py, datamodule.py, dataset.py, composite_bootstrap_ci.py +21; cited in CLAUDE.md, data.md, known_issues.md, narrative.md | 128 |
| `src/utils/metrics.py` | DELETE | 0 importers; duplicate of eval_onset_skill.skill_by_phase (L185). CLAUDE.md table row must be updated in Fase 2 | cited in CLAUDE.md | 39 |
| `src/utils/paths.py` | KEEP-CORE | reachable from train_partition/eval/XAI entry points | imported by hobday.py, composite_bootstrap_ci.py, ig_simple.py, persistence_remote_sst.py +29; cited in audit_plan.md, narrative.md | 39 |
| `src/utils/sampling.py` | KEEP-CORE | required (explicitly) by task; imported by ig/gradcam/gradientshap/occlusion | imported by gradientshap_quantile_partition.py, occlusion_ptho_bot_sanity_check.py, ig_partition_quantile.py, gradcam_quantile_partition.py; cited in audit_plan.md, known_issues.md | 46 |
| `src/xai/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 2 |
| `src/xai/grad_cam.py` | KEEP-CORE | imported by gradcam_quantile_partition.py (canonical XAI) | imported by gradcam_partition.py, run_xai.py, gradcam_quantile_partition.py; cited in CLAUDE.md, audit_plan.md, known_issues.md, narrative.md | 285 |
| `src/xai/integrated_gradients.py` | DELETE | orphan once its 5 importers (all DELETE) go; known_issues #20 marks its wrappers dead; ig_partition_quantile.py does NOT import it. Re-verify cascade before rm | imported by composite_ig.py, eval_ig.py, composite_ig_signed.py, run_xai.py +1; cited in CLAUDE.md, audit_plan.md, known_issues.md | 247 |
| `src/xai/utils.py` | DELETE | 12-line helper; all 10 importers are DELETE (old-era scripts). Re-verify cascade before rm | imported by composite_ig.py, eval_ig.py, diag_attention.py, ensemble_skill.py +6; cited in CLAUDE.md, audit_plan.md | 12 |
| `tests/__init__.py` | KEEP-CORE | package marker | no importer / sbatch / doc reference | 1 |
| `tests/test_checkpoints.py` | KEEP-CORE | MANDATORY per CONTRIBUTING (needs MHW_DATA_FILE/kfold ckpts; skipped otherwise) | cited in CLAUDE.md, CONTRIBUTING.md, known_issues.md | 157 |
| `tests/test_ig_diff_head.py` | KEEP-CORE | guards ig_partition_quantile.py diff head | sbatch: _dryrun_ig_diff_head.sh, submit_ig_diff_nearest_fold.sh, submit_ig_quantile_v2_fold0_diff.sh | 137 |
| `tests/test_masking.py` | KEEP-CORE | MANDATORY per CONTRIBUTING; run in CI | cited in CLAUDE.md, CONTRIBUTING.md, known_issues.md | 84 |
| `tests/test_splits.py` | KEEP-CORE | MANDATORY per CONTRIBUTING; run in CI | cited in CLAUDE.md, CONTRIBUTING.md, known_issues.md | 216 |
| `tests/test_state_feature.py` | KEEP-PENDING | guards state_feature (hybrid model, open decision) | sbatch: submit_gnll_quantile_v2_landfill_hybrid_folds0_1.sh | 140 |

## Hallazgos sospechosos / contradicciones con known_issues (para tu decisión, no actuados)

1. **`results/all_results.csv` no existe en el repo.** `.gitignore` tiene `results/*` (solo `.gitkeep` está trackeado), pero CONTRIBUTING dice "The CSV is versioned" y known_issues/narrative citan filas (`ns_box_enrichment_raw_data_ground_truth`, `event_detection_lead_time_sweep`, ...). Los números "citados" no son trazables a una fila versionada desde este checkout. Propuesta de hallazgo para known_issues.
2. **known_issues #45 / #56.3** piden conservar configs v1 y `_deprecated_v1/` como registro histórico; borrarlos (DOUBT arriba) los contradice. Tampoco se toca nada de `full/`, `full_gnll*/` (todos KEEP-PROVENANCE).
3. **known_issues #20** marca `analyze_integrated_gradients()`/`analyze_gradcam()` como candidatas a borrado "cuando termine el back-check #15"; `src/xai/integrated_gradients.py` va a DELETE por cascada (sus 5 importadores son DELETE). `grad_cam.py` se queda (lo usa `gradcam_quantile_partition.py`; ojo: `analyze_gradcam()` sigue dentro de ese archivo, no se edita aquí).
4. **known_issues #29** deja pendiente arreglar `diag_attention.py` antes de usarlo con checkpoints `attention_only`; el inventario lo pone en DELETE (script de diagnóstico puntual, `NEEDS_RERUN`, importa `run_xai.py`). Hay que ajustar la mención en la Fase 2 solo si lo apruebas.
5. **Docs que citan scripts a borrar:** `CLAUDE.md` (l.66: `causal_triangulation.py`, `thermal_inertia_test.py`; l.93: `src/utils/metrics.py`), `README.md` (`scripts/train.py`), `CONTRIBUTING.md` (`configs/partition/remote.yaml`, que ya no existe: fue movido a `_deprecated_v1/`, doc obsoleto independiente de esta limpieza). `known_issues.md`, `narrative.md` y `audit_plan.md` mencionan decenas de scripts históricos por nombre: propongo NO editarlos (son registro histórico) y que `CLEANUP_LOG.md` sirva de puente (comando de recuperación desde el tag).
6. **CI solo ejecuta `test_splits.py` y `test_masking.py`**; `test_checkpoints.py` (obligatorio en CONTRIBUTING) exige `MHW_DATA_FILE` + checkpoints kfold `TbotAtm_lstmonly_fold*` que probablemente ya no existen. No se puede verificar desde Levante.
7. **Esta sesión corre en DKRZ Levante**, sin acceso a `merged_daily.nc`/`sst_climatology_doy.nc`; ninguna config apunta a rutas alcanzables. Por eso el baseline de la Fase 0 sigue pendiente (ver arriba).
8. `archive/` (19 scripts de póster EGU + `shap_analysis.py`) queda fuera del alcance; ningún archivo de `archive/` importa `src/` (el grafo lo confirma), por lo que no bloquea ningún borrado.
9. `src_spatial/` y `scripts_spatial/` **no importan nada de `src/`** (comprobado por AST y grep): no hay archivos de `src/` retenidos por ellos.
10. **Límite del método:** la evidencia "citado en docs" se basa en coincidencia literal de nombre de archivo/salida; un número citado sin nombre de script ni de artefacto podría escapar. Por eso todo lo dudoso está en DOUBT y no en DELETE.

## Refactors propuestos (no actuados)
- `submit_train.sh` duplica la lógica de `submit_local_only_v2_partition.sh`/`submit_remote_only_v2_partition.sh`.
- Duplicación de `best_ckpt()`/`val_loss()` en `test_checkpoints.py` y scripts (audit Phase 5); migrar los ~24 constructores directos de `CNNLSTMModel` a `load_model_config()` (known_issues #30).
- Caja NS definida con dos lat/lon distintos (`hobday.py` vs `calibrate_mhw_area_threshold.py`, known_issues #57 P2-1).
- Ruta `data_dir`/`output_dir` Raven hardcodeada en cada yaml.
