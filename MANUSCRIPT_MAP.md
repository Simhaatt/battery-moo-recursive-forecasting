# Manuscript-to-artifact map

The final numerical specification is `manuscript/final_manuscript.tex`. Labels below identify the primary source artifact and reproduction output. Line numbers are intentionally avoided because LaTeX editing changes them.

| Manuscript item | Authoritative input | Reproduced output/code |
|---|---|---|
| `tab:primary_dataset`, `tab:data_split` | `data/splits/fixed_cell_split.csv`; archived processed manifest | `tests/test_splits.py` |
| `tab:input_features` | `configs/data.yaml` | configuration contract |
| `fig:rollout_protocol` | archived training runners | `configs/model.yaml`, `configs/search.yaml` |
| `tab:lstm_architecture` | final runner and manifests | `configs/model.yaml` |
| `tab:search_space` | `results/optimizer/all_optimizer_trials.csv` | search integrity tests |
| `tab:evolutionary_comparison` | archived optimizer source | `notebooks/archive/10_asc_optimizer_study/` |
| `tab:search_budget` | complete search log | `results/generated/optimizer_final_quality.csv` |
| `tab:training_schedules` | archived manifests/runners | `configs/search.yaml` |
| `tab:noise_configurations`, `tab:noise_results` | `results/noise/noise_floor_*.csv` | artifact tables |
| `tab:final_front_quality`, `tab:front_quality_stats` | `hv_igd_by_run_and_evaluation.csv`, optimizer tests | `optimizer_final_quality.csv`; contract check |
| `fig:optimizer_convergence` | `hv_igd_convergence_summary.csv` | `figures/generated/optimizer_convergence.png` |
| `tab:budget_parity`, `fig:budget_to_parity` | `results/audit/budget_to_parity_*.csv` | generated CSV and PNG |
| `tab:front_structure`, `tab:pooled_front` | `pooled_pareto_tidy.csv` | `figures/generated/pooled_pareto.png` |
| finalist selection/rescreening tables | `candidate_rescreen_*.csv`, `selected_finalists.csv` | deposited artifacts |
| final configuration and accuracy tables | `final_ten_seed_raw.csv`, `final_ten_seed_summary.csv` | `final_model_comparison.csv`; contract check |
| final paired statistics | `final_paired_tests_holm.csv` | contract check |
| feature ablation table/figure | `phase1e_*ablation*csv` | `figures/generated/feature_ablation.png` |
| stress ablation table | `stress_ablation_seed_metrics.csv`, summary | generated CSV/PNG; contract check |
| modern sequence baselines | `tcn_*.csv`, `mamba_*.csv` | deposited summaries |
| PINN comparison | `phase4d/5/6/7` extracted summary CSVs | deposited baseline artifacts |
| grouped-fold ranks/statistics | `phase1_grouped_cv_metrics_raw.csv`, `statistical_*` | deposited exact tests |
| external transfer protocol/table/figure | `results/transfer/transfer_*` | generated CSV/PNG; contract check |

`configs/expected_values.yaml` is the executable mapping for headline values. Every row is resolved by `src/battery_moo/reproduce.py` and written to `results/reproduction_check.json`.

