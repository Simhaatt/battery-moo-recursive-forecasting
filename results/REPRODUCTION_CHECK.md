# Reproduction check

Overall status: **PASS**

| Expected manuscript value | Reproduced value | Status |
|---|---:|:---:|
| NSGA-II HV (1.197236) | 1.197236 | PASS |
| NSGA-III HV (1.157911) | 1.157911 | PASS |
| Random HV (1.141019) | 1.141019 | PASS |
| Pooled Pareto count (28.000000) | 28.000000 | PASS |
| NSGA-II pooled points (21.000000) | 21.000000 | PASS |
| NSGA-III pooled points (2.000000) | 2.000000 | PASS |
| Random pooled points (5.000000) | 5.000000 | PASS |
| NSGA-II evaluations to parity (71.200000) | 71.200000 | PASS |
| NSGA-III evaluations to parity (125.100000) | 125.100000 | PASS |
| Random evaluations to parity (166.625000) | 166.625000 | PASS |
| NSGA-II runs reaching parity (15.000000) | 15.000000 | PASS |
| NSGA-III runs reaching parity (10.000000) | 10.000000 | PASS |
| Random runs reaching parity (8.000000) | 8.000000 | PASS |
| Final NSGA-II macro MAPE (1.809732) | 1.809732 | PASS |
| Final NSGA-II macro MAPE SD (0.371581) | 0.371581 | PASS |
| Final NSGA-III macro MAPE (1.871584) | 1.871584 | PASS |
| Final manual macro MAPE (1.855862) | 1.855862 | PASS |
| NSGA-II minus manual MAPE (-0.046130) | -0.046130 | PASS |
| NSGA-II minus manual CI low (-0.383561) | -0.383561 | PASS |
| NSGA-II minus manual CI high (0.229344) | 0.229344 | PASS |
| NSGA-II versus manual Wilcoxon p (0.769531) | 0.769531 | PASS |
| Stress descriptor delta (0.622698) | 0.622698 | PASS |
| Stress descriptor Wilcoxon p (0.001953) | 0.001953 | PASS |
| Transfer minus target-only delta (6.987501) | 6.987501 | PASS |
| Transfer cell-level Wilcoxon p (0.000000) | 0.000000 | PASS |
| External evaluation cells (40.000000) | 40.000000 | PASS |
| External evaluation points (246.000000) | 246.000000 | PASS |
| Parameter reduction percent (92.500000) | 92.521633 | PASS |

## Search-integrity checks

- [x] row_count_12600
- [x] three_methods
- [x] fifteen_runs_each
- [x] 280_evaluations_per_run
- [x] candidate_ids_unique_within_run
- [x] allowed_window_lengths
- [x] allowed_hidden_sizes
- [x] allowed_layer_counts
- [x] allowed_rollout_horizons
- [x] learning_rate_bounds
- [x] window_horizon_constraint
- [x] finite_objectives
- [x] positive_objectives
