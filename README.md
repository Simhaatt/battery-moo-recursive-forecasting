# Multi-objective recursive battery-degradation forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22057384.svg)](https://doi.org/10.5281/zenodo.22057384)

Publication-quality code and result artifacts for the manuscript submitted to *Applied Soft Computing*. The study combines knowledge-based battery descriptors, recursive LSTM forecasting, and matched-budget multi-objective search with NSGA-II, NSGA-III, and random search.

## 1. What this repository reproduces

The default workflow performs **Level-1 reproduction**: it reads the deposited raw result artifacts, recomputes reported summaries, regenerates the principal figures and tables, checks the complete 12,600-trial search log, and compares headline values with the final LaTeX manuscript. It does not retrain neural networks.

## 2. Headline finding

The selected NSGA-II model obtained test macro MAPE `1.809732 ± 0.371581%` with 36,418 parameters. The corrected manual model obtained `1.855862 ± 0.226975%` with 486,978 parameters. The mean paired difference was `−0.046130` percentage points (95% CI `−0.383561` to `0.229344`; raw Wilcoxon `p=0.769531`, Holm-adjusted `p=1.0`). Thus, the defensible result is a 92.52% parameter reduction with statistically indistinguishable accuracy—not an accuracy breakthrough.

## 3. Quick start

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python analysis/reproduce_all.py
pytest -q
```

Expected terminal result: `OVERALL: PASS`. Generated material appears in `figures/generated/`, `results/generated/`, and `results/reproduction_check.json`.

## 4. Repository levels

- **Level 1 — artifact reproduction:** seconds to minutes on CPU; no dataset or GPU required.
- **Level 2 — model retraining:** requires the source datasets and is intended for an NVIDIA T4-class GPU or better. The exact historical runners are retained under `notebooks/archive/` and exposed by scripts in `scripts/`.

## 5. Data

The fixed corpus contains 228 cells and 3,980 checkup records. The immutable split is 209/3,461 train cells/records, 9/243 validation cells/records, and 10/276 test cells/records. Common recursive evaluation yields 63 validation windows and 76 test windows. Third-party raw data are not redistributed; see `data/README.md`.

## 6. Inputs and terminology

The nine public descriptors are `k_exp`, `T`, `Q0`, `Re0`, `Rpulse0`, `Cchg`, `Cdis`, `S_SOC`, and `A_type`; the recurrent input has 11 channels after including autoregressive states. Archived files use `Rct0` for `Rpulse0`. It is an empirical pulse-resistance proxy and must not be described as charge-transfer resistance.

## 7. Targets and metrics

The targets are capacity `Q` and ohmic resistance `Re`. Target-wise MAPE is computed first and macro MAPE is their unweighted mean. Resistance values in the evaluated corpus remain away from zero, so the MAPE denominator is well defined. RMSE and R² are retained as secondary metrics.

## 8. Forecasting protocol

Evaluation is strictly recursive: future target values are never fed to the model after the forecast origin. Candidate window length `L` and rollout horizon `H` obey `L + H ≤ 29`, reflecting the shortest usable training trajectory. A corrected common-row protocol ensures every candidate is compared on identical validation rows.

## 9. LSTM architecture

The final LSTM hidden state feeds `Dense(128) → ReLU → Dropout(0.1) → Dense(64) → ReLU → Dense(2)`. Recurrent dropout is 0.2 for multilayer LSTMs and zero for one layer. AdamW uses weight decay `1e-5`; teacher/rollout batch sizes are 512/128; gradients are clipped at 1.0; rollout fine-tuning uses `0.3 ×` the candidate learning rate.

## 10. Search space

The decision vector is `[L, hidden, layers, H, learning_rate]`: `L∈{10,15,20}`, hidden width `∈{64,96,128,160,192}`, layers `∈{1,2,3}`, `H∈{3,5,8,10,15}`, and learning rate log-uniform on `[1e-4,2e-3]`. The corrected manual configuration is `[20,192,2,8,0.001]`.

## 11. Optimizer protocol

NSGA-II, NSGA-III, and random search each use 15 matched runs, population 28, and 280 unique evaluations per run: 4,200 evaluations per method and 12,600 total. Run seeds are `20270001…20270015`; matched training seeds are `41001…41015`. Objectives are validation macro MAPE, `log10(parameters)`, and `log10(latency_ms)`.

## 12. Search analysis

Objectives are globally normalized with the deposited extrema and hypervolume reference `[1.1,1.1,1.1]`. Mean final hypervolumes are 1.197236 (NSGA-II), 1.157911 (NSGA-III), and 1.141019 (random). The pooled nondominated set contains 28 points: 21/2/5 contributed by NSGA-II/NSGA-III/random.

## 13. Budget parity

All 15 NSGA-II runs, 10 NSGA-III runs, and 8 random runs reached the defined parity threshold. Among reaching runs, mean evaluations were 71.2, 125.1, and 166.625. The reported 2.34× factor is therefore explicitly conditional on reaching runs.

## 14. Noise, rescreening, and final evaluation

The noise panel contains eight configurations × ten seeds = 80 fits. Search-stage seed standard deviations are roughly 0.21–0.52 percentage points. Five candidates per method were rescreened at seeds 42, 52, and 62 (45 trainings), followed by a method-specific resource-aware knee. Four finalists were then evaluated with ten matched seeds (40 trainings).

## 15. Auxiliary experiments

The clean stress ablation used ten matched seeds and found the descriptor worsened macro MAPE by `+0.622698` points in all ten pairs (`p=0.001953`). The Mamba baseline obtained `2.945251 ± 0.208178%`; the TCN obtained `2.088285 ± 0.307598%`. Grouped-fold statistics and PINN artifacts are included under `results/`.

## 16. External transfer audit

Forty Kirkaldy 21700 cells contributed 246 later-window evaluation points. The target-only LSTM reached mean SOH MAPE 4.233551%, while frozen-source-representation/head-only adaptation reached 11.720338%. The cell-level difference was `+6.987501` points (`p=1.644366×10⁻9`), so these results do **not** support beneficial representation transfer. SOH persistence and parametric fade controls are included.

## 17. Reproducibility contract

`manuscript/final_manuscript.tex` is the expected-output specification. `configs/expected_values.yaml` maps its numerical claims to authoritative artifacts. `analysis/reproduce_all.py` writes a machine-readable and human-readable comparison. Any mismatch beyond `1e-6` fails the command.

## 18. Testing

`pytest -q` checks search cardinality, method/run budgets, constraints, fixed split cardinality, expected manuscript values, and the documented transfer discrepancy. Tests use artifacts only and run on CPU.

## 19. Provenance and notebooks

Original authored and executed notebooks are preserved unchanged in `notebooks/archive/`. `SOURCE_INVENTORY.csv` records path, byte size, SHA-256 hash, and classification for 587 source-package files. The clean analysis code lives in `src/battery_moo/`; notebooks are historical provenance, not the primary API.

## 20. Citation, release, and availability

Release `v1.0.0` is archived at [Zenodo](https://doi.org/10.5281/zenodo.22057384). Cite using `CITATION.cff`. Code is MIT licensed; third-party datasets retain their original terms.

## Repository map

| Path | Purpose |
|---|---|
| `analysis/reproduce_all.py` | one-command artifact reproduction |
| `scripts/preprocess_phase2_split.py` | corrected legacy PINN phase-two preprocessing and 180/27/21 cell split |
| `configs/` | immutable data, model, search, and manuscript contracts |
| `data/splits/` | fixed cell split and grouped-fold assignments |
| `results/` | authoritative result artifacts and generated tables |
| `figures/generated/` | regenerated figures |
| `manuscript/` | final LaTeX numerical specification |
| `notebooks/archive/` | immutable historical notebooks and runners |
| `tests/` | automated integrity and manuscript checks |

## Limitations

The repository cannot remove uncertainties inherited from the data-export provenance. In particular, historical derivations of `k_exp`, SOC/type descriptors, and activation-energy regularization are incompletely recorded. Full GPU reruns can also differ slightly across hardware/library versions. These limitations are enumerated in `KNOWN_ISSUES.md` and are not concealed by the passing artifact-reproduction check.
