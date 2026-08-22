"""Build the evidence-only ASC methods/results dossier from verified artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROUND2 = BASE / "07_round2_revision" / "outputs"
AUDIT = BASE / "06_manuscript_context" / "submission_audit_csv"
FINAL = BASE / "09_asc_rewrite_evidence" / "final_results"
OPT = BASE / "10_asc_optimizer_study" / "complete_results_audit_20260822"
OPT_RUNS = BASE / "10_asc_optimizer_study" / "received_results_audit_20260821"
OUT = BASE / "ASC_COMPLETE_METHODS_RESULTS_DOSSIER_20260822.md"


def fmt_frame(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return result


def table(frame: pd.DataFrame, digits: int = 6) -> str:
    formatted = fmt_frame(frame, digits).astype(str)
    columns = [str(column).replace("|", "\\|") for column in formatted.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in formatted.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(BASE / relative)


parts: list[str] = []


def add(text: str = "") -> None:
    parts.append(text.strip("\n") + "\n")


add("""# Applied Soft Computing Battery Forecasting Study — Complete Methods and Results Dossier

**Purpose:** evidence-only source file for writing the paper; this is not manuscript prose.  
**Target journal:** *Applied Soft Computing* (Elsevier).  
**Evidence cutoff:** 22 August 2026.  
**Status:** all planned computation is complete.  
**Authoritative optimizer archive:** `C:\\Users\\hp\\Downloads\\asc_optimizer_complete_results.zip`.  
**Important supersession rule:** the 15-run NSGA-II/NSGA-III/random study and ten-seed final evaluation in this dossier supersede the earlier 32-evaluation/five-seed NSGA-II–random pilot.

---

## 1. Completed experimental program

| Component | Final scope | Status |
|---|---:|---|
| Original model-family benchmark | 9 core model rows plus tree ensembles and PINN-feature hybrids | Complete |
| Grouped cell-wise cross-validation | 7 variants × 5 folds | Complete |
| Stress-descriptor ablation | 2 matched variants × 10 seeds | Complete |
| Modern sequence baselines | TCN and official Mamba, 5 seeds each | Complete |
| External transfer audit | 40 cells, 246 later-half points, 5 neural seeds and 4 deterministic controls | Complete |
| Optimizer noise floor | 8 fixed configurations × 10 seeds = 80 trainings | Complete |
| NSGA-II | 15 runs × 280 unique configurations = 4,200 trainings | Complete |
| NSGA-III | 15 runs × 280 unique configurations = 4,200 trainings | Complete |
| Matched random search | 15 runs × 280 unique configurations = 4,200 trainings | Complete |
| Optimizer candidate re-screen | 15 configurations × 3 seeds = 45 trainings | Complete |
| Final optimizer confirmation | manual + 3 selected configurations × 10 seeds = 40 trainings | Complete |
| Final optimizer predictions | 40 runs × 76 test rows = 3,040 predictions | Complete |

The final optimizer study contains **12,600 search evaluations**, **45 re-screen evaluations**, and **40 final confirmation evaluations**. Its raw tables contain no missing objective values, no duplicate run/evaluation identifiers, no infeasible `L + H > 29` candidates, and exactly 280 evaluations in every one of the 45 optimizer runs.

---

## 2. Primary dataset and variables

### 2.1 Luh–Blank source dataset

- Commercial NMC/C–SiO 18650 cells: **228**.
- Processed checkup records: **3,980**.
- Nominal temperatures: **0, 10, 25, and 40 °C**.
- Charge-rate range: **0–1.67 C**.
- Discharge-rate range: **0–1 C**.
- Capacity range in the processed source table: **1.512790–2.970551 Ah**.
- Electrolyte-resistance range: **12.990333–28.150000 mΩ**.
- Beginning-of-life capacity range: **2.926306–2.970551 Ah**.
- Beginning-of-life electrolyte-resistance range: **12.990333–19.766333 mΩ**.
- Beginning-of-life charge-transfer-resistance range: **0.020467–11.714533** in the archived processed units.
- Ageing index range: **0–63,884**.
- Resistance targets remain safely away from zero; MAPE is therefore numerically well-defined.

### 2.2 Full nine exogenous features

1. Experimental ageing index `k_exp`.
2. Temperature `T`.
3. Beginning-of-life capacity `Q0`.
4. Beginning-of-life electrolyte resistance `Re0`.
5. Beginning-of-life charge-transfer resistance `Rct0`.
6. Charge C-rate.
7. Discharge C-rate.
8. SOC window.
9. Ageing/protocol type.

Sequence models append measured `Q` and `Re` history during training. During recursive deployment, unknown future measured history is replaced with the model's preceding predictions.

### 2.3 Targets

- Capacity `Q` in Ah.
- Electrolyte resistance `Re` in mΩ.
- Repeated `Rct` is not predicted because of unstable/noisy repeated values; only `Rct0` is used as a descriptor.

### 2.4 Source splits

Original broad benchmark split:

- 180 training cells.
- 27 validation cells.
- 21 held-out test cells.

Final sequence-eligible split:

- 209 training cells containing 3,461 rows.
- 9 validation cells containing 243 rows.
- 10 test cells containing 276 rows.
- Common validation rollout points: 63.
- Common held-out test points: 76.
- Common scoring starts at trajectory row index 20 for every sequence length.

No sequence crosses a cell boundary. All scalers are fitted on training cells only. Validation controls early stopping/model selection; test data are used only after final configuration selection.

---

## 3. Preprocessing, sequence construction, and leakage control

- Records are grouped by `cell_id` and ordered by `k_exp`.
- Exogenous and target scalers are `StandardScaler` objects fitted only on training data.
- A one-step sample at index `i` is `records[i-L:i] → [Q(i), Re(i)]`.
- Sequence channels are retained exogenous features followed by Q/Re history.
- Recursive inference copies each trajectory into a working buffer, begins scoring at row 20, predicts Q/Re, inserts those predictions into the buffer, and continues without future measured targets.
- Physical-unit metrics are computed only after inverse transformation.
- Python, NumPy, PyTorch CPU, and PyTorch CUDA seeds are fixed; shuffled data loaders use explicitly seeded generators.

The longest training trajectory contains 29 records. Rollout segment construction therefore requires:

```text
L + H <= 29
```

The old `L=20, H=10` manual setting generated zero rollout segments. The corrected manual setting is `L=20, H=8`.

---

## 4. Metrics and uncertainty

For target `j`:

```text
MAPE_j = (100/N) Σ_i |(y_ij - yhat_ij) / y_ij|
RMSE_j = sqrt[(1/N) Σ_i (y_ij - yhat_ij)^2]
R2_j = 1 - Σ_i(y_ij-yhat_ij)^2 / Σ_i(y_ij-mean(y_j))^2
Macro MAPE = (MAPE_Q + MAPE_Re) / 2
```

- Seed uncertainty is the sample standard deviation across independently trained seeds.
- Final optimizer summaries additionally report 10,000-resample bootstrap 95% confidence intervals of seed means.
- Deterministic external controls have no seed standard deviation.
- Transfer cell-effect intervals use 10,000 bootstrap resamples of 40 paired cell differences.
- Two-sided Wilcoxon signed-rank tests are used for paired seed/cell comparisons.
- Multiple optimizer pairwise tests use Holm correction.
- Hypervolume is maximized; inverted generational distance (IGD) is minimized.

---

## 5. Core rollout LSTM

### 5.1 Architecture

Candidate-dependent batch-first PyTorch LSTM followed by:

```text
last recurrent state
→ Linear(hidden, 128)
→ ReLU
→ Dropout(0.1)
→ Linear(128, 64)
→ ReLU
→ Linear(64, 2)
```

- LSTM dropout: 0.2 for more than one recurrent layer; 0 for one layer.
- Outputs: scaled Q and Re.

### 5.2 Teacher pretraining

- AdamW; candidate learning rate `η`; weight decay `1e-5`.
- Batch size 512.
- Smooth-L1 loss in scaled target space.
- Global gradient-norm clipping at 1.0.
- Validation macro MAPE early stopping.
- Best validation checkpoint restored.

### 5.3 Rollout segments and fine-tuning

- Each segment: initial L-row window, H future exogenous rows, H Q/Re targets.
- Segment stride: 2.
- At every step, predicted Q/Re are concatenated with known exogenous inputs and fed back.
- AdamW rollout learning rate `0.30η`; weight decay `1e-5`.
- Batch size 128.
- Per-step MSE in scaled target space, averaged over H steps.
- Gradient clipping 1.0.
- Early stopping on common-window validation rollout macro MAPE.
- Best rollout checkpoint restored.

---

## 6. Final repeated multi-objective optimizer protocol

### 6.1 Search variables

| Variable | Domain |
|---|---|
| Sequence length L | {10, 15, 20} |
| Hidden width | {64, 96, 128, 160, 192} |
| LSTM layers | {1, 2, 3} |
| Rollout horizon H | {3, 5, 8, 10, 15}, with L+H≤29 |
| Learning rate | log-uniform from 1e-4 to 2e-3 |

### 6.2 Objectives

Minimize simultaneously:

1. Common-window validation rollout macro MAPE.
2. Trainable parameter count.
3. Median batch-one forward-pass latency.

Search latency is not full-trajectory latency. It uses one batch-one input window, 20 warm-ups, five timed blocks, and the median block time. Final trajectory latency includes all recursive calls and is divided by the number of test trajectories.

### 6.3 Matched budget

- 15 independent runs per method.
- Population/reference size: 28.
- Ten generations counting the initial population (`0–9`).
- 28 evaluations per generation.
- 280 unique trained configurations per run.
- 4,200 evaluations per method.
- 12,600 evaluations across NSGA-II, NSGA-III, and random search.
- Run RNG seeds shared across methods: `20270001–20270015`.
- Candidate-training seeds shared run-wise across methods: `41001–41015`.
- Identical initial populations within each matched run; later offspring diverge by selection method.

### 6.4 Search training schedule

| Stage | Teacher max epochs | Teacher patience | Rollout max epochs | Rollout patience |
|---|---:|---:|---:|---:|
| Optimizer search | 80 | 18 | 15 | 5 |
| Candidate re-screen | 160 | 30 | 30 | 10 |
| Final confirmation | 220 | 40 | 40 | 15 |

### 6.5 Variation and feasibility repair

- Each child field is inherited independently from either parent with probability 0.5.
- Discrete-field mutation probability: 0.25 per field.
- Learning-rate mutation probability: 0.25; Gaussian perturbation in log10 space with SD 0.28, clipped to the search bounds.
- If no field mutates, one randomly selected field is forced to mutate.
- Infeasible `L+H>29` children are repaired using the closest allowed feasible H; ties prefer the larger H.
- Duplicate candidates are regenerated and do not consume the unique-evaluation budget.

### 6.6 NSGA-II

- Binary tournaments use Pareto rank first and crowding distance second.
- Parent and offspring populations are combined.
- Fast nondominated sorting fills complete fronts.
- A partial final front is truncated by descending crowding distance.

### 6.7 NSGA-III

- Three-objective Das–Dennis reference directions with division parameter `H=6`.
- Number of reference directions: 28.
- Objective values are min–max normalized within the combined population for association.
- Candidates associate with the reference direction having minimum perpendicular distance.
- Complete nondominated fronts are retained; the split front is filled using least-occupied reference niches, choosing the closest point for an empty niche and a random associated point otherwise.

### 6.8 Random search

- Uniform sampling over the same discrete domains and log-uniform learning-rate interval.
- Same feasibility rule, training schedule, run-wise training seeds, evaluation budget, objectives, and latency routine.
- Every random candidate is unique within its run.

### 6.9 Global indicator calculation

- Pool all 12,600 evaluations.
- Transform objectives to `[MAPE, log10(parameters), log10(latency)]`.
- Global lower bounds: `[0.7308796223, 4.5613160916, -0.4342352697]`.
- Global upper bounds: `[8.8913714513, 5.8939979806, 0.0246944637]`.
- Normalize each transformed objective to [0,1] using those global bounds.
- Hypervolume reference point: `(1.1, 1.1, 1.1)`.
- IGD reference set: pooled nondominated union, 28 points.
- Indicators are calculated after every 28 evaluations: 28, 56, ..., 280.
- Final indicator omnibus test: Kruskal–Wallis.
- Pairwise tests: two-sided Mann–Whitney U with Holm correction.
- Effect size: Vargha–Delaney A12, reported as probability that the first method has the greater numerical indicator.

### 6.10 Candidate re-screen and final selection

- Per optimizer, pooled nondominated candidates are ordered by normalized distance to the ideal; five unique candidates are retained.
- Re-screen seeds: 42, 52, 62.
- Re-screened mean MAPE, log-parameters, and log-latency are normalized across the 15 candidates.
- Euclidean robust-knee distance determines one finalist per method.
- Final seeds: 42, 52, 62, 72, 82, 92, 102, 112, 122, 132.
- Final comparison: corrected manual LSTM plus one NSGA-II, one NSGA-III, and one random finalist.
- All final runs use the identical 76-row, ten-cell test panel.
""")


# Noise-floor results
noise = pd.read_csv(OPT_RUNS / "nsga2" / "noise_floor_summary.csv")
add("""---

## 7. Optimizer noise-floor experiment

Eight fixed configurations were trained with the search schedule over ten seeds (`42–132` in increments of ten). This quantifies stochastic training variation and measurement noise before interpreting one-off search winners.
""")
add(table(noise, 6))
add("""
Key facts:

- MAPE SD across fixed configurations ranges from approximately 0.21 to 0.52 percentage points.
- Latency SD is much smaller, approximately 0.001–0.026 ms in this panel.
- Single-search best MAPE values are therefore not final evidence; re-screening and ten-seed confirmation are mandatory.
""")


# Search run audit and best MAPE
trials = pd.read_csv(OPT / "all_optimizer_trials.csv")
run_best = trials.loc[trials.groupby(["method", "run"])["validation_macro_MAPE"].idxmin(),
                      ["method", "run", "evaluation", "generation", "candidate_id", "validation_macro_MAPE", "parameters", "latency_ms"]]
indicators = pd.read_csv(OPT / "hv_igd_by_run_and_evaluation.csv")
final_indicators = indicators[indicators["evaluations"].eq(280)][["method", "run", "hypervolume", "IGD"]]
run_ledger = run_best.merge(final_indicators, on=["method", "run"]).sort_values(["method", "run"])
add("""---

## 8. Optimizer search results

### 8.1 Completeness and uniqueness

| Method | Runs | Evaluations/run | Total rows | Unique candidate IDs across method |
|---|---:|---:|---:|---:|
| NSGA-II | 15 | 280 | 4,200 | 4,112 |
| NSGA-III | 15 | 280 | 4,200 | 4,098 |
| Random | 15 | 280 | 4,200 | 4,200 |

Repeated IDs across different optimizer runs are allowed because runs are independent. Within every run, all 280 IDs are unique.

### 8.2 Per-run best validation candidate and final indicators
""")
add(table(run_ledger, 6))

indicator_summary = final_indicators.groupby("method", as_index=False).agg(
    hypervolume_mean=("hypervolume", "mean"), hypervolume_std=("hypervolume", "std"),
    IGD_mean=("IGD", "mean"), IGD_std=("IGD", "std"))
best_summary = run_best.groupby("method", as_index=False).agg(
    best_MAPE_mean=("validation_macro_MAPE", "mean"), best_MAPE_std=("validation_macro_MAPE", "std"),
    best_MAPE_median=("validation_macro_MAPE", "median"), best_MAPE_min=("validation_macro_MAPE", "min"),
    best_MAPE_max=("validation_macro_MAPE", "max"))
add("### 8.3 Final 280-evaluation indicator summary")
add(table(indicator_summary, 6))
add("### 8.4 Per-run best-MAPE distribution")
add(table(best_summary, 6))

conv = pd.read_csv(OPT / "hv_igd_convergence_summary.csv")
add("### 8.5 Complete convergence summary at 28-evaluation intervals")
add(table(conv, 6))

tests = pd.read_csv(OPT / "optimizer_statistical_tests.csv")
add("### 8.6 Optimizer statistical tests")
add(table(tests, 9))
add("""
Interpretation constraints:

- Overall method differences are significant for hypervolume (`p=2.1121e-6`) and IGD (`p=0.0003658`).
- NSGA-II exceeds NSGA-III in hypervolume after Holm correction (`p=0.0002713`, A12=0.9111).
- NSGA-II exceeds random search in hypervolume (`p=1.2430e-5`, A12=0.9956).
- NSGA-II has lower/better IGD than random search (`p=0.0004069`; A12 for first-greater is 0.0889, hence the first method is usually lower).
- NSGA-III versus random is not significant after Holm correction for either indicator.
- Search-level Pareto efficiency supports NSGA-II; it does not by itself prove lower held-out test MAPE.
""")


rescreen = pd.read_csv(OPT / "candidate_rescreen_summary.csv").sort_values(["method", "robust_knee_distance"])
finalists = pd.read_csv(OPT / "selected_finalists.csv")
add("""---

## 9. Re-screened candidate ledger

All 15 candidates below were trained under the longer re-screen schedule using seeds 42, 52, and 62.
""")
add(table(rescreen, 6))
add("### 9.1 Selected finalists")
add(table(finalists, 9))


final_raw = pd.read_csv(OPT / "final_ten_seed_raw.csv")
final_summary = pd.read_csv(OPT / "final_ten_seed_summary.csv")
final_tests = pd.read_csv(OPT / "final_paired_tests_holm.csv")
add("""---

## 10. Final ten-seed optimizer confirmation

### 10.1 Final configurations

| Configuration | L | Hidden | Layers | H | Learning rate | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Manual | 20 | 192 | 2 | 8 | 0.001000000 | 486,978 |
| NSGA-II | 15 | 64 | 1 | 8 | 0.001215422 | 36,418 |
| NSGA-III | 20 | 64 | 1 | 8 | 0.000868730 | 36,418 |
| Random | 15 | 64 | 1 | 5 | 0.000739324 | 36,418 |

### 10.2 Complete per-seed accuracy and fit ledger
""")
accuracy_cols = ["configuration", "seed", "validation_macro_MAPE", "Q_MAPE", "Q_RMSE", "Q_R2",
                 "Re_MAPE", "Re_RMSE", "Re_R2", "macro_MAPE"]
add(table(final_raw[accuracy_cols], 6))
add("### 10.3 Complete per-seed compute ledger")
compute_cols = ["configuration", "seed", "parameters", "latency_ms", "train_seconds", "inference_ms_per_trajectory"]
add(table(final_raw[compute_cols], 6))

acc_summary_cols = ["configuration", "n", "macro_MAPE_mean", "macro_MAPE_std", "macro_MAPE_ci95_low", "macro_MAPE_ci95_high",
                    "Q_MAPE_mean", "Q_MAPE_std", "Q_MAPE_ci95_low", "Q_MAPE_ci95_high",
                    "Re_MAPE_mean", "Re_MAPE_std", "Re_MAPE_ci95_low", "Re_MAPE_ci95_high"]
compute_summary_cols = ["configuration", "parameters_mean", "latency_ms_mean", "latency_ms_std", "latency_ms_ci95_low", "latency_ms_ci95_high",
                        "train_seconds_mean", "train_seconds_std", "train_seconds_ci95_low", "train_seconds_ci95_high"]
add("### 10.4 Accuracy summary with bootstrap confidence intervals")
add(table(final_summary[acc_summary_cols], 6))
add("### 10.5 Compute summary with bootstrap confidence intervals")
add(table(final_summary[compute_summary_cols], 6))
add("### 10.6 Paired final tests against manual tuning")
add(table(final_tests, 9))

manual = final_summary.set_index("configuration").loc["Manual"]
resource_rows = []
for name in ["NSGA-II", "NSGA-III", "Random"]:
    row = final_summary.set_index("configuration").loc[name]
    resource_rows.append({
        "configuration": name,
        "macro_difference_pp": row.macro_MAPE_mean - manual.macro_MAPE_mean,
        "relative_macro_change_pct": 100 * (row.macro_MAPE_mean / manual.macro_MAPE_mean - 1),
        "parameter_reduction_pct": 100 * (1 - row.parameters_mean / manual.parameters_mean),
        "forward_latency_reduction_pct": 100 * (1 - row.latency_ms_mean / manual.latency_ms_mean),
        "training_time_reduction_pct": 100 * (1 - row.train_seconds_mean / manual.train_seconds_mean),
    })
resource = pd.DataFrame(resource_rows)
infer_means = final_raw.groupby("configuration").inference_ms_per_trajectory.mean()
resource["trajectory_latency_reduction_pct"] = [100*(1-infer_means[n]/infer_means["Manual"]) for n in resource.configuration]
add("### 10.7 Derived changes relative to manual")
add(table(resource, 6))
add("""
Final-result constraints:

- NSGA-II has the lowest numerical final macro MAPE: `1.809732 ± 0.371581%` versus manual `1.855862 ± 0.226975%`.
- The NSGA-II–manual macro difference is `-0.046130` percentage points and is not significant (`Holm p=1.0`).
- NSGA-II uses 92.52% fewer parameters, 38.93% lower batch-one forward latency, about 62.22% less training time, and about 35.31% lower full-trajectory latency.
- No evolutionary finalist establishes a significant held-out accuracy improvement.
- The bootstrap CI for NSGA-II minus manual macro MAPE, `[-0.383561, 0.229344]`, is not contained in the prespecified ±0.20-point practical-equivalence interval; formal equivalence is not established.
- Random minus manual macro MAPE has CI `[-0.186130, 0.148552]`, contained within ±0.20 points in the implemented bootstrap criterion.
- Defensible claim: NSGA-II significantly improves search-level Pareto quality and identifies a far smaller/faster model with statistically unresolved, numerically similar test error.
""")


# TCN and Mamba
tcn = pd.read_csv(ROUND2 / "tcn_seed_metrics.csv")
mamba = pd.read_csv(FINAL / "mamba" / "mamba_seed_metrics.csv")
add("""---

## 11. Modern sequence baselines

### 11.1 TCN architecture and protocol

- Eleven channels: nine exogenous features plus Q/Re history.
- 1D projection convolution: 11→96 channels, kernel 1.
- Four residual causal blocks with dilations 1, 2, 4, 8.
- Each block: kernel-3 convolution, causal trimming, batch normalization, ReLU, dropout 0.1, residual addition.
- Head: `Linear(96,64) → ReLU → Dropout(0.1) → Linear(64,2)`.
- Parameters: 119,234.
- Sequence length: 20.
- Same 76-row recursive test panel and seeds 42–82.

### 11.2 TCN per-seed results
""")
add(table(tcn, 6))
add("""
TCN summary: macro MAPE `2.088285 ± 0.307598%`, Q-MAPE `3.107962 ± 0.361827%`, Re-MAPE `1.068608 ± 0.337508%`, CPU train time `26.775 ± 7.388 s`, CPU trajectory latency `17.080 ± 1.527 ms`.

### 11.3 Official Mamba architecture and protocol

- Official `mamba_ssm.Mamba`, version 2.3.2.post1.
- Input projection 11→64.
- Two residual Mamba blocks with pre-block layer normalization.
- State dimension 16; convolution width 4; expansion factor 2.
- Final layer norm and two-output head from the final time position.
- Parameters: 70,722.
- Sequence length 15; rollout horizon 10.
- AdamW learning rate `7e-4`, weight decay `1e-5`.
- Teacher batch 256; rollout batch 128.
- Teacher maximum 220/patience 40; rollout maximum 40/patience 15.
- Tesla T4; seeds 42–82; 76 test rows per seed.

### 11.4 Mamba per-seed results
""")
add(table(mamba, 6))
add("Mamba summary: macro MAPE `2.945251 ± 0.208178%`, Q-MAPE `4.392401 ± 0.471298%`, Re-MAPE `1.498101 ± 0.166333%`, train time `12.038 s`, trajectory latency `13.925 ms`.")


# Stress
stress_summary = pd.read_csv(ROUND2 / "stress_ablation_summary.csv")
stress_raw = pd.read_csv(ROUND2 / "stress_ablation_seed_metrics.csv")
stress_pivot = stress_raw.pivot(index="seed", columns="variant", values=["macro_MAPE", "Q_MAPE", "Re_MAPE"]).reset_index()
stress_pivot.columns = ["seed"] + [f"{metric}_{variant}" for metric, variant in stress_pivot.columns[1:]]
add("""---

## 12. Clean stress-descriptor ablation

### 12.1 Protocol

- Base exogenous input: `k_exp, Re0, Rct0, Q0`.
- Comparison adds only scalar `stress`.
- Sequence also contains Q/Re history: six channels without stress and seven with stress.
- Ten matched seeds: 42, 52, 62, 72, 82, 92, 102, 112, 122, 132.
- The larger model is initialized first; the smaller first-layer input matrix is obtained by deleting the stress column and all same-shaped parameters are copied exactly.
- Same data order, preprocessing, training protocol, and 76-row recursive evaluation.

### 12.2 Summary
""")
add(table(stress_summary, 6))
add("### 12.3 Per-seed target-wise ledger")
add(table(stress_pivot, 6))
add("""
- Mean macro deterioration after adding stress: **+0.622698 percentage points**.
- Paired Wilcoxon `p=0.001953`.
- Macro MAPE worsens for all ten matched seeds.
- The implemented descriptor is not an accuracy contribution; it may only be described as a diagnostic knowledge variable.
""")


# Grouped folds
folds = pd.read_csv(AUDIT / "grouped_cv_per_fold_macro_mape.csv")
ranks = pd.read_csv(ROUND2 / "statistical_mean_ranks.csv")
pairwise = pd.read_csv(ROUND2 / "statistical_pairwise_wilcoxon_holm.csv")
omnibus = json.loads((ROUND2 / "statistical_omnibus.json").read_text())
add("""---

## 13. Five-fold grouped-cell evidence

### 13.1 Per-fold macro MAPE
""")
add(table(folds, 6))
add("### 13.2 Mean ranks")
add(table(ranks, 6))
add(f"""### 13.3 Omnibus result

- Friedman χ² = `{omnibus['friedman_chi_square']:.6f}`.
- Friedman p = `{omnibus['friedman_p']:.9f}`.
- Folds = {omnibus['n_folds']}; models = {omnibus['n_models']}.
- Nemenyi critical difference at α=0.05 = `{omnibus['nemenyi_critical_difference_alpha_0_05']:.6f}` ranks.
- Minimum non-zero exact two-sided Wilcoxon p with five folds is 0.0625.
- No pairwise comparison survives Holm correction.

### 13.4 All Holm-corrected pairwise tests
""")
add(table(pairwise, 6))
add("The Friedman result rejects equality of all ranks. Five folds do not support corrected pairwise-superiority claims.")


# PINN and feature evidence
add("""---

## 14. PINN, knowledge descriptor, and feature-ablation evidence

### 14.1 Physics-constrained PINN equations

```text
dQhat/dk = -alpha0 exp(-Ea/(R T)) Qhat^beta
R_Q = dQhat/dk + alpha0 exp(-Ea/(R T)) Qhat^beta
R_Re = dRehat/dk - g_theta(k,T,x)

L_PINN = L_data + lambda_IC L_IC
         + gamma r(epoch)[lambda_m L_mono + lambda_s L_smooth + lambda_p L_phys]

L_phys = ||R_Q||^2 + 0.3||R_Re||^2 + L_diversity/prior
```

The final implementation does **not** use unexplained fixed `lambda_Q, lambda_R, lambda_m, lambda_s, lambda_E` values from the older draft. It uses adaptive gradient-norm balancing:

- Adaptive coefficients initialized at 1.
- Update every five epochs.
- EMA momentum 0.8.
- `lambda_m` range 0.01–5.
- `lambda_s` range 0.001–1.
- `lambda_p` range 0.02–0.5.
- `lambda_IC` range 0.01–3.
- Capacity residual internal weight 1.
- Resistance residual internal weight 0.3.
- Activation-energy mean-prior and diversity terms each have coefficient 0.05 before multiplication by `lambda_p`, `gamma`, and the epoch ramp.

Activation energy:

- Prior `Ea,0 = 56 kJ mol^-1`.
- Prior scale 7 kJ mol^-1.
- Diversity floor 4 kJ mol^-1.
- Allowed range 40–80 kJ mol^-1.
- Recovered full-model value ≈57.5 kJ mol^-1.
- Source-cell fitted range 51.75–61.48 kJ mol^-1.
- Source-cell mean `55.69 ± 2.86 kJ mol^-1`.
- This is a bounded, prior-guided diagnostic—not an independent physical measurement.

### 14.2 PINN grouped-fold accuracy

| Variant | Macro MAPE mean ± SD |
|---|---:|
| Data-only PINN prediction backbone | 7.560 ± 0.195% |
| Physics-constrained PINN | 8.049 ± 0.294% |

The implemented physics loss does not improve prediction over its data-only backbone.

### 14.3 Feature-ablation results

| Variant | Teacher-forced macro MAPE | Recursive rollout macro MAPE |
|---|---:|---:|
| Full baseline | 1.168 | 1.457 |
| No temperature | 1.312 | 2.004 |
| No ageing index | 1.374 | 1.847 |
| k + T | 0.613 | 1.938 |
| k + Re0 | 0.616 | **1.284** |
| k + T + Re0 | 0.682 | 2.090 |
| k + T + Re0 + Rct0 + Q0 | 0.806 | 1.406 |

Feature conclusions reverse between teacher forcing and rollout. `k + Re0` is the strongest compact recursive variant in this ablation.

`No near-zero features` in the archived feature table means SOC window, discharge rate, and ageing type were removed because decision-tree importances were ≤0.02%; it does not mean rows or MAPE denominators were filtered.

### 14.4 Baseline-number reconciliation

- 1.712%: original direct-rollout fixed-split run.
- 1.457%: separately retrained full-feature ablation baseline.
- 1.627%: grouped-fold mean.
- 1.855862 ± 0.226975%: final corrected ten-seed manual model on the common 76-point window.

These are different fits/protocols and must not be mixed as repeated estimates of one model.
""")


# Broad fixed split
broad = pd.read_csv(AUDIT / "detailed_source_domain_fixed_split_metrics.csv")
hparams = pd.read_csv(AUDIT / "hyperparameter_reproducibility_table.csv")
add("""---

## 15. Original broad fixed-split benchmark

These results are historical/model-family context. They are single-run fixed-split results unless their label explicitly states an ensemble. They must not be mixed with the final ten-seed common-window optimizer table.
""")
add(table(broad, 6))
add("""### 15.1 Archived hyperparameter and reproducibility ledger

This table records the exact information recoverable from the original benchmark artifacts. `Not recorded` means the archived result does not support inventing a value.
""")
add(table(hparams, 6))
add("""### 15.2 Roles of the original baselines

- **Decision tree:** pointwise nonlinear low-compute baseline and preliminary feature-importance tool; no temporal state.
- **Random Forest / ExtraTrees / XGBoost:** pointwise full-nine-feature ensemble controls selected by validation grids.
- **Residual MLP:** pointwise neural control with shared residual encoder; no recurrent degradation state.
- **LSTM v1:** teacher-forced diagnostic using measured prior Q/Re during inference.
- **LSTM v2:** exogenous history without target feedback.
- **LSTM v3:** direct recursive Q/Re feedback.
- **LSTM v4:** teacher-pretrained and rollout-fine-tuned recursive model.
- **PINN_pred:** data-only counterpart with the same prediction backbone as PINN_phys.
- **PINN_phys:** prediction backbone plus initial-condition, monotonicity, smoothness, and Arrhenius residual penalties.
- **Neural ODE:** state `[Q,Re]`, learned vector field conditioned on exogenous variables, fixed-step RK4 propagation; approximately 2.5 h training and no accuracy advantage.
""")


# Transfer
transfer_summary = pd.read_csv(FINAL / "transfer" / "transfer_summary.csv")
transfer_seeds = pd.read_csv(FINAL / "transfer" / "transfer_seed_metrics.csv")
transfer_exp = pd.read_csv(FINAL / "transfer" / "transfer_experiment_metrics.csv")
transfer_tests = pd.read_csv(FINAL / "transfer" / "transfer_cell_level_tests.csv")
transfer_timing = pd.read_csv(FINAL / "transfer" / "transfer_timing.csv")
exp_controls = transfer_exp[transfer_exp.seed.isna()][["model", "exp", "Q_MAPE", "n_eval_rows", "n_eval_cells"]]
exp_neural = transfer_exp[transfer_exp.seed.notna()].groupby(["model", "exp"], as_index=False).agg(
    Q_MAPE_mean=("Q_MAPE", "mean"), Q_MAPE_std=("Q_MAPE", "std"), n_eval_rows=("n_eval_rows", "first"), n_eval_cells=("n_eval_cells", "first"))
add("""---

## 16. Complete 40-cell external transfer audit

### 16.1 External dataset

- LG M50T 21700 cells: 40.
- Five experiments with cell counts 9, 6, 9, 8, 8.
- Total checkup records: 511.
- Post-BOL records: 471.
- Later-half evaluation points: 246.
- Temperatures: nominal 10, 25, 40 °C; parsed range 10.0–43.098 °C.
- Ageing-cycle range: 0–6,204.
- Capacity range: 2.184908–4.905906 Ah.
- External comparison target: capacity/SOH only.

### 16.2 Calibration and evaluation

- For every trajectory, the first 50% is calibration data and the later 50% is evaluation-only.
- Cutoff is `floor((n-1)*0.50)`, bounded to preserve at least one calibration transition and one evaluation row.
- External SOH is `Q/Q0`; resistance normalization is `Re/Re0` where used as input.
- Early sequences shorter than 20 are left-padded by repeating the earliest row.

### 16.3 Neural architecture

- Two-layer LSTM, hidden width 96, dropout 0.1, sequence length 20.
- Smooth-L1 loss, AdamW, gradient clipping.
- Source model: source-scaled pretraining on 209 source train/9 validation cells, maximum 140 epochs, patience 25, learning rate 1e-3, weight decay 1e-4, batch 512.
- Head adaptation: recurrent parameters frozen, prediction head only, learning rate 1e-4, gradient clip 0.25, maximum 180 epochs, patience 25.
- Target-only: identical architecture freshly initialized, target-only scalers, learning rate 1e-3, weight decay 1e-4, maximum 180 epochs, patience 25.
- Neural seeds: 42, 52, 62, 72, 82.

### 16.4 Deterministic SOH controls

1. BOL persistence predicts the first SOH.
2. Last-observation persistence predicts the final calibration SOH.
3. Linear fade fits `SOH = intercept + slope*k`.
4. Square-root fade fits `SOH = intercept + slope*sqrt(k)`.

Positive fitted slopes are clipped to zero. SOH predictions are clipped to [0.01,1.20], then multiplied by Q0.

### 16.5 Aggregate results
""")
add(table(transfer_summary, 6))
add("### 16.6 Per-seed neural and deterministic-control ledger")
add(table(transfer_seeds, 6))
add("### 16.7 Per-experiment deterministic controls")
add(table(exp_controls, 6))
add("### 16.8 Per-experiment neural mean ± SD")
add(table(exp_neural, 6))
add("### 16.9 Cell-level paired tests")
add(table(transfer_tests, 9))
add("### 16.10 Transfer timing by seed")
add(table(transfer_timing, 6))
add("""
Additional target-only versus square-root-fade cell comparison derived from the final cell table:

- Mean paired cell difference: approximately -0.215 percentage points.
- Bootstrap 95% CI: approximately [-1.024, 0.650].
- Wilcoxon p ≈0.485.
- Target-only wins on 20 cells; square-root fade wins on 20 cells.

Interpretation constraints:

- Source-pretrained head adaptation is significantly worse than target-only training and square-root fade.
- This is negative transfer under the implemented frozen-representation/head-only protocol.
- Target-only and square-root fade are statistically unresolved at cell level.
- The old eight-cell 3.42% source-adaptation result is superseded and cannot support a transfer benefit.
""")


add("""---

## 17. Runtime and hardware ledger

### 17.1 Final optimizer environment

- Kaggle Tesla T4 GPU (`cuda`).
- Python 3.12.13.
- PyTorch 2.10.0+cu128.
- NSGA-II 15-run search elapsed time: 6.1545 h, excluding the preceding noise-floor stage.
- NSGA-III 15-run search elapsed time: 6.2173 h.
- Random search and final analysis ran in Notebook 3; individual final training times are in Section 10.

### 17.2 CPU round-two measurements

| Model | Training | Trajectory inference | Notes |
|---|---:|---:|---|
| TCN | 26.775 ± 7.388 s | 17.080 ± 1.527 ms | five seeds |
| PINN-feature hybrid | 22.745 s | 10.262 ± 2.890 ms | one CPU thread, 100 repetitions |
| Archived old manual rollout LSTM | 52.05 s | 11.45 ms | obsolete optimization protocol |
| Archived old compact DE LSTM | 36.01 s | 8.20 ms | obsolete optimization protocol |

Never calculate speedups across CPU and T4 panels as if hardware were identical.

---

## 18. Evidence hierarchy and superseded results

### 18.1 Authoritative final evidence

1. `10_asc_optimizer_study/complete_results_audit_20260822/` — final repeated optimizer, re-screen, ten-seed confirmation, predictions, statistics, convergence.
2. `10_asc_optimizer_study/received_results_audit_20260821/nsga2/` — NSGA-II run files and noise floor.
3. `10_asc_optimizer_study/received_results_audit_20260821/nsga3/` — NSGA-III run files.
4. `09_asc_rewrite_evidence/final_results/mamba/` — official Mamba baseline.
5. `09_asc_rewrite_evidence/final_results/transfer/` — complete 40-cell transfer audit.
6. `07_round2_revision/outputs/` — TCN, stress ablation, grouped-fold statistics, PINN/feature/runtime evidence.
7. `06_manuscript_context/submission_audit_csv/` — original benchmark and reproducibility ledgers.

### 18.2 Superseded/obsolete evidence

- Earlier 32-evaluation NSGA-II/random pilot and its five-seed final table.
- Old DE search and old DE-selected result.
- Manual `L=20,H=10` rollout-tuned result, because it has zero rollout-training segments.
- Old optimizer comparisons where changing L changed the number of evaluated rows.
- Old eight-cell Kirkaldy transfer result (3.42%).
- BOL persistence calculated in raw Ah space rather than SOH space.
- Claims that the stress descriptor improves accuracy.
- Claims that 57.5 kJ mol^-1 is an independent physical measurement.

---

## 19. Complete artifact map

| Evidence | File |
|---|---|
| All 12,600 optimizer trials | `10_asc_optimizer_study/complete_results_audit_20260822/all_optimizer_trials.csv` |
| HV/IGD by run and budget | `.../hv_igd_by_run_and_evaluation.csv` |
| HV/IGD convergence means | `.../hv_igd_convergence_summary.csv` |
| Global normalization | `.../global_normalization.json` |
| Optimizer statistical tests | `.../optimizer_statistical_tests.csv` |
| Re-screen candidates | `.../rescreen_candidates.csv` |
| Re-screen raw metrics | `.../candidate_rescreen_raw.csv` |
| Re-screen summaries | `.../candidate_rescreen_summary.csv` |
| Final selected configurations | `.../selected_finalists.csv` |
| Ten-seed final raw metrics | `.../final_ten_seed_raw.csv` |
| Ten-seed final summary/CIs | `.../final_ten_seed_summary.csv` |
| Final paired tests | `.../final_paired_tests_holm.csv` |
| Final 3,040 predictions | `.../final_predictions.csv` |
| Optimizer convergence PNG/PDF | `.../optimizer_convergence.png`, `optimizer_convergence.pdf` |
| Noise-floor raw/summary | `10_asc_optimizer_study/received_results_audit_20260821/nsga2/noise_floor_raw.csv`, `noise_floor_summary.csv` |
| Individual NSGA run trials/fronts | corresponding `run_XX_trials.csv`, `run_XX_front.csv` files |
| Mamba metrics/predictions/history | `09_asc_rewrite_evidence/final_results/mamba/` |
| Transfer metrics/predictions/tests | `09_asc_rewrite_evidence/final_results/transfer/` |
| Stress ablation | `07_round2_revision/outputs/stress_ablation_*` |
| TCN | `07_round2_revision/outputs/tcn_*` |
| Fold omnibus/ranks/pairwise | `07_round2_revision/outputs/statistical_*` |
| Original fixed-split metrics | `06_manuscript_context/submission_audit_csv/detailed_source_domain_fixed_split_metrics.csv` |

---

## 20. Final evidence-based claim boundaries

Supported:

- NSGA-II has significantly better multi-objective search hypervolume than NSGA-III and matched random search under the 4,200-evaluation budget.
- NSGA-II has significantly better IGD than random search after Holm correction.
- The NSGA-II finalist is 92.5% smaller and about 39% faster per forward pass than the corrected manual LSTM, with a small nonsignificant numerical MAPE reduction.
- The compact finalist demonstrates resource-aware model selection; it does not establish significant accuracy improvement or formal ±0.20-point equivalence.
- TCN and official Mamba are weaker than the compact LSTM configurations on this sparse-checkup task.
- The implemented stress descriptor significantly worsens macro MAPE.
- Physics constraints do not improve the PINN backbone's predictive accuracy.
- The recovered activation energy is prior-guided.
- Frozen source-representation/head-only adaptation causes significant negative transfer on the full 40-cell audit.

Unsupported and prohibited:

- NSGA-II significantly improves final test MAPE.
- NSGA-II is a novel optimization algorithm.
- NSGA-III is superior because it produces more nondominated points.
- Stress improves forecasting.
- The PINN recovers an independently measured physical activation energy.
- Source pretraining improves Kirkaldy performance.
- Runtime values from different hardware are directly comparable.
- Five grouped folds establish Holm-corrected pairwise model superiority.

---

## 21. Items still requiring author input rather than computation

- Final repository URL and release status.
- Final author names, affiliations, ORCIDs, corresponding-author details.
- Funding statement and grant identifiers.
- Competing-interest declaration.
- Final code/data license.
- Confirmation of whether any figure/table should move to supplementary material.

All scientific computation required for drafting the revised paper is complete.
""")


OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)
print("bytes", OUT.stat().st_size, "lines", len(OUT.read_text(encoding="utf-8").splitlines()))
