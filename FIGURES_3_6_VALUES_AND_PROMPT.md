# Figures 3–6: exact values and regeneration prompt

## Files containing every plotted value

Upload these six files with the prompt below. They are the authoritative numeric inputs.

1. `results/figures_3_6/fig3_checkpoint_summary.csv`
   - 30 rows: mean and sample standard deviation of hypervolume.
   - Three methods × ten evaluation checkpoints × 15 runs.
2. `results/figures_3_6/fig3_first_passage_by_run.csv`
   - 45 rows: target attainment and first-passage evaluation for every run.
3. `results/optimizer/all_optimizer_trials.csv`
   - 12,600 rows: complete candidate cloud used as the grey background in Figures 4a and 4b.
4. `results/figures_3_6/pooled_nondominated_28.csv`
   - All 28 pooled nondominated candidates highlighted in Figures 4a and 4b.
5. `results/figures_3_6/fig5_teacher_rollout_values.csv`
   - All seven feature-set ablation values and ranks used in Figure 5.
6. `results/figures_3_6/fig6_transfer_cell_effects.csv`
   - All 40 cell-level transfer, target-only, and paired-difference values used in Figure 6.

## Key numerical checks

### Figure 3

Evaluation checkpoints: 28, 56, 84, 112, 140, 168, 196, 224, 252, and 280.

Final hypervolume at 280 evaluations, reported as mean ± sample SD over 15 runs:

- NSGA-II: 1.197236167549 ± 0.018458037917
- NSGA-III: 1.157910513212 ± 0.024418660083
- Random search: 1.141018530370 ± 0.016368085044

Target hypervolume: 1.141019.

First-passage results:

- NSGA-II: 15/15 runs reached the target; mean 71.200 evaluations; SD 49.765.
- NSGA-III: 10/15 runs reached the target; mean 125.100 evaluations; SD 66.963.
- Random search: 8/15 runs reached the target; mean 166.625 evaluations; SD 68.025.
- Runs that did not reach the target are plotted as `×` at 280 evaluations.

### Figures 4a and 4b

- Total candidates: 12,600.
- Candidates per method: 4,200.
- Pooled three-objective nondominated candidates: 28.
- Contribution to pooled front: NSGA-II 21, NSGA-III 2, Random search 5.
- Objectives are minimized: validation macro MAPE, trainable parameters, and inference latency.
- Parameter and latency values are log-transformed only for dominance normalization and displayed with logarithmic x axes.
- Fixed normalization bounds:
  - MAPE: 0.7308796223 to 8.8913714513
  - log10(parameters): 4.5613160916 to 5.8939979806
  - log10(latency in ms): -0.4342352697 to 0.0246944637

### Figure 5

| Feature set | Teacher-forced MAPE | Recursive-rollout MAPE | Teacher rank | Rollout rank |
|---|---:|---:|---:|---:|
| Full features | 1.168 | 1.457 | 5 | 3 |
| No temperature | 1.312 | 2.004 | 6 | 6 |
| No ageing index | 1.374 | 1.847 | 7 | 4 |
| $k_{exp}+T$ | 0.613 | 1.938 | 1 | 5 |
| $k_{exp}+R_{e,0}$ | 0.616 | 1.284 | 2 | 1 |
| $k_{exp}+T+R_{e,0}$ | 0.682 | 2.090 | 3 | 7 |
| $k_{exp}+T+R_{e,0}+R_{pulse,0}+Q_0$ | 0.806 | 1.406 | 4 | 2 |

### Figure 6

The paired difference is:

`transfer capacity MAPE − target-only capacity MAPE`, in percentage points.

- External cells: 40.
- Mean paired difference: +6.987501122438 pp.
- Median paired difference: +5.034457523492 pp.
- Bootstrap 95% CI for the mean: [5.187332211161, 8.794860466086] pp.
- Bootstrap seed: 20260820.
- Bootstrap resamples: 10,000.
- Two-sided paired Wilcoxon signed-rank statistic: W = 25.0.
- Wilcoxon p-value: 1.6443664208054543 × 10⁻9.
- Six cells have a negative difference and 34 have a positive difference.
- Negative difference means beneficial transfer; positive difference means negative transfer.

## Copy-ready figure-generation prompt

```text
Create publication-quality Figures 3–6 for an Elsevier Applied Soft Computing manuscript using ONLY the values in the six attached CSV files. Do not estimate or digitize values from screenshots. Do not change, smooth, omit, winsorize, or recompute the supplied observations except for the explicitly described grouping and statistics.

GENERAL FORMAT
- Use Python, pandas, NumPy, SciPy, and Matplotlib.
- Font: DejaVu Sans.
- Base font 10 pt; axis labels 11 pt and bold; tick labels at least 9.5 pt; legend 9.5 pt.
- Axis linewidth 0.8 pt. Outward ticks, width 0.8 pt, length 3 pt.
- Remove top and right spines.
- Use light horizontal grid lines: #d9d9d9, linewidth 0.55, alpha 0.7.
- Export every figure as vector PDF and 600-dpi PNG with tight bounding boxes.
- Use embedded TrueType fonts in PDF (`pdf.fonttype = 42`, `ps.fonttype = 42`).
- Keep backgrounds white. Avoid unnecessary prose inside plots.
- Method colors: NSGA-II #2166ac; NSGA-III #d6604d; Random #4d4d4d.
- Method markers: NSGA-II circle; NSGA-III upward triangle; Random square.
- Use “Random search” in legends.

FIGURE 3 — OPTIMIZER CONVERGENCE AND BUDGET TO PARITY
Output names: `fig_optimizer_convergence_budget.png` and `.pdf`.
Create a two-panel horizontal figure, 8.2 × 3.7 inches, constrained layout, panel-width ratio 1.72:1.

Panel (a):
- Read `fig3_checkpoint_summary.csv`.
- Plot mean hypervolume versus candidate evaluations for NSGA-II, NSGA-III, and Random.
- Plot mean lines with the assigned colors and markers; linewidth 1.45, marker size 3.8.
- Add a translucent band of mean ± sample SD, same color, alpha 0.13, no edge.
- Add a horizontal dashed target line at HV = 1.141019, color #555555, linewidth 0.9.
- X ticks: 28, 56, 84, 112, 140, 168, 196, 224, 252, 280; rotate 45°.
- Bold x label: “Candidate evaluations”.
- Bold y label: “Hypervolume (HV)”.
- Legend inside lower right, frameless.
- Put bold panel label “(a)” at the upper-left.

Panel (b):
- Read `fig3_first_passage_by_run.csv`.
- For each method, plot each reached run at its first-passage evaluation using a filled circle in the method color, size 18, thin white edge.
- Horizontally jitter the 15 run positions deterministically from −0.16 to +0.16.
- Plot each unreached run as an `×` at y = 280, size 24, linewidth 1.0.
- Calculate the mean only over reached runs and draw a black horizontal mean bar from x−0.22 to x+0.22, linewidth 1.8.
- Near the bottom of each group write the reached count and reached-run mean on two lines: NSGA-II `15/15` and `71.2`; NSGA-III `10/15` and `125.1`; Random `8/15` and `166.6`.
- Y limits 0–300.
- Bold x label: “Search method”.
- Bold y label: “Evaluations to target”.
- Put bold panel label “(b)” at the upper-left.

FIGURE 4a — POOLED PARETO FRONT: MAPE VERSUS PARAMETERS
Output names: `fig4a_pareto_mape_parameters.png` and `.pdf`.
- This must be an independent PNG/PDF, not a subplot combined with Figure 4b.
- Size: 4.6 × 3.65 inches, constrained layout.
- Read all 12,600 rows from `all_optimizer_trials.csv` and draw them as the grey candidate cloud: x = parameters, y = validation_macro_MAPE, marker size 2, #bdbdbd, alpha 0.20, no edges, rasterized.
- Read `pooled_nondominated_28.csv` and overlay all 28 nondominated candidates using their method color and marker, size 25, edge #222222, edge width 0.45.
- Use a logarithmic x axis.
- Bold x label: “Trainable parameters”.
- Bold y label: “Validation macro MAPE (%)”.
- Put a frameless three-column legend above and outside the axes, centered.

FIGURE 4b — POOLED PARETO FRONT: MAPE VERSUS LATENCY
Output names: `fig4b_pareto_mape_latency.png` and `.pdf`.
- This must be an independent PNG/PDF.
- Use the same size, complete grey candidate cloud, 28 highlighted nondominated points, colors, markers, and external legend as Figure 4a.
- X = latency_ms; y = validation_macro_MAPE.
- Use a logarithmic x axis.
- Bold x label: “Inference latency (ms)”.
- Bold y label: “Validation macro MAPE (%)”.

FIGURE 5 — TEACHER-FORCED VERSUS RECURSIVE-ROLLOUT ABLATION
Output names: `fig_teacher_rollout_ablation.png` and `.pdf`.
- Size: 7.5 × 4.3 inches.
- Read all seven rows from `fig5_teacher_rollout_values.csv`.
- Make a slopegraph from x = 0 (“Teacher forced”) to x = 1 (“Recursive rollout”).
- Plot every feature set’s two values and connect them.
- Highlight `$k_{exp}+T$` in red #b2182b, linewidth 2.0, point size 28.
- Highlight `$k_{exp}+R_{e,0}$` in blue #2166ac, linewidth 2.0, point size 28.
- Plot “Full features” in grey #8c8c8c, linewidth 1.35, alpha 0.72.
- Plot the other grey series with linewidth 0.9, alpha 0.72, point size 18.
- Label each feature set to the left of its teacher-forced point.
- Print each recursive-rollout value to three decimals to the right of its rollout point.
- Add red annotation “Rank 1 → 5” and blue annotation “Rank 2 → 1”.
- X limits −0.75 to 1.28; y limits 0.5 to 2.2.
- Bold y label: “Macro MAPE (%)”.
- Make the two x-category labels bold and place them clearly below the plotting area with at least 17 pt tick padding and sufficient bottom margin. They must not sit inside the data area.

FIGURE 6 — CELL-LEVEL EXTERNAL TRANSFER EFFECT
Output names: `fig_external_transfer.png` and `.pdf`.
- Read `fig6_transfer_cell_effects.csv`, already sorted by delta_pp ascending. Preserve this order exactly.
- Use a two-column figure, 8.2 × 3.8 inches, shared y axis, width ratio 8.5:1.2, constrained layout.
- Main panel: x = ordered positions 1–40; y = delta_pp.
- Color delta_pp < 0 blue #2166ac and label it “Beneficial transfer”.
- Color delta_pp ≥ 0 red #b2182b and label it “Negative transfer”.
- Scatter size 23, thin white edge width 0.35.
- Draw a horizontal zero line in #555555, linewidth 0.9.
- X ticks at 1, 10, 20, 30, and 40.
- Bold x label: “External cells ordered by paired effect”.
- Bold y label: “Paired MAPE difference (pp)”.
- Do not place statistical prose inside the main data panel.
- Separate right panel: show the mean +6.987501122438 as a white diamond with black edge and asymmetric 95% CI [5.187332211161, 8.794860466086], black error bar, cap size 4, linewidth 1.2.
- Put bold text “+6.99” above the interval.
- Use x tick label “Mean” in bold and panel title “95% CI” in bold.
- Put the legend above and outside the axes, centered, three columns, frameless. Legend order: Negative transfer, Beneficial transfer, Mean and 95% CI.
- Ensure the legend does not overlap data, axes, titles, or labels.

NUMERIC VALIDATION — FAIL IF ANY CHECK DOES NOT MATCH
- Figure 3 final mean HV: NSGA-II 1.197236167549; NSGA-III 1.157910513212; Random 1.141018530370.
- Figure 3 target reach: 15/15, 10/15, and 8/15, respectively.
- Figure 4 total candidate count: 12,600.
- Figure 4 pooled nondominated count: 28, comprising 21 NSGA-II, 2 NSGA-III, and 5 Random candidates.
- Figure 5 must contain exactly seven feature sets and reproduce every supplied MAPE and rank.
- Figure 6 must contain exactly 40 cells; mean +6.987501122438 pp; median +5.034457523492 pp; bootstrap CI [5.187332211161, 8.794860466086]; Wilcoxon W = 25.0 and p = 1.6443664208054543e-09.

Return the Python generation script plus all ten final outputs: five PNGs and five PDFs. Do not add chart titles or explanatory paragraphs inside the data regions.
```
