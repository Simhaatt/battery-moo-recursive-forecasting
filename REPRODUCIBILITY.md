# Reproducibility guide

## Level 1: reproduce reported analysis

Requirements: Python 3.10+ and CPU. Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python analysis/reproduce_all.py
pytest -q
```

This validates 12,600 search evaluations, recomputes final hypervolume means, pooled-front contributions, final accuracy/complexity values, stress and transfer claims, and regenerates five core figures plus manuscript-ready CSV tables. It should take under one minute on a normal laptop and requires neither raw data nor a GPU.

## Level 2: rerun neural experiments

Requirements: licensed source datasets, the exact processed input schemas, PyTorch, and preferably an NVIDIA T4-class GPU. Full matched searches comprise 12,600 candidate trainings and are intentionally split by method/run on Kaggle. Expected approximate T4 times from the completed study were: noise floor under 30 min; each 8-run optimizer block 3–4 h; each 7-run block about 3 h; ten-seed final evaluation and analysis under 1 h. Runtime varies with Kaggle load.

The immutable historical runners are under `notebooks/archive/07_round2_revision`, `08_softcomputing_extension`, and `10_asc_optimizer_study`. Use `scripts/run_search.py` and `scripts/run_auxiliary.py`, which require explicit paths and refuse to overwrite a nonempty output directory.

## Determinism

Run seeds are 20270001–20270015; training seeds are 41001–41015. Noise seeds are 42, 52, 62, 72, 82, 92, 102, 112, 122, and 132. Rescreen seeds are 42, 52, and 62. CUDA kernels and timing are hardware-dependent, so exact retraining equality is not guaranteed; artifact-level recomputation is exact to the configured tolerance.

## Training schedules

- Search: 80 teacher epochs and 18 rollout epochs (the archived runner records its effective patience schedule in its manifest).
- Rescreen: 160/30 epochs, patience 30, minimum 10.
- Final: 220/40 epochs, patience 40, minimum 15.
- AdamW, weight decay 1e-5, gradient clipping 1.0, batches 512/128, rollout LR multiplier 0.3.

## Release workflow

1. Run Level 1 and tests from a clean checkout.
2. Confirm `results/reproduction_check.json` says `PASS`.
3. Commit only derived/public data allowed by source licenses.
4. Push to GitHub and create annotated tag/release `v1.0.0`.
5. Connect the repository to Zenodo and archive the release.
6. Replace placeholder repository and DOI strings in `CITATION.cff`, README, and manuscript.
7. Add the release archive checksum to the release notes.

## Data and Code Availability text

```latex
\section*{Data and Code Availability}
The source battery datasets are available from their original providers under
their respective terms and are not redistributed. Derived split assignments,
experiment outputs, analysis code, and the manuscript-value reproduction check
are available at \url{https://github.com/Simhaatt/battery-moo-recursive-forecasting}.
The version used for this article is archived at Zenodo: \url{https://doi.org/10.5281/zenodo.22057384}.
```
