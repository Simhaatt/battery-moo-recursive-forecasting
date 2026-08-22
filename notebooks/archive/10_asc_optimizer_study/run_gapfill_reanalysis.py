"""Evidence-only reanalysis for ASC_GAPFILL_20260822.md.

This script reads existing artifacts and writes only new CSV files under the
complete-results audit's gapfill directory. It performs no model training.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import iqr, kruskal, levene, mannwhitneyu, wilcoxon


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "complete_results_audit_20260822"
RECEIVED = ROOT / "received_results_audit_20260821"
OUT = AUDIT / "gapfill"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ("NSGA-II", "NSGA-III", "Random")


def nondominated(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    # Incremental exact minimization front. The observed pooled front is tiny
    # relative to 12,600 points, making this much faster than an n-by-n matrix.
    front: list[int] = []
    for idx, point in enumerate(values):
        if front:
            current = values[front]
            if np.any(np.all(current <= point, axis=1) & np.any(current < point, axis=1)):
                continue
            keep = ~(np.all(point <= current, axis=1) & np.any(point < current, axis=1))
            front = [old for old, flag in zip(front, keep) if flag]
        front.append(idx)
    return np.asarray(front, dtype=int)


def hv2(points: np.ndarray, ref: np.ndarray) -> float:
    if not len(points):
        return 0.0
    ys = sorted(set(points[:, 0].tolist())) + [float(ref[0])]
    total = 0.0
    for left, right in zip(ys[:-1], ys[1:]):
        active = points[points[:, 0] <= left]
        if len(active):
            total += max(0.0, right-left) * max(0.0, ref[1]-active[:, 1].min())
    return float(total)


def hv3(points: np.ndarray, ref=np.array([1.1, 1.1, 1.1])) -> float:
    points = points[np.all(points < ref, axis=1)]
    if not len(points):
        return 0.0
    points = points[nondominated(points)]
    xs = sorted(set(points[:, 0].tolist())) + [float(ref[0])]
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = points[points[:, 0] <= left][:, 1:]
        total += max(0.0, right-left) * hv2(active, ref[1:])
    return float(total)


def holm(rows: list[dict]) -> list[dict]:
    order = sorted(range(len(rows)), key=lambda j: rows[j]["p_raw"])
    adjusted = np.zeros(len(rows))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(rows)-rank)*rows[idx]["p_raw"])
        adjusted[idx] = min(1.0, running)
    for idx, val in enumerate(adjusted):
        rows[idx]["p_holm"] = val
    return rows


def a12(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + .5*(x[:, None] == y[None, :]).sum())/(len(x)*len(y)))


def tests(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        groups = [frame.loc[frame.method == m, metric].to_numpy() for m in METHODS]
        stat, p = kruskal(*groups)
        rows.append(dict(metric=metric, test="Kruskal-Wallis", comparison="all",
                         statistic=stat, p_raw=p, p_holm=p, A12_first_greater=np.nan))
        pairs = []
        for first, second in combinations(METHODS, 2):
            x = frame.loc[frame.method == first, metric].to_numpy()
            y = frame.loc[frame.method == second, metric].to_numpy()
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            pairs.append(dict(metric=metric, test="Mann-Whitney U",
                              comparison=f"{first} vs {second}", statistic=stat,
                              p_raw=p, A12_first_greater=a12(x, y)))
        rows.extend(holm(pairs))
    return pd.DataFrame(rows)


trials = pd.read_csv(AUDIT / "all_optimizer_trials.csv")
normal = json.loads((AUDIT / "global_normalization.json").read_text(encoding="utf-8"))
lo = np.asarray(normal["lower"], float)
hi = np.asarray(normal["upper"], float)
raw_obj = np.column_stack((trials.validation_macro_MAPE,
                           np.log10(trials.parameters), np.log10(trials.latency_ms)))
norm_obj = (raw_obj-lo)/np.maximum(hi-lo, 1e-12)
trials[["norm_mape", "norm_log_params", "norm_log_latency"]] = norm_obj
OBJ = ["norm_mape", "norm_log_params", "norm_log_latency"]

# 10. Winner's curse.
best = trials.sort_values(["method", "run", "validation_macro_MAPE", "evaluation"]).groupby(
    ["method", "run"], as_index=False).first()
rescreen = pd.read_csv(AUDIT / "candidate_rescreen_raw.csv")
noise = pd.read_csv(RECEIVED / "nsga2" / "noise_floor_raw.csv")
rescreen["multi_seed_source"] = str(AUDIT / "candidate_rescreen_raw.csv")
noise["multi_seed_source"] = str(RECEIVED / "nsga2" / "noise_floor_raw.csv")
multi = pd.concat([rescreen, noise], ignore_index=True)
# A candidate configuration has identical outcomes for the same seed regardless of
# the method label through which it was nominated; deduplicate candidate/seed.
multi = multi.sort_values(["candidate_id", "seed", "multi_seed_source"]).drop_duplicates(
    ["candidate_id", "seed"])
multi_sum = multi.groupby("candidate_id", as_index=False).agg(
    multi_seed_mean=("validation_macro_MAPE", "mean"),
    multi_seed_sd=("validation_macro_MAPE", "std"),
    multi_seed_n=("seed", "nunique"),
    multi_seed_source=("multi_seed_source", lambda x: " | ".join(sorted(set(x)))))
winner = best[["method", "run", "evaluation", "candidate_id", "validation_macro_MAPE"]].rename(
    columns={"validation_macro_MAPE": "search_stage_best_MAPE"}).merge(multi_sum, how="left", on="candidate_id")
winner["optimism_gap_pp"] = winner.multi_seed_mean - winner.search_stage_best_MAPE
winner.to_csv(OUT / "winner_curse_by_run.csv", index=False)
summary_rows = []
for method in METHODS:
    d = winner[(winner.method == method) & winner.multi_seed_mean.notna()]
    if len(d):
        try:
            wstat, wp = wilcoxon(d.search_stage_best_MAPE, d.multi_seed_mean, alternative="two-sided")
        except ValueError:
            wstat, wp = np.nan, np.nan
    else:
        wstat, wp = np.nan, np.nan
    summary_rows.append(dict(method=method, matched_candidates=len(d), total_runs=15,
                             mean_optimism_gap_pp=d.optimism_gap_pp.mean(),
                             sd_optimism_gap_pp=d.optimism_gap_pp.std(),
                             wilcoxon_statistic=wstat, wilcoxon_p=wp))
pd.DataFrame(summary_rows).to_csv(OUT / "winner_curse_summary.csv", index=False)

# 11. Evaluations to random final mean HV (at every evaluation, not only 28-step checkpoints).
target = 1.141019
parity_rows = []
for (method, run), g in trials.groupby(["method", "run"], sort=False):
    g = g.sort_values("evaluation")
    # Hypervolume is monotone as points are added, so binary search gives the
    # exact first evaluation with far fewer repeated front calculations.
    reached = np.nan
    if hv3(g[OBJ].to_numpy()) >= target:
        left, right = 1, len(g)
        while left < right:
            middle = (left + right)//2
            if hv3(g.iloc[:middle][OBJ].to_numpy()) >= target:
                right = middle
            else:
                left = middle + 1
        reached = left
    parity_rows.append(dict(method=method, run=run, target_hv=target,
                            evaluations_to_parity=reached, reached=not np.isnan(reached)))
parity = pd.DataFrame(parity_rows)
parity.to_csv(OUT / "budget_to_parity_by_run.csv", index=False)
ps = parity.groupby("method", as_index=False).agg(
    runs_reaching=("reached", "sum"), total_runs=("run", "count"),
    mean_evaluations_reaching=("evaluations_to_parity", "mean"),
    sd_evaluations_reaching=("evaluations_to_parity", "std"))
random_mean = float(ps.loc[ps.method == "Random", "mean_evaluations_reaching"].iloc[0])
ps["speedup_vs_random_reachers"] = random_mean / ps.mean_evaluations_reaching
ps.to_csv(OUT / "budget_to_parity_summary.csv", index=False)

# 12. IGD+ and method-specific leave-one-method-out Euclidean IGD.
igd_rows = []
for method in METHODS:
    excluded_pool = trials.loc[trials.method != method, OBJ].to_numpy()
    loo_ref = excluded_pool[nondominated(excluded_pool)]
    for run, g in trials[trials.method == method].groupby("run"):
        front = g[OBJ].to_numpy()
        front = front[nondominated(front)]
        pooled_ref = norm_obj[nondominated(norm_obj)]
        plus = np.sqrt(np.square(np.maximum(front[None, :, :] - pooled_ref[:, None, :], 0)).sum(axis=2))
        euclid = np.linalg.norm(loo_ref[:, None, :] - front[None, :, :], axis=2)
        igd_rows.append(dict(method=method, run=run, IGD_plus=plus.min(axis=1).mean(),
                             LOO_IGD=euclid.min(axis=1).mean(),
                             pooled_reference_size=len(pooled_ref),
                             loo_reference_size=len(loo_ref)))
igd_frame = pd.DataFrame(igd_rows)
igd_frame.to_csv(OUT / "igd_plus_loo_by_run.csv", index=False)
tests(igd_frame, ["IGD_plus", "LOO_IGD"]).to_csv(OUT / "igd_plus_loo_tests.csv", index=False)
igd_frame.groupby("method", as_index=False).agg(
    IGD_plus_mean=("IGD_plus", "mean"), IGD_plus_sd=("IGD_plus", "std"),
    LOO_IGD_mean=("LOO_IGD", "mean"), LOO_IGD_sd=("LOO_IGD", "std"), n=("run", "count")
).to_csv(OUT / "igd_plus_loo_summary.csv", index=False)

# 13. Final-front size and Schott spacing (nearest-neighbour L1 distance SD).
front_rows = []
for (method, run), g in trials.groupby(["method", "run"]):
    vals = g[OBJ].to_numpy()
    f = vals[nondominated(vals)]
    if len(f) > 1:
        distance = np.abs(f[:, None, :] - f[None, :, :]).sum(axis=2)
        np.fill_diagonal(distance, np.inf)
        nearest = distance.min(axis=1)
        spacing = np.sqrt(np.square(nearest-nearest.mean()).sum()/(len(f)-1))
    else:
        spacing = 0.0
    front_rows.append(dict(method=method, run=run, nondominated_front_size=len(f),
                           schott_spacing=spacing))
fronts = pd.DataFrame(front_rows)
fronts.to_csv(OUT / "front_size_spacing_by_run.csv", index=False)
fronts.groupby("method", as_index=False).agg(
    front_size_mean=("nondominated_front_size", "mean"),
    front_size_sd=("nondominated_front_size", "std"),
    spacing_mean=("schott_spacing", "mean"), spacing_sd=("schott_spacing", "std"), n=("run", "count")
).to_csv(OUT / "front_size_spacing_summary.csv", index=False)
tests(fronts, ["nondominated_front_size", "schott_spacing"]).to_csv(
    OUT / "front_size_spacing_tests.csv", index=False)

# 14. Pooled Pareto source data.
pooled_idx = nondominated(norm_obj)
pooled = trials.iloc[pooled_idx].copy()
pooled["normalized_objectives"] = [json.dumps([float(v) for v in row], separators=(",", ":"))
                                    for row in pooled[OBJ].to_numpy()]
pooled[["method", "run", "evaluation", "candidate_id", "validation_macro_MAPE",
        "parameters", "latency_ms", "normalized_objectives"]].to_csv(
            OUT / "pooled_pareto_tidy.csv", index=False)

# 15. Selection sensitivity.
screen = pd.read_csv(AUDIT / "candidate_rescreen_summary.csv")
implemented = pd.read_csv(AUDIT / "selected_finalists.csv")
rules = []
for method in METHODS:
    d = screen[screen.method == method]
    choices = {
        "minimum_rescreened_mean_MAPE": d.sort_values(["mape_mean", "parameters", "latency_ms"]).iloc[0],
        "minimum_MAPE_parameters_le_100000": d[d.parameters <= 100000].sort_values(
            ["mape_mean", "parameters", "latency_ms"]).iloc[0],
        "implemented_robust_knee": implemented[implemented.method == method].iloc[0],
    }
    for rule, row in choices.items():
        rules.append(dict(method=method, selection_rule=rule, candidate_id=row.candidate_id,
                          L=row.L, hidden=row.hidden, layers=row.layers, H=row.H, lr=row.lr,
                          mape_mean=row.mape_mean, mape_std=row.mape_std,
                          parameters=row.parameters, latency_ms=row.latency_ms,
                          robust_knee_distance=row.robust_knee_distance))
pd.DataFrame(rules).to_csv(OUT / "selection_sensitivity.csv", index=False)

# 16. Finalist variance.
final = pd.read_csv(AUDIT / "final_ten_seed_raw.csv")
variance = final.groupby("configuration", as_index=False).agg(
    n=("macro_MAPE", "count"), sd=("macro_MAPE", "std"),
    minimum=("macro_MAPE", "min"), maximum=("macro_MAPE", "max"))
variance["IQR"] = variance.configuration.map(
    lambda name: float(iqr(final.loc[final.configuration == name, "macro_MAPE"], rng=(25, 75))))
variance["range"] = variance.maximum-variance.minimum
variance.to_csv(OUT / "finalist_seed_variance.csv", index=False)
groups = [final.loc[final.configuration == name, "macro_MAPE"].to_numpy()
          for name in sorted(final.configuration.unique())]
lev_stat, lev_p = levene(*groups, center="mean")
bf_stat, bf_p = levene(*groups, center="median")
pd.DataFrame([
    dict(test="Levene", center="mean", statistic=lev_stat, p_value=lev_p),
    dict(test="Brown-Forsythe", center="median", statistic=bf_stat, p_value=bf_p),
]).to_csv(OUT / "finalist_variance_tests.csv", index=False)

# 22. Matched RNG provenance and initial-population identity.
seed_check = []
base = trials[trials.method == "NSGA-II"]
fields = ["candidate_id", "L", "hidden", "layers", "H", "lr"]
for run in range(1, 16):
    ref = base[(base.run == run) & (base.generation == 0)].sort_values("evaluation")[fields].reset_index(drop=True)
    for method in METHODS:
        d = trials[(trials.method == method) & (trials.run == run) & (trials.generation == 0)].sort_values("evaluation")
        cand = d[fields].reset_index(drop=True)
        # CSV-parsed numeric arrays use canonical little-endian bytes for a strict check.
        numeric_fields = ["L", "hidden", "layers", "H", "lr"]
        byte_equal = (cand[numeric_fields].to_numpy(dtype="<f8").tobytes() ==
                      ref[numeric_fields].to_numpy(dtype="<f8").tobytes())
        seed_check.append(dict(run=run, method=method,
                               run_seed=int(d.run.iloc[0] + 20270000),
                               training_seed=int(d.training_seed.iloc[0]),
                               initial_population_n=len(cand),
                               candidate_ids_identical=cand.candidate_id.equals(ref.candidate_id),
                               canonical_numeric_bytes_identical=byte_equal,
                               all_training_seeds_identical_within_run=(d.training_seed.nunique() == 1)))
pd.DataFrame(seed_check).to_csv(OUT / "random_seed_initial_population_check.csv", index=False)

# 23. Exactly recorded model-training seconds (not total wall-clock/GPU-hours).
compute_rows = []
for method, group in trials.groupby("method"):
    compute_rows.append(dict(stage=f"search_{method}", records=len(group),
                             recorded_train_seconds=group.train_seconds.sum(),
                             recorded_train_hours=group.train_seconds.sum()/3600,
                             source=str(AUDIT / "all_optimizer_trials.csv")))
for stage, frame, source in (
    ("noise_floor", noise, RECEIVED / "nsga2" / "noise_floor_raw.csv"),
    ("candidate_rescreen", rescreen, AUDIT / "candidate_rescreen_raw.csv"),
    ("final_confirmation", final, AUDIT / "final_ten_seed_raw.csv"),
):
    compute_rows.append(dict(stage=stage, records=len(frame),
                             recorded_train_seconds=frame.train_seconds.sum(),
                             recorded_train_hours=frame.train_seconds.sum()/3600,
                             source=str(source)))
compute = pd.DataFrame(compute_rows)
compute.loc[len(compute)] = dict(stage="TOTAL_RECORDED_MODEL_TRAINING", records=compute.records.sum(),
                                 recorded_train_seconds=compute.recorded_train_seconds.sum(),
                                 recorded_train_hours=compute.recorded_train_hours.sum(),
                                 source="sum of preceding rows; excludes orchestration/analysis/I-O")
compute.to_csv(OUT / "recorded_compute_time_components.csv", index=False)

# Descriptive processed-table encoding checks used by Part A.
processed = pd.read_csv(ROOT.parent / "07_round2_revision" / "inputs" / "phase1_cv_all_rows.csv")
enc_rows = []
for field in ("k_exp", "cycle", "checkup_idx", "soc_window", "age_type", "stress", "Q0", "Re0", "Rct0"):
    s = processed[field]
    enc_rows.append(dict(field=field, n=len(s), missing=int(s.isna().sum()), unique=s.nunique(dropna=True),
                         minimum=s.min(), maximum=s.max()))
pd.DataFrame(enc_rows).to_csv(OUT / "processed_feature_ranges.csv", index=False)
processed.soc_window.value_counts(dropna=False).sort_index().rename_axis("soc_window").reset_index(
    name="row_count").to_csv(OUT / "soc_window_value_counts.csv", index=False)
processed.age_type.value_counts(dropna=False).sort_index().rename_axis("age_type").reset_index(
    name="row_count").to_csv(OUT / "age_type_value_counts.csv", index=False)
pd.DataFrame([{
    "rows": len(processed),
    "k_exp_equals_cycle_all_rows": bool(np.array_equal(processed.k_exp.to_numpy(), processed.cycle.to_numpy())),
    "k_exp_equals_checkup_idx_all_rows": bool(np.array_equal(processed.k_exp.to_numpy(), processed.checkup_idx.to_numpy())),
    "cyc_charged_unique": processed.cyc_charged.nunique(),
    "cyc_charged_min": processed.cyc_charged.min(),
    "cyc_charged_max": processed.cyc_charged.max(),
}]).to_csv(OUT / "k_exp_identity_check.csv", index=False)

print(f"Wrote {len(list(OUT.glob('*.csv')))} gap-fill CSVs to {OUT}")
