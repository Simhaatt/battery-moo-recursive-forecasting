"""Merge the three ASC optimizer runs, compute indicators/statistics, and confirm finalists."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wilcoxon


HERE = Path(__file__).resolve().parent
CORE = HERE / "round2_core"
if not CORE.exists():
    CORE = HERE.parent / "07_round2_revision"
EXTENSION = HERE if (HERE / "run_softcomputing_search.py").exists() else HERE.parent / "08_softcomputing_extension"
sys.path.insert(0, str(CORE)); sys.path.insert(0, str(EXTENSION))
import run_metaheuristic_search as meta_search  # noqa: E402
import run_softcomputing_search as base  # noqa: E402
from run_metaheuristic_search import Candidate, FULL9, metrics_2d, train_candidate  # noqa: E402
from run_round2_experiments import load_phase1_split  # noqa: E402

meta_search.rollout_predict = base.rollout_predict_common

METHODS = ("NSGA-II", "NSGA-III", "Random")
CHECKPOINTS = tuple(range(28, 281, 28))
FINAL_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
MANUAL = Candidate(20, 192, 2, 8, 1e-3)


def nondominated(values: np.ndarray) -> np.ndarray:
    """Return the minimization front using bounded-memory vectorized blocks.

    The original educational sorter is convenient for populations of 28, but
    its Python-level O(n^2) loop is unnecessarily slow for the pooled 12,600
    search evaluations used here.
    """
    values = np.asarray(values, dtype=float)
    dominated = np.zeros(len(values), dtype=bool)
    block_size = 128
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        no_worse = np.all(values[None, :, :] <= block[:, None, :], axis=2)
        strictly_better = np.any(values[None, :, :] < block[:, None, :], axis=2)
        dominated[start : start + len(block)] = np.any(no_worse & strictly_better, axis=1)
    return np.flatnonzero(~dominated)


def transform(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack((frame.validation_macro_MAPE, np.log10(frame.parameters), np.log10(frame.latency_ms)))


def hv2(points: np.ndarray, reference: np.ndarray) -> float:
    if not len(points): return 0.0
    ys = sorted(set(points[:, 0].tolist())) + [float(reference[0])]
    area = 0.0
    for left, right in zip(ys[:-1], ys[1:]):
        active = points[points[:, 0] <= left]
        if len(active): area += max(0.0, right-left) * max(0.0, reference[1]-active[:, 1].min())
    return float(area)


def hypervolume3(points: np.ndarray, reference=np.array([1.1, 1.1, 1.1])) -> float:
    points = points[np.all(points < reference, axis=1)]
    if not len(points): return 0.0
    points = points[nondominated(points)]
    xs = sorted(set(points[:, 0].tolist())) + [float(reference[0])]
    volume = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        active = points[points[:, 0] <= left][:, 1:]
        volume += max(0.0, right-left) * hv2(active, reference[1:])
    return float(volume)


def igd(points: np.ndarray, reference_front: np.ndarray) -> float:
    front = points[nondominated(points)]
    distances = np.linalg.norm(reference_front[:, None, :] - front[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())


def holm(pairs: list[dict], key="p_raw") -> list[dict]:
    ordered = sorted(range(len(pairs)), key=lambda i: pairs[i][key])
    adjusted = np.zeros(len(pairs))
    running = 0.0
    m = len(pairs)
    for rank, idx in enumerate(ordered):
        running = max(running, (m-rank) * pairs[idx][key])
        adjusted[idx] = min(1.0, running)
    for i, value in enumerate(adjusted): pairs[i]["p_holm"] = value
    return pairs


def a12(x, y) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + .5*(x[:, None] == y[None, :]).sum()) / (len(x)*len(y)))


def bootstrap_ci(values, seed=20260821, n=10000):
    values = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def locate_trials(root: Path, expected_method: str) -> pd.DataFrame:
    candidates = list(root.rglob("all_trials.csv"))
    for candidate in candidates:
        frame = pd.read_csv(candidate)
        method_col = "method" if "method" in frame else "algorithm"
        if len(frame) and str(frame[method_col].iloc[0]) == expected_method:
            if method_col != "method": frame = frame.rename(columns={method_col: "method"})
            return frame
    files = list(root.rglob("run_*_trials.csv"))
    frames = [pd.read_csv(p) for p in files]
    for frame in frames:
        if "algorithm" in frame: frame.rename(columns={"algorithm": "method"}, inplace=True)
    selected = [f for f in frames if len(f) and f.method.iloc[0] == expected_method]
    if selected: return pd.concat(selected, ignore_index=True)
    raise FileNotFoundError(f"Could not find {expected_method} trials below {root}")


def indicators(all_trials: pd.DataFrame, output: Path, smoke: bool):
    raw = transform(all_trials)
    lower, upper = raw.min(0), raw.max(0)
    normalized = (raw-lower) / np.maximum(upper-lower, 1e-12)
    all_trials = all_trials.copy()
    all_trials[["norm_mape", "norm_log_params", "norm_log_latency"]] = normalized
    ref_front = normalized[nondominated(normalized)]
    rows = []
    for (method, run), group in all_trials.groupby(["method", "run"], sort=False):
        group = group.sort_values("evaluation")
        checkpoints = [len(group)] if smoke else CHECKPOINTS
        for count in checkpoints:
            subset = group[group.evaluation <= count][["norm_mape", "norm_log_params", "norm_log_latency"]].to_numpy()
            rows.append({"method": method, "run": run, "evaluations": count,
                         "hypervolume": hypervolume3(subset), "IGD": igd(subset, ref_front)})
    trajectory = pd.DataFrame(rows)
    trajectory.to_csv(output / "hv_igd_by_run_and_evaluation.csv", index=False)
    summary = trajectory.groupby(["method", "evaluations"]).agg(
        hv_mean=("hypervolume", "mean"), hv_std=("hypervolume", "std"),
        igd_mean=("IGD", "mean"), igd_std=("IGD", "std"), n=("run", "count")).reset_index()
    summary.to_csv(output / "hv_igd_convergence_summary.csv", index=False)
    (output / "global_normalization.json").write_text(json.dumps({"columns": ["MAPE", "log10_parameters", "log10_latency"],
        "lower": lower.tolist(), "upper": upper.tolist(), "hv_reference": [1.1]*3,
        "igd_reference_front_size": len(ref_front)}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in METHODS:
        s = summary[summary.method == method]
        axes[0].plot(s.evaluations, s.hv_mean, marker="o", label=method)
        axes[0].fill_between(s.evaluations, s.hv_mean-s.hv_std.fillna(0), s.hv_mean+s.hv_std.fillna(0), alpha=.14)
        axes[1].plot(s.evaluations, s.igd_mean, marker="o", label=method)
        axes[1].fill_between(s.evaluations, s.igd_mean-s.igd_std.fillna(0), s.igd_mean+s.igd_std.fillna(0), alpha=.14)
    axes[0].set(xlabel="Unique trained configurations", ylabel="Hypervolume", title="Higher is better")
    axes[1].set(xlabel="Unique trained configurations", ylabel="IGD", title="Lower is better")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "optimizer_convergence.png", dpi=300); fig.savefig(output / "optimizer_convergence.pdf"); plt.close(fig)
    final = trajectory.sort_values("evaluations").groupby(["method", "run"]).tail(1)
    stats = []
    for metric in ("hypervolume", "IGD"):
        groups = [final[final.method == m][metric].to_numpy() for m in METHODS]
        h, p = kruskal(*groups)
        stats.append({"metric": metric, "test": "Kruskal-Wallis", "comparison": "all", "statistic": h, "p_raw": p, "p_holm": p})
        pairs = []
        for a, b in combinations(METHODS, 2):
            x, y = final[final.method == a][metric], final[final.method == b][metric]
            u, pv = mannwhitneyu(x, y, alternative="two-sided")
            pairs.append({"metric": metric, "test": "Mann-Whitney U", "comparison": f"{a} vs {b}",
                          "statistic": u, "p_raw": pv, "A12_first_greater": a12(x, y)})
        stats.extend(holm(pairs))
    pd.DataFrame(stats).to_csv(output / "optimizer_statistical_tests.csv", index=False)
    return all_trials, final


def top_candidates(all_trials: pd.DataFrame, per_method: int = 5) -> pd.DataFrame:
    selected = []
    for method, group in all_trials.groupby("method"):
        values = group[["norm_mape", "norm_log_params", "norm_log_latency"]].to_numpy()
        group = group.copy(); group["rank"] = 1
        group.iloc[nondominated(values), group.columns.get_loc("rank")] = 0
        group["ideal_distance"] = np.linalg.norm(values, axis=1)
        candidates = pd.concat([group[group["rank"] == 0].sort_values("ideal_distance"), group.sort_values("ideal_distance")])
        selected.append(candidates.drop_duplicates("candidate_id").head(per_method))
    return pd.concat(selected, ignore_index=True)


def as_candidate(row) -> Candidate:
    return Candidate(int(row.L), int(row.hidden), int(row.layers), int(row.H), float(row.lr))


def select_robust(train, validation, candidates: pd.DataFrame, output: Path, smoke: bool):
    path = output / "candidate_rescreen_raw.csv"
    seeds = (42,) if smoke else (42, 52, 62)
    schedule = {"teacher_epochs": 2, "teacher_patience": 1, "rollout_epochs": 1, "rollout_patience": 1} if smoke else {
        "teacher_epochs": 160, "teacher_patience": 30, "rollout_epochs": 30, "rollout_patience": 10}
    rows = []
    for row in candidates.itertuples(index=False):
        c = as_candidate(row)
        for seed in seeds:
            model, _, _, _, mape, seconds = train_candidate(train, validation, c, seed=seed, **schedule)
            rows.append({"method": row.method, "candidate_id": row.candidate_id, **asdict(c), "seed": seed,
                         "validation_macro_MAPE": mape, "parameters": base.count_parameters(model),
                         "latency_ms": base.forward_latency_ms(model, c.L, len(FULL9)+2), "train_seconds": seconds})
            pd.DataFrame(rows).to_csv(path, index=False)
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["method", "candidate_id", "L", "hidden", "layers", "H", "lr"], as_index=False).agg(
        mape_mean=("validation_macro_MAPE", "mean"), mape_std=("validation_macro_MAPE", "std"),
        parameters=("parameters", "mean"), latency_ms=("latency_ms", "mean"))
    z = np.column_stack((summary.mape_mean, np.log10(summary.parameters), np.log10(summary.latency_ms)))
    z = (z-z.min(0))/np.maximum(z.max(0)-z.min(0), 1e-12)
    summary["robust_knee_distance"] = np.linalg.norm(z, axis=1)
    winners = summary.sort_values("robust_knee_distance").groupby("method", as_index=False).head(1)
    summary.to_csv(output / "candidate_rescreen_summary.csv", index=False)
    winners.to_csv(output / "selected_finalists.csv", index=False)
    return winners


def final_evaluation(train, validation, test, winners: pd.DataFrame, output: Path, smoke: bool):
    seeds = FINAL_SEEDS[:2] if smoke else FINAL_SEEDS
    schedule = {"teacher_epochs": 2, "teacher_patience": 1, "rollout_epochs": 1, "rollout_patience": 1} if smoke else {
        "teacher_epochs": 220, "teacher_patience": 40, "rollout_epochs": 40, "rollout_patience": 15}
    configs = [("Manual", MANUAL)] + [(row.method, as_candidate(row)) for row in winners.itertuples(index=False)]
    rows, predictions = [], []
    for label, c in configs:
        for seed in seeds:
            model, sx, sy, _, val_mape, seconds = train_candidate(train, validation, c, seed=seed, **schedule)
            yt, yp, metadata, infer_seconds = base.rollout_predict_common(model, sx, sy, test, FULL9, c.L)
            result = {"configuration": label, "seed": seed, **asdict(c), "validation_macro_MAPE": val_mape,
                      **metrics_2d(yt, yp), "parameters": base.count_parameters(model),
                      "latency_ms": base.forward_latency_ms(model, c.L, len(FULL9)+2), "train_seconds": seconds,
                      "inference_ms_per_trajectory": infer_seconds/max(metadata.cell_id.nunique(), 1)*1000}
            rows.append(result)
            pred = metadata.copy(); pred[["Q_true", "Re_true"]] = yt; pred[["Q_pred", "Re_pred"]] = yp
            pred["configuration"] = label; pred["seed"] = seed; predictions.append(pred)
            pd.DataFrame(rows).to_csv(output / "final_ten_seed_raw.csv", index=False)
    raw = pd.DataFrame(rows); pd.concat(predictions, ignore_index=True).to_csv(output / "final_predictions.csv", index=False)
    metrics = ["macro_MAPE", "Q_MAPE", "Re_MAPE", "parameters", "latency_ms", "train_seconds"]
    summary_rows = []
    for name, group in raw.groupby("configuration", sort=False):
        row = {"configuration": name, "n": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean(); row[f"{metric}_std"] = group[metric].std()
            row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = bootstrap_ci(group[metric])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output / "final_ten_seed_summary.csv", index=False)
    comparisons = []
    manual = raw[raw.configuration == "Manual"].sort_values("seed")
    for name in raw.configuration.unique():
        if name == "Manual": continue
        other = raw[raw.configuration == name].sort_values("seed")
        for metric in ("macro_MAPE", "Q_MAPE", "Re_MAPE"):
            diff = other[metric].to_numpy()-manual[metric].to_numpy()
            statistic, p = wilcoxon(diff, alternative="two-sided")
            lo, hi = bootstrap_ci(diff)
            comparisons.append({"comparison": f"{name} - Manual", "metric": metric, "mean_difference": diff.mean(),
                                "difference_ci95_low": lo, "difference_ci95_high": hi, "wilcoxon_statistic": statistic,
                                "p_raw": p, "practically_equivalent_within_0.20pp": bool(lo > -.20 and hi < .20)})
    pd.DataFrame(holm(comparisons)).to_csv(output / "final_paired_tests_holm.csv", index=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--nsga2", type=Path, required=True); p.add_argument("--nsga3", type=Path, required=True)
    p.add_argument("--random", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--smoke", action="store_true"); args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frames = [locate_trials(args.nsga2, "NSGA-II"), locate_trials(args.nsga3, "NSGA-III"), locate_trials(args.random, "Random")]
    all_trials = pd.concat(frames, ignore_index=True); all_trials.to_csv(args.output / "all_optimizer_trials.csv", index=False)
    if not args.smoke:
        counts = all_trials.groupby(["method", "run"]).size()
        if len(counts) != 45 or not (counts == 280).all(): raise RuntimeError(f"Expected 45 complete runs x 280 evaluations; got\n{counts}")
    normalized, _ = indicators(all_trials, args.output, args.smoke)
    candidates = top_candidates(normalized, 2 if args.smoke else 5); candidates.to_csv(args.output / "rescreen_candidates.csv", index=False)
    train, validation, test = load_phase1_split(); winners = select_robust(train, validation, candidates, args.output, args.smoke)
    final_evaluation(train, validation, test, winners, args.output, args.smoke)
    shutil.make_archive(str(args.archive.with_suffix("")), "zip", args.output)
    print("COMPLETE", args.archive)


if __name__ == "__main__": main()
