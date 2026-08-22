"""Matched-budget multi-objective search for the rollout-tuned LSTM.

The experiment compares NSGA-II with random search under the same search
space, number of unique trained configurations, training schedule, validation
split, and seed. Three objectives are minimized: validation rollout macro
MAPE, trainable parameter count, and measured batch-1 forward latency.

The fixed test trajectories are touched only after stochastic re-screening has
selected one configuration from each search algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
LOCAL_CORE = HERE / "round2_core"
ROUND2_CORE = LOCAL_CORE if LOCAL_CORE.exists() else HERE.parent / "07_round2_revision"
sys.path.insert(0, str(ROUND2_CORE))

import run_metaheuristic_search as meta_search  # noqa: E402
from run_metaheuristic_search import (  # noqa: E402
    Candidate,
    FULL9,
    MANUAL as LEGACY_MANUAL,
    metrics_2d,
    train_candidate,
)
from run_round2_experiments import (  # noqa: E402
    DEVICE,
    SEEDS,
    load_phase1_split,
    set_seed,
)


OUTPUT = HERE / "outputs_softcomputing_search"
OUTPUT.mkdir(parents=True, exist_ok=True)

L_VALUES = (10, 15, 20)
HIDDEN_VALUES = (64, 96, 128, 160, 192)
LAYER_VALUES = (1, 2, 3)
HORIZON_VALUES = (3, 5, 8, 10, 15)
LOG_LR_BOUNDS = (-4.0, math.log10(2e-3))
MAX_TRAIN_TRAJECTORY_LENGTH = 29
COMMON_EVALUATION_START = 20
MANUAL = {**LEGACY_MANUAL, "H": 8}


def rollout_predict_common(model, scaler_x, scaler_y, frame, features, sequence_length=20):
    """Autoregressive prediction on identical rows for every candidate L."""
    model.eval()
    sequence_columns = features + ["Q", "Re"]
    truth, predictions, metadata = [], [], []
    inference_seconds = 0.0
    for cell_id, group0 in frame.groupby("cell_id", sort=False):
        group = group0.sort_values("k_exp").reset_index(drop=True)
        if len(group) <= COMMON_EVALUATION_START:
            continue
        buffer = group[sequence_columns].to_numpy(np.float32).copy()
        for index in range(COMMON_EVALUATION_START, len(group)):
            window = buffer[index - sequence_length : index]
            transformed = scaler_x.transform(window).reshape(
                1, sequence_length, len(sequence_columns)
            ).astype(np.float32)
            start = time.perf_counter()
            with torch.no_grad():
                prediction = model(torch.tensor(transformed, device=DEVICE)).cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            prediction = scaler_y.inverse_transform(prediction)[0]
            prediction = np.clip(prediction, [1e-6, 1e-9], None)
            truth.append(group.loc[index, ["Q", "Re"]].to_numpy(np.float32))
            predictions.append(prediction)
            metadata.append({"cell_id": cell_id, "k_exp": group.loc[index, "k_exp"]})
            buffer[index, len(features) : len(features) + 2] = prediction
    return np.asarray(truth), np.asarray(predictions), pd.DataFrame(metadata), inference_seconds
SEARCH_SEED = 20260820
SEARCH_TRAIN_SEED = 2026
RESCREEN_SEEDS = (42, 52, 62)


def candidate_id(candidate: Candidate) -> str:
    return (
        f"L{candidate.L}_D{candidate.hidden}_N{candidate.layers}_"
        f"H{candidate.H}_LR{candidate.lr:.8f}"
    )


def random_candidate(rng: np.random.Generator) -> Candidate:
    while True:
        candidate = Candidate(
            L=int(rng.choice(L_VALUES)),
            hidden=int(rng.choice(HIDDEN_VALUES)),
            layers=int(rng.choice(LAYER_VALUES)),
            H=int(rng.choice(HORIZON_VALUES)),
            lr=float(10 ** rng.uniform(*LOG_LR_BOUNDS)),
        )
        if candidate.L + candidate.H <= MAX_TRAIN_TRAJECTORY_LENGTH:
            return candidate


def crossover_and_mutate(
    parent_a: Candidate, parent_b: Candidate, rng: np.random.Generator, mutation_probability: float = 0.25
) -> Candidate:
    values = {
        field: getattr(parent_a if rng.random() < 0.5 else parent_b, field)
        for field in ("L", "hidden", "layers", "H", "lr")
    }
    pools = {"L": L_VALUES, "hidden": HIDDEN_VALUES, "layers": LAYER_VALUES, "H": HORIZON_VALUES}
    mutated = False
    for field, pool in pools.items():
        if rng.random() < mutation_probability:
            choices = [value for value in pool if value != values[field]]
            values[field] = int(rng.choice(choices))
            mutated = True
    if rng.random() < mutation_probability:
        log_lr = np.clip(math.log10(values["lr"]) + rng.normal(0.0, 0.28), *LOG_LR_BOUNDS)
        values["lr"] = float(10**log_lr)
        mutated = True
    if not mutated:
        field = str(rng.choice(["L", "hidden", "layers", "H", "lr"]))
        if field == "lr":
            values["lr"] = float(10 ** rng.uniform(*LOG_LR_BOUNDS))
        else:
            choices = [value for value in pools[field] if value != values[field]]
            values[field] = int(rng.choice(choices))
    candidate = Candidate(**values)
    if candidate.L + candidate.H > MAX_TRAIN_TRAJECTORY_LENGTH:
        feasible_horizons = [value for value in HORIZON_VALUES if candidate.L + value <= MAX_TRAIN_TRAJECTORY_LENGTH]
        candidate = Candidate(candidate.L, candidate.hidden, candidate.layers, int(rng.choice(feasible_horizons)), candidate.lr)
    return candidate


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def forward_latency_ms(model: torch.nn.Module, sequence_length: int, input_dim: int, repeats: int = 100) -> float:
    model.eval()
    sample = torch.zeros((1, sequence_length, input_dim), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        block_times = []
        block_size = max(10, repeats // 5)
        for _ in range(5):
            if DEVICE.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(block_size):
                    model(sample)
                end.record()
                torch.cuda.synchronize()
                block_times.append(float(start.elapsed_time(end)) / block_size)
            else:
                start_time = time.perf_counter()
                for _ in range(block_size):
                    model(sample)
                block_times.append((time.perf_counter() - start_time) * 1000.0 / block_size)
    return float(np.median(block_times))


def objective_array(records: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["validation_macro_MAPE"], row["parameters"], row["latency_ms"]] for row in records],
        dtype=float,
    )


def dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))


def nondominated_sort(values: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    n = len(values)
    dominates_set = [set() for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(values[p], values[q]):
                dominates_set[p].add(q)
            elif dominates(values[q], values[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    rank = np.full(n, -1, dtype=int)
    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front = []
        for p in fronts[front_index]:
            rank[p] = front_index
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        front_index += 1
    return fronts, rank


def crowding_distance(values: np.ndarray, front: list[int]) -> dict[int, float]:
    distance = {index: 0.0 for index in front}
    if len(front) <= 2:
        return {index: float("inf") for index in front}
    for objective in range(values.shape[1]):
        ordered = sorted(front, key=lambda index: values[index, objective])
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        low, high = values[ordered[0], objective], values[ordered[-1], objective]
        if high <= low:
            continue
        for position in range(1, len(ordered) - 1):
            previous_value = values[ordered[position - 1], objective]
            next_value = values[ordered[position + 1], objective]
            distance[ordered[position]] += float((next_value - previous_value) / (high - low))
    return distance


def ranks_and_crowding(records: list[dict]) -> tuple[np.ndarray, dict[int, float]]:
    values = objective_array(records)
    fronts, ranks = nondominated_sort(values)
    crowding = {}
    for front in fronts:
        crowding.update(crowding_distance(values, front))
    return ranks, crowding


def environmental_selection(records: list[dict], population_size: int) -> list[dict]:
    values = objective_array(records)
    fronts, _ = nondominated_sort(values)
    selected: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
        else:
            distance = crowding_distance(values, front)
            remaining = population_size - len(selected)
            selected.extend(sorted(front, key=lambda index: distance[index], reverse=True)[:remaining])
            break
    return [records[index] for index in selected]


def tournament(records: list[dict], rng: np.random.Generator) -> dict:
    ranks, crowding = ranks_and_crowding(records)
    left, right = rng.choice(len(records), size=2, replace=False)
    if ranks[left] != ranks[right]:
        winner = left if ranks[left] < ranks[right] else right
    elif crowding[left] != crowding[right]:
        winner = left if crowding[left] > crowding[right] else right
    else:
        winner = int(rng.choice([left, right]))
    return records[winner]


class Evaluator:
    def __init__(self, train: pd.DataFrame, validation: pd.DataFrame, algorithm: str, schedule: dict):
        self.train = train
        self.validation = validation
        self.algorithm = algorithm
        self.schedule = schedule
        self.records: list[dict] = []

    def evaluate(self, candidate: Candidate, generation: int) -> dict:
        model, _, _, _, validation_mape, seconds = train_candidate(
            self.train,
            self.validation,
            candidate,
            seed=SEARCH_TRAIN_SEED,
            teacher_epochs=self.schedule["teacher_epochs"],
            teacher_patience=self.schedule["teacher_patience"],
            rollout_epochs=self.schedule["rollout_epochs"],
            rollout_patience=self.schedule["rollout_patience"],
        )
        row = {
            "algorithm": self.algorithm,
            "evaluation": len(self.records) + 1,
            "generation": generation,
            "candidate_id": candidate_id(candidate),
            **asdict(candidate),
            "validation_macro_MAPE": float(validation_mape),
            "parameters": count_parameters(model),
            "latency_ms": forward_latency_ms(model, candidate.L, len(FULL9) + 2),
            "train_seconds": float(seconds),
        }
        self.records.append(row)
        print(
            f"{self.algorithm:8s} eval={row['evaluation']:02d} gen={generation} "
            f"val={row['validation_macro_MAPE']:.4f}% params={row['parameters']:,} "
            f"lat={row['latency_ms']:.4f} ms {row['candidate_id']}",
            flush=True,
        )
        return row


def run_nsga2(train, validation, population_size: int, generations: int, schedule: dict) -> list[dict]:
    rng = np.random.default_rng(SEARCH_SEED)
    evaluator = Evaluator(train, validation, "NSGA-II", schedule)
    seen = set()
    population = []
    while len(population) < population_size:
        candidate = random_candidate(rng)
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        population.append(evaluator.evaluate(candidate, generation=0))

    for generation in range(1, generations + 1):
        offspring = []
        while len(offspring) < population_size:
            parent_a_row = tournament(population, rng)
            parent_b_row = tournament(population, rng)
            parent_a = Candidate(**{key: parent_a_row[key] for key in ("L", "hidden", "layers", "H", "lr")})
            parent_b = Candidate(**{key: parent_b_row[key] for key in ("L", "hidden", "layers", "H", "lr")})
            child = crossover_and_mutate(parent_a, parent_b, rng)
            attempts = 0
            while child.key in seen and attempts < 100:
                child = crossover_and_mutate(parent_a, parent_b, rng)
                attempts += 1
            if child.key in seen:
                child = random_candidate(rng)
                while child.key in seen:
                    child = random_candidate(rng)
            seen.add(child.key)
            offspring.append(evaluator.evaluate(child, generation=generation))
        population = environmental_selection(population + offspring, population_size)
    return evaluator.records


def run_random_search(train, validation, evaluations: int, schedule: dict) -> list[dict]:
    rng = np.random.default_rng(SEARCH_SEED + 1)
    evaluator = Evaluator(train, validation, "Random", schedule)
    seen = set()
    while len(evaluator.records) < evaluations:
        candidate = random_candidate(rng)
        if candidate.key in seen:
            continue
        seen.add(candidate.key)
        evaluator.evaluate(candidate, generation=0)
    return evaluator.records


def add_front_and_knee(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    values = result[["validation_macro_MAPE", "parameters", "latency_ms"]].to_numpy(float)
    _, rank = nondominated_sort(values)
    result["pareto_rank"] = rank

    transformed_reference = np.column_stack(
        [
            reference["validation_macro_MAPE"].to_numpy(float),
            np.log10(reference["parameters"].to_numpy(float)),
            np.log10(reference["latency_ms"].to_numpy(float)),
        ]
    )
    transformed = np.column_stack(
        [result["validation_macro_MAPE"], np.log10(result["parameters"]), np.log10(result["latency_ms"])]
    )
    lower, upper = transformed_reference.min(axis=0), transformed_reference.max(axis=0)
    normalized = (transformed - lower) / np.maximum(upper - lower, 1e-12)
    result["knee_distance"] = np.sqrt(np.square(normalized).sum(axis=1))
    return result


def choose_rescreen_candidates(search_results: pd.DataFrame, per_algorithm: int) -> pd.DataFrame:
    ranked_frames = []
    for algorithm, group in search_results.groupby("algorithm", sort=False):
        ranked = add_front_and_knee(group, search_results)
        front = ranked[ranked["pareto_rank"].eq(0)].sort_values("knee_distance")
        remainder = ranked[~ranked.index.isin(front.index)].sort_values("knee_distance")
        selected = pd.concat([front, remainder]).drop_duplicates("candidate_id").head(per_algorithm).copy()
        selected["rescreen_order"] = np.arange(1, len(selected) + 1)
        ranked_frames.append(selected)
    return pd.concat(ranked_frames, ignore_index=True)


def row_candidate(row) -> Candidate:
    return Candidate(int(row.L), int(row.hidden), int(row.layers), int(row.H), float(row.lr))


def run_rescreen(train, validation, selected: pd.DataFrame, seeds: tuple[int, ...], schedule: dict) -> pd.DataFrame:
    rows = []
    for row in selected.itertuples(index=False):
        candidate = row_candidate(row)
        for seed in seeds:
            model, _, _, _, validation_mape, seconds = train_candidate(
                train,
                validation,
                candidate,
                seed=seed,
                teacher_epochs=schedule["teacher_epochs"],
                teacher_patience=schedule["teacher_patience"],
                rollout_epochs=schedule["rollout_epochs"],
                rollout_patience=schedule["rollout_patience"],
            )
            rows.append(
                {
                    "algorithm": row.algorithm,
                    "candidate_id": row.candidate_id,
                    **asdict(candidate),
                    "seed": seed,
                    "validation_macro_MAPE": validation_mape,
                    "parameters": count_parameters(model),
                    "latency_ms": forward_latency_ms(model, candidate.L, len(FULL9) + 2),
                    "train_seconds": seconds,
                }
            )
            print(f"rescreen {row.algorithm} {row.candidate_id} seed={seed}: {validation_mape:.4f}%", flush=True)
    return pd.DataFrame(rows)


def choose_robust_final(rescreen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rescreen.groupby(["algorithm", "candidate_id", "L", "hidden", "layers", "H", "lr"], as_index=False)
        .agg(
            validation_macro_MAPE=("validation_macro_MAPE", "mean"),
            validation_macro_MAPE_std=("validation_macro_MAPE", "std"),
            parameters=("parameters", "first"),
            latency_ms=("latency_ms", "median"),
            train_seconds=("train_seconds", "sum"),
        )
    )
    ranked_frames, chosen = [], []
    for algorithm, group in summary.groupby("algorithm", sort=False):
        ranked = add_front_and_knee(group, summary)
        ranked_frames.append(ranked)
        selected = ranked[ranked["pareto_rank"].eq(0)].sort_values("knee_distance").iloc[0]
        chosen.append(selected)
    return pd.concat(ranked_frames, ignore_index=True), pd.DataFrame(chosen).reset_index(drop=True)


def run_final_test(train, validation, test, chosen: pd.DataFrame, seeds: tuple[int, ...], schedule: dict):
    configurations = [("manual", Candidate(**MANUAL))]
    for row in chosen.itertuples(index=False):
        configurations.append((f"{row.algorithm}_selected", row_candidate(row)))
    rows, histories, predictions = [], [], []
    for label, candidate in configurations:
        for seed in seeds:
            model, sc_x, sc_y, history, validation_mape, seconds = train_candidate(
                train,
                validation,
                candidate,
                seed=seed,
                teacher_epochs=schedule["teacher_epochs"],
                teacher_patience=schedule["teacher_patience"],
                rollout_epochs=schedule["rollout_epochs"],
                rollout_patience=schedule["rollout_patience"],
            )
            y_true, y_pred, metadata, inference_seconds = rollout_predict_common(
                model, sc_x, sc_y, test, FULL9, candidate.L
            )
            result = {
                "configuration": label,
                "seed": seed,
                **asdict(candidate),
                "validation_macro_MAPE": validation_mape,
                **metrics_2d(y_true, y_pred),
                "parameters": count_parameters(model),
                "train_seconds": seconds,
                "inference_ms_per_trajectory": inference_seconds / metadata["cell_id"].nunique() * 1000.0,
                "n_eval_rows": len(y_true),
                "n_eval_cells": int(metadata["cell_id"].nunique()),
            }
            rows.append(result)
            history["configuration"], history["seed"] = label, seed
            histories.append(history)
            prediction = metadata.copy()
            prediction["configuration"], prediction["seed"] = label, seed
            prediction[["Q_true", "Re_true"]] = y_true
            prediction[["Q_pred", "Re_pred"]] = y_pred
            predictions.append(prediction)
            print(f"FINAL {label} seed={seed}: test={result['macro_MAPE']:.4f}%", flush=True)
    return pd.DataFrame(rows), pd.concat(histories, ignore_index=True), pd.concat(predictions, ignore_index=True)


def paired_tests(final_results: pd.DataFrame) -> pd.DataFrame:
    pivot = final_results.pivot(index="seed", columns="configuration", values="macro_MAPE")
    rows = []
    for configuration in pivot.columns:
        if configuration == "manual":
            continue
        difference = pivot[configuration] - pivot["manual"]
        if len(difference) >= 5:
            statistic, p_value = wilcoxon(difference, alternative="two-sided", method="exact")
        else:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "comparison": f"{configuration} - manual",
                "mean_difference_pp": float(difference.mean()),
                "wilcoxon_W": statistic,
                "p_value_two_sided": p_value,
                "n_seeds": len(difference),
            }
        )
    return pd.DataFrame(rows)


def save_figures(search_results: pd.DataFrame, final_results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for algorithm, group in search_results.groupby("algorithm", sort=False):
        ordered = group.sort_values("evaluation")
        ax.step(ordered["evaluation"], ordered["validation_macro_MAPE"].cummin(), where="post", label=algorithm)
    ax.set_xlabel("Unique trained configurations")
    ax.set_ylabel("Best validation rollout macro MAPE (%)")
    ax.set_title("Matched-budget search convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_search_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_search_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for algorithm, group in search_results.groupby("algorithm", sort=False):
        ranked = add_front_and_knee(group, search_results)
        axes[0].scatter(group["parameters"], group["validation_macro_MAPE"], alpha=0.45, label=algorithm)
        front = ranked[ranked["pareto_rank"].eq(0)].sort_values("parameters")
        axes[0].plot(front["parameters"], front["validation_macro_MAPE"], marker="o")
        axes[1].scatter(group["latency_ms"], group["validation_macro_MAPE"], alpha=0.45, label=algorithm)
        front = ranked[ranked["pareto_rank"].eq(0)].sort_values("latency_ms")
        axes[1].plot(front["latency_ms"], front["validation_macro_MAPE"], marker="o")
    axes[0].set_xlabel("Trainable parameters")
    axes[1].set_xlabel("Batch-1 forward latency (ms)")
    for axis in axes:
        axis.set_ylabel("Validation rollout macro MAPE (%)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Three-objective search outcomes and nondominated fronts")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_search_pareto.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_search_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = final_results.groupby("configuration", as_index=False).agg(
        mean_macro_MAPE=("macro_MAPE", "mean"),
        std_macro_MAPE=("macro_MAPE", "std"),
        parameters=("parameters", "first"),
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(
        summary["parameters"], summary["mean_macro_MAPE"], yerr=summary["std_macro_MAPE"],
        fmt="o", capsize=4,
    )
    for row in summary.itertuples(index=False):
        ax.annotate(row.configuration.replace("_", " "), (row.parameters, row.mean_macro_MAPE), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Test rollout macro MAPE, mean ± SD (%)")
    ax.set_title("Accuracy–complexity trade-off after five-seed confirmation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_final_accuracy_complexity.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_final_accuracy_complexity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def schedules(smoke: bool) -> tuple[dict, dict, dict]:
    if smoke:
        return (
            {"teacher_epochs": 2, "teacher_patience": 2, "rollout_epochs": 1, "rollout_patience": 1},
            {"teacher_epochs": 3, "teacher_patience": 3, "rollout_epochs": 1, "rollout_patience": 1},
            {"teacher_epochs": 3, "teacher_patience": 3, "rollout_epochs": 1, "rollout_patience": 1},
        )
    return (
        {"teacher_epochs": 80, "teacher_patience": 18, "rollout_epochs": 15, "rollout_patience": 5},
        {"teacher_epochs": 160, "teacher_patience": 30, "rollout_epochs": 30, "rollout_patience": 10},
        {"teacher_epochs": 220, "teacher_patience": 40, "rollout_epochs": 40, "rollout_patience": 15},
    )


def main(smoke: bool = False) -> None:
    set_seed(SEARCH_SEED)
    train, validation, test = load_phase1_split()
    meta_search.rollout_predict = rollout_predict_common
    observed_maximum = int(train.groupby("cell_id").size().max())
    if observed_maximum != MAX_TRAIN_TRAJECTORY_LENGTH:
        raise RuntimeError(
            f"Expected maximum training trajectory length {MAX_TRAIN_TRAJECTORY_LENGTH}, found {observed_maximum}."
        )
    search_schedule, rescreen_schedule, final_schedule = schedules(smoke)
    population_size = 4 if smoke else 8
    generations = 1 if smoke else 3
    evaluation_budget = population_size * (generations + 1)
    rescreen_count = 2 if smoke else 5
    rescreen_seeds = (42,) if smoke else RESCREEN_SEEDS
    final_seeds = (42,) if smoke else tuple(SEEDS)

    total_start = time.perf_counter()
    nsga_records = run_nsga2(train, validation, population_size, generations, search_schedule)
    random_records = run_random_search(train, validation, evaluation_budget, search_schedule)
    search_results = pd.DataFrame(nsga_records + random_records)
    search_results.to_csv(OUTPUT / "search_trials.csv", index=False)

    ranked = pd.concat(
        [add_front_and_knee(group, search_results) for _, group in search_results.groupby("algorithm", sort=False)],
        ignore_index=True,
    )
    ranked.to_csv(OUTPUT / "search_trials_ranked.csv", index=False)
    selected = choose_rescreen_candidates(search_results, rescreen_count)
    selected.to_csv(OUTPUT / "rescreen_candidates.csv", index=False)

    rescreen = run_rescreen(train, validation, selected, rescreen_seeds, rescreen_schedule)
    rescreen.to_csv(OUTPUT / "rescreen_seed_metrics.csv", index=False)
    rescreen_summary, chosen = choose_robust_final(rescreen)
    rescreen_summary.to_csv(OUTPUT / "rescreen_summary.csv", index=False)
    chosen.to_csv(OUTPUT / "selected_configurations.csv", index=False)

    final_results, final_history, final_predictions = run_final_test(
        train, validation, test, chosen, final_seeds, final_schedule
    )
    final_results.to_csv(OUTPUT / "final_seed_metrics.csv", index=False)
    final_history.to_csv(OUTPUT / "final_training_history.csv", index=False)
    final_predictions.to_csv(OUTPUT / "final_predictions.csv", index=False)
    final_summary = final_results.groupby("configuration", as_index=False).agg(
        mean_macro_MAPE=("macro_MAPE", "mean"),
        std_macro_MAPE=("macro_MAPE", "std"),
        mean_Q_MAPE=("Q_MAPE", "mean"),
        std_Q_MAPE=("Q_MAPE", "std"),
        mean_Re_MAPE=("Re_MAPE", "mean"),
        std_Re_MAPE=("Re_MAPE", "std"),
        parameters=("parameters", "first"),
        mean_train_seconds=("train_seconds", "mean"),
        mean_inference_ms_per_trajectory=("inference_ms_per_trajectory", "mean"),
    )
    final_summary.to_csv(OUTPUT / "final_summary.csv", index=False)
    paired_tests(final_results).to_csv(OUTPUT / "final_paired_tests.csv", index=False)
    save_figures(search_results, final_results)

    environment = {
        "journal_target": "Soft Computing (Springer)",
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "search_seed": SEARCH_SEED,
        "search_training_seed": SEARCH_TRAIN_SEED,
        "objectives": ["validation rollout macro MAPE", "trainable parameters", "batch-1 forward latency (ms)"],
        "common_evaluation_start": COMMON_EVALUATION_START,
        "common_test_points": 76,
        "population_size": population_size,
        "offspring_generations": generations,
        "unique_evaluations_per_algorithm": evaluation_budget,
        "search_space": {
            "L": L_VALUES,
            "hidden": HIDDEN_VALUES,
            "layers": LAYER_VALUES,
            "H": HORIZON_VALUES,
            "learning_rate": [10 ** LOG_LR_BOUNDS[0], 10 ** LOG_LR_BOUNDS[1]],
            "feasibility_constraint": "L + H <= 29",
        },
        "legacy_manual_configuration": LEGACY_MANUAL,
        "feasible_manual_configuration": MANUAL,
        "search_schedule": search_schedule,
        "rescreen_schedule": rescreen_schedule,
        "final_schedule": final_schedule,
        "rescreen_seeds": rescreen_seeds,
        "final_seeds": final_seeds,
        "wall_seconds_total": time.perf_counter() - total_start,
        "smoke_test": smoke,
    }
    (OUTPUT / "experiment_manifest.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(final_summary.to_string(index=False))
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end validation")
    arguments = parser.parse_args()
    main(smoke=arguments.smoke)
