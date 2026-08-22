"""Repeated matched-budget optimizer experiments for the ASC revision.

Each full run evaluates 280 unique rollout-LSTM configurations (population 28,
ten generations including the initial population).  The objectives are
validation rollout macro-MAPE, trainable parameters, and batch-one latency.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
CORE = HERE / "round2_core"
if not CORE.exists():
    CORE = HERE.parent / "07_round2_revision"
EXTENSION = HERE if (HERE / "run_softcomputing_search.py").exists() else HERE.parent / "08_softcomputing_extension"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(EXTENSION))

import run_metaheuristic_search as meta_search  # noqa: E402
import run_softcomputing_search as base  # noqa: E402
from run_metaheuristic_search import Candidate, FULL9, train_candidate  # noqa: E402
from run_round2_experiments import DEVICE, load_phase1_split  # noqa: E402


# Force the corrected identical-row validation protocol for every sequence length.
meta_search.rollout_predict = base.rollout_predict_common

POPULATION = 28
GENERATIONS = 10  # generation 0 plus nine offspring generations
EVALUATIONS = POPULATION * GENERATIONS
RUNS = 15
RUN_SEEDS = tuple(20270000 + i for i in range(1, RUNS + 1))
TRAIN_SEEDS = tuple(41000 + i for i in range(1, RUNS + 1))
SEARCH_SCHEDULE = {
    "teacher_epochs": 80,
    "teacher_patience": 18,
    "rollout_epochs": 15,
    "rollout_patience": 5,
}
NOISE_SEEDS = (42, 52, 62, 72, 82, 92, 102, 112, 122, 132)
NOISE_PANEL = (
    Candidate(20, 192, 2, 8, 1.0e-3),
    Candidate(20, 96, 1, 3, 4.20925e-4),
    Candidate(15, 96, 1, 5, 5.16493e-4),
    Candidate(10, 64, 1, 3, 7.0e-4),
    Candidate(10, 192, 3, 10, 1.0e-3),
    Candidate(10, 128, 2, 15, 7.0e-4),
    Candidate(15, 64, 2, 10, 1.2e-3),
    Candidate(20, 160, 3, 5, 2.0e-4),
)


def repair(candidate: Candidate) -> tuple[Candidate, bool]:
    """Repair L+H violations using the closest allowed feasible horizon."""
    if candidate.L + candidate.H <= base.MAX_TRAIN_TRAJECTORY_LENGTH:
        return candidate, False
    feasible = [h for h in base.HORIZON_VALUES if candidate.L + h <= base.MAX_TRAIN_TRAJECTORY_LENGTH]
    replacement = min(feasible, key=lambda h: (abs(h - candidate.H), -h))
    return Candidate(candidate.L, candidate.hidden, candidate.layers, replacement, candidate.lr), True


def mate(a: Candidate, b: Candidate, rng: np.random.Generator) -> tuple[Candidate, bool]:
    values = {
        name: getattr(a if rng.random() < 0.5 else b, name)
        for name in ("L", "hidden", "layers", "H", "lr")
    }
    pools = {
        "L": base.L_VALUES,
        "hidden": base.HIDDEN_VALUES,
        "layers": base.LAYER_VALUES,
        "H": base.HORIZON_VALUES,
    }
    changed = False
    for name, pool in pools.items():
        if rng.random() < 0.25:
            values[name] = int(rng.choice([v for v in pool if v != values[name]]))
            changed = True
    if rng.random() < 0.25:
        values["lr"] = float(10 ** np.clip(math.log10(values["lr"]) + rng.normal(0, 0.28), *base.LOG_LR_BOUNDS))
        changed = True
    if not changed:
        name = str(rng.choice(["L", "hidden", "layers", "H", "lr"]))
        if name == "lr":
            values[name] = float(10 ** rng.uniform(*base.LOG_LR_BOUNDS))
        else:
            values[name] = int(rng.choice([v for v in pools[name] if v != values[name]]))
    return repair(Candidate(**values))


def directions_h6() -> np.ndarray:
    directions = []
    for i in range(7):
        for j in range(7 - i):
            k = 6 - i - j
            directions.append((i / 6, j / 6, k / 6))
    return np.asarray(directions, dtype=float)


def associate(values: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = values.min(axis=0), values.max(axis=0)
    normalized = (values - low) / np.maximum(high - low, 1e-12)
    d = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    projection = normalized @ d.T
    residual = normalized[:, None, :] - projection[:, :, None] * d[None, :, :]
    distances = np.linalg.norm(residual, axis=2)
    niches = distances.argmin(axis=1)
    return niches, distances[np.arange(len(values)), niches]


def nsga3_selection(records: list[dict], size: int, rng: np.random.Generator) -> list[dict]:
    values = base.objective_array(records)
    fronts, _ = base.nondominated_sort(values)
    selected: list[int] = []
    split: list[int] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            split = front
            break
    if len(selected) == size:
        return [records[i] for i in selected]
    dirs = directions_h6()
    niches, distances = associate(values, dirs)
    counts = np.bincount(niches[selected], minlength=len(dirs)) if selected else np.zeros(len(dirs), dtype=int)
    available = set(split)
    while len(selected) < size and available:
        active_dirs = sorted({int(niches[i]) for i in available}, key=lambda n: (counts[n], n))
        minimum = counts[active_dirs[0]]
        tied = [n for n in active_dirs if counts[n] == minimum]
        niche = int(rng.choice(tied))
        candidates = [i for i in available if niches[i] == niche]
        chosen = min(candidates, key=lambda i: distances[i]) if counts[niche] == 0 else int(rng.choice(candidates))
        selected.append(chosen)
        available.remove(chosen)
        counts[niche] += 1
    return [records[i] for i in selected]


class RunEvaluator:
    def __init__(self, train, validation, method: str, run: int, train_seed: int, schedule: dict):
        self.train, self.validation = train, validation
        self.method, self.run, self.train_seed, self.schedule = method, run, train_seed, schedule
        self.records: list[dict] = []

    def evaluate(self, candidate: Candidate, generation: int, repaired: bool = False) -> dict:
        model, _, _, _, mape, seconds = train_candidate(
            self.train, self.validation, candidate, seed=self.train_seed,
            teacher_epochs=self.schedule["teacher_epochs"], teacher_patience=self.schedule["teacher_patience"],
            rollout_epochs=self.schedule["rollout_epochs"], rollout_patience=self.schedule["rollout_patience"],
        )
        row = {
            "method": self.method, "run": self.run, "evaluation": len(self.records) + 1,
            "generation": generation, "candidate_id": base.candidate_id(candidate), **asdict(candidate),
            "validation_macro_MAPE": float(mape), "parameters": base.count_parameters(model),
            "latency_ms": base.forward_latency_ms(model, candidate.L, len(FULL9) + 2),
            "train_seconds": float(seconds), "repaired": bool(repaired), "training_seed": self.train_seed,
        }
        self.records.append(row)
        print(f"{self.method} run={self.run:02d} eval={row['evaluation']:03d}/{EVALUATIONS} "
              f"MAPE={mape:.4f} params={row['parameters']:,} latency={row['latency_ms']:.4f}", flush=True)
        return row


def unique_random(rng, seen) -> Candidate:
    candidate = base.random_candidate(rng)
    while candidate.key in seen:
        candidate = base.random_candidate(rng)
    return candidate


def run_evolution(method, train, validation, run, run_seed, train_seed, pop_size, generations, schedule):
    rng = np.random.default_rng(run_seed)
    ev = RunEvaluator(train, validation, method, run, train_seed, schedule)
    seen: set[tuple] = set()
    population = []
    for _ in range(pop_size):
        c = unique_random(rng, seen); seen.add(c.key)
        population.append(ev.evaluate(c, 0))
    for generation in range(1, generations):
        offspring = []
        while len(offspring) < pop_size:
            pa, pb = base.tournament(population, rng), base.tournament(population, rng)
            a = Candidate(**{k: pa[k] for k in ("L", "hidden", "layers", "H", "lr")})
            b = Candidate(**{k: pb[k] for k in ("L", "hidden", "layers", "H", "lr")})
            c, repaired = mate(a, b, rng)
            attempts = 0
            while c.key in seen and attempts < 100:
                c, repaired = mate(a, b, rng); attempts += 1
            if c.key in seen:
                c = unique_random(rng, seen); repaired = False
            seen.add(c.key)
            offspring.append(ev.evaluate(c, generation, repaired))
        combined = population + offspring
        population = base.environmental_selection(combined, pop_size) if method == "NSGA-II" else nsga3_selection(combined, pop_size, rng)
    return ev.records


def run_random(train, validation, run, run_seed, train_seed, evaluations, schedule):
    rng = np.random.default_rng(run_seed)
    ev = RunEvaluator(train, validation, "Random", run, train_seed, schedule)
    seen = set()
    while len(ev.records) < evaluations:
        c = unique_random(rng, seen); seen.add(c.key)
        ev.evaluate(c, (len(ev.records)) // POPULATION)
    return ev.records


def write_front(frame: pd.DataFrame, destination: Path):
    values = frame[["validation_macro_MAPE", "parameters", "latency_ms"]].to_numpy(float)
    _, rank = base.nondominated_sort(values)
    result = frame.copy(); result["pareto_rank"] = rank
    result[result["pareto_rank"].eq(0)].to_csv(destination, index=False)


def refresh_archive(output: Path, archive: Path):
    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = archive.with_suffix("")
    made = Path(shutil.make_archive(str(temp), "zip", output))
    if made != archive:
        made.replace(archive)


def noise_floor(train, validation, output: Path, smoke: bool):
    path = output / "noise_floor_raw.csv"
    panel = NOISE_PANEL[:2] if smoke else NOISE_PANEL
    seeds = NOISE_SEEDS[:2] if smoke else NOISE_SEEDS
    if path.exists() and len(pd.read_csv(path)) == len(panel) * len(seeds):
        return
    rows = []
    schedule = {"teacher_epochs": 2, "teacher_patience": 1, "rollout_epochs": 1, "rollout_patience": 1} if smoke else SEARCH_SCHEDULE
    for candidate in panel:
        for seed in seeds:
            model, _, _, _, mape, seconds = train_candidate(
                train, validation, candidate, seed=seed,
                teacher_epochs=schedule["teacher_epochs"], teacher_patience=schedule["teacher_patience"],
                rollout_epochs=schedule["rollout_epochs"], rollout_patience=schedule["rollout_patience"],
            )
            rows.append({"candidate_id": base.candidate_id(candidate), **asdict(candidate), "seed": seed,
                         "validation_macro_MAPE": mape, "parameters": base.count_parameters(model),
                         "latency_ms": base.forward_latency_ms(model, candidate.L, len(FULL9)+2),
                         "train_seconds": seconds})
            pd.DataFrame(rows).to_csv(path, index=False)
    raw = pd.DataFrame(rows)
    raw.groupby("candidate_id").agg(mape_mean=("validation_macro_MAPE", "mean"),
        mape_std=("validation_macro_MAPE", "std"), latency_mean=("latency_ms", "mean"),
        latency_std=("latency_ms", "std"), n=("seed", "count")).reset_index().to_csv(output / "noise_floor_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("nsga2", "nsga3", "random"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--noise-floor", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train, validation, _ = load_phase1_split()
    if args.noise_floor:
        noise_floor(train, validation, args.output, args.smoke)
        refresh_archive(args.output, args.archive)
    runs, pop, generations = (2, 6, 2) if args.smoke else (RUNS, POPULATION, GENERATIONS)
    schedule = {"teacher_epochs": 2, "teacher_patience": 1, "rollout_epochs": 1, "rollout_patience": 1} if args.smoke else SEARCH_SCHEDULE
    method_label = {"nsga2": "NSGA-II", "nsga3": "NSGA-III", "random": "Random"}[args.method]
    started = time.time()
    for run in range(1, runs + 1):
        trials_path = args.output / f"run_{run:02d}_trials.csv"
        expected = pop * generations
        if trials_path.exists() and len(pd.read_csv(trials_path)) == expected:
            print("Skipping complete", trials_path.name, flush=True); continue
        if args.method == "random":
            records = run_random(train, validation, run, RUN_SEEDS[run-1], TRAIN_SEEDS[run-1], expected, schedule)
        else:
            records = run_evolution(method_label, train, validation, run, RUN_SEEDS[run-1], TRAIN_SEEDS[run-1], pop, generations, schedule)
        frame = pd.DataFrame(records)
        frame.to_csv(trials_path, index=False)
        write_front(frame, args.output / f"run_{run:02d}_front.csv")
        status = {"method": method_label, "completed_runs": run, "requested_runs": runs,
                  "evaluations_per_run": expected, "elapsed_hours": (time.time()-started)/3600,
                  "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        (args.output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        refresh_archive(args.output, args.archive)
    all_trials = pd.concat([pd.read_csv(p) for p in sorted(args.output.glob("run_*_trials.csv"))], ignore_index=True)
    all_trials.to_csv(args.output / "all_trials.csv", index=False)
    manifest = {"method": method_label, "population": pop, "generations_including_initial": generations,
                "evaluations_per_run": pop*generations, "runs": runs, "run_seeds": RUN_SEEDS[:runs],
                "training_seeds": TRAIN_SEEDS[:runs], "schedule": schedule, "constraint": "L + H <= 29",
                "repair": "closest allowed feasible H; ties prefer larger H", "python": platform.python_version(),
                "torch": torch.__version__, "device": str(DEVICE)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    refresh_archive(args.output, args.archive)
    print("COMPLETE", args.archive, flush=True)


if __name__ == "__main__":
    main()
