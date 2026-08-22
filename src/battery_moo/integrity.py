"""Artifact integrity checks independent of model retraining."""
from __future__ import annotations

import pandas as pd

ALLOWED_L = {10, 15, 20}
ALLOWED_HIDDEN = {64, 96, 128, 160, 192}
ALLOWED_LAYERS = {1, 2, 3}
ALLOWED_H = {3, 5, 8, 10, 15}
METHODS = {"NSGA-II", "NSGA-III", "Random"}


def validate_search_trials(df: pd.DataFrame) -> dict[str, bool]:
    checks = {
        "row_count_12600": len(df) == 12600,
        "three_methods": set(df["method"]) == METHODS,
        "fifteen_runs_each": df.groupby("method")["run"].nunique().eq(15).all(),
        "280_evaluations_per_run": df.groupby(["method", "run"]).size().eq(280).all(),
        "candidate_ids_unique_within_run": not df.duplicated(["method", "run", "evaluation"]).any(),
        "allowed_window_lengths": set(df["L"]).issubset(ALLOWED_L),
        "allowed_hidden_sizes": set(df["hidden"]).issubset(ALLOWED_HIDDEN),
        "allowed_layer_counts": set(df["layers"]).issubset(ALLOWED_LAYERS),
        "allowed_rollout_horizons": set(df["H"]).issubset(ALLOWED_H),
        "learning_rate_bounds": df["lr"].between(1e-4, 2e-3).all(),
        "window_horizon_constraint": (df["L"] + df["H"] <= 29).all(),
        "finite_objectives": df[["validation_macro_MAPE", "parameters", "latency_ms"]].notna().all().all(),
        "positive_objectives": (df[["validation_macro_MAPE", "parameters", "latency_ms"]] > 0).all().all(),
    }
    return {key: bool(value) for key, value in checks.items()}
