"""Forty-cell Kirkaldy transfer audit for the Soft Computing revision.

The first 50% of every target trajectory is calibration data. Evaluation uses
only the later 50%. A source-pretrained/head-adapted LSTM is compared with the
same architecture trained from scratch on target calibration data and with
SOH-space persistence and parametric-fade controls.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
LOCAL_CORE = HERE / "round2_core"
ROUND2_CORE = LOCAL_CORE if LOCAL_CORE.exists() else HERE.parent / "07_round2_revision"
sys.path.insert(0, str(ROUND2_CORE))

from run_round2_experiments import DEVICE, SEEDS, load_phase1_split, mape, set_seed  # noqa: E402


OUTPUT = HERE / "outputs_kirkaldy40_transfer"
OUTPUT.mkdir(parents=True, exist_ok=True)
KIRKALDY_FILE = HERE / "kirkaldy_40_normalized_features.csv"
EPS = 1e-9
SEQ_LEN = 20
PINN_FEATURES = [
    "k_exp",
    "temperature",
    "c_rate_chg",
    "c_rate_dischg",
    "soc_window",
    "age_type",
    "stress",
    "stress_logEa",
    "Ea_kJ_mol_mean",
]
TARGETS = ["SOH", "Re_norm"]
SEQUENCE_FEATURES = PINN_FEATURES + TARGETS


class LSTMSOH(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, inputs):
        hidden, _ = self.lstm(inputs)
        return self.head(hidden[:, -1, :])


def enrich(frame: pd.DataFrame, external: bool = False) -> pd.DataFrame:
    result = frame.copy()
    if "cell_key" not in result:
        result["cell_key"] = result["cell_id"].astype(str)
    for column in ["Q", "Q0", "Re", "Re0", "k_exp", "temperature", "c_rate_chg", "soc_window"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["SOH"] = result["Q"] / result["Q0"].replace(0, np.nan)
    result["Re_norm"] = result["Re"] / result["Re0"].replace(0, np.nan)
    if "Ea_kJ_mol_mean" not in result:
        result["Ea_kJ_mol_mean"] = 56.0
    result["Ea_kJ_mol_mean"] = pd.to_numeric(result["Ea_kJ_mol_mean"], errors="coerce").fillna(56.0)
    temperature_kelvin = result["temperature"].fillna(25.0) + 273.15
    k_safe = np.clip(result["k_exp"].fillna(0.0), 1e-6, None)
    result["stress"] = (
        result["c_rate_chg"].fillna(0.3).abs()
        * (result["soc_window"].fillna(1.0).abs() + EPS)
        * np.exp((temperature_kelvin - 298.15) / 50.0)
    )
    result["stress_logEa"] = np.log(k_safe) - result["Ea_kJ_mol_mean"] / (8.314e-3 * temperature_kelvin)
    for column in PINN_FEATURES + TARGETS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["Re_norm"] = result.groupby("cell_key")["Re_norm"].transform(
        lambda series: series.interpolate(limit_direction="both").fillna(series.median())
    )
    result[PINN_FEATURES] = result[PINN_FEATURES].fillna(result[PINN_FEATURES].median(numeric_only=True)).fillna(0.0)
    result = result.dropna(subset=TARGETS).sort_values(["cell_key", "k_exp"]).reset_index(drop=True)
    if external and result["cell_key"].nunique() != 40:
        raise RuntimeError(f"Expected 40 external cells, found {result['cell_key'].nunique()}")
    return result


def source_sequences(frame, scaler_x, scaler_y, sequence_length=SEQ_LEN):
    inputs, targets = [], []
    for _, group0 in frame.groupby("cell_key", sort=False):
        group = group0.sort_values("k_exp").reset_index(drop=True)
        x_values = group[SEQUENCE_FEATURES].to_numpy(float)
        y_values = group[TARGETS].to_numpy(float)
        for index in range(sequence_length, len(group)):
            inputs.append(scaler_x.transform(x_values[index - sequence_length : index]))
            targets.append(scaler_y.transform(y_values[index : index + 1])[0])
    return np.asarray(inputs, np.float32), np.asarray(targets, np.float32)


def cutoffs(frame: pd.DataFrame) -> dict[str, int]:
    return {
        cell_key: min(max(1, int(np.floor((len(group) - 1) * 0.50))), len(group) - 2)
        for cell_key, group in frame.groupby("cell_key", sort=False)
    }


def left_padded_window(values: np.ndarray, index: int, sequence_length: int = SEQ_LEN) -> np.ndarray:
    window = values[max(0, index - sequence_length) : index]
    if len(window) < sequence_length:
        window = np.vstack([np.repeat(window[:1], sequence_length - len(window), axis=0), window])
    return window


def calibration_sequences(frame, cutoff_map, scaler_x, scaler_y):
    inputs, targets = [], []
    for cell_key, group0 in frame.groupby("cell_key", sort=False):
        group = group0.sort_values("k_exp").reset_index(drop=True)
        x_values = group[SEQUENCE_FEATURES].to_numpy(float)
        y_values = group[TARGETS].to_numpy(float)
        for index in range(1, cutoff_map[cell_key] + 1):
            inputs.append(scaler_x.transform(left_padded_window(x_values, index)))
            targets.append(scaler_y.transform(y_values[index : index + 1])[0])
    return np.asarray(inputs, np.float32), np.asarray(targets, np.float32)


def train_source(source_train, source_validation, seed, max_epochs, patience):
    set_seed(seed)
    scaler_x = StandardScaler().fit(source_train[SEQUENCE_FEATURES].to_numpy(float))
    scaler_y = StandardScaler().fit(source_train[TARGETS].to_numpy(float))
    x_train, y_train = source_sequences(source_train, scaler_x, scaler_y)
    x_validation, y_validation = source_sequences(source_validation, scaler_x, scaler_y)
    model = LSTMSOH(len(SEQUENCE_FEATURES)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_tensor = torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    y_val_tensor = torch.tensor(y_validation, dtype=torch.float32, device=DEVICE)
    best_state, best_loss, wait, history = None, np.inf, 0, []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.smooth_l1_loss(model(x_val_tensor), y_val_tensor).item())
        history.append({"stage": "source_pretraining", "epoch": epoch, "train_loss": np.mean(losses), "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-7:
            best_state, best_loss, wait = copy.deepcopy(model.state_dict()), validation_loss, 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, scaler_x, scaler_y, pd.DataFrame(history), time.perf_counter() - start


def train_on_calibration(model, frame, cutoff_map, scaler_x, scaler_y, seed, mode, max_epochs, patience):
    set_seed(seed)
    x_train, y_train = calibration_sequences(frame, cutoff_map, scaler_x, scaler_y)
    if mode == "head_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("head")
        learning_rate = 1e-4
    elif mode == "from_scratch":
        learning_rate = 1e-3
    else:
        raise ValueError(mode)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=min(64, len(x_train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_loss, wait, history = copy.deepcopy(model.state_dict()), np.inf, 0, []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 0.25 if mode == "head_only" else 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        history.append({"stage": mode, "epoch": epoch, "calibration_loss": mean_loss})
        if mean_loss < best_loss - 1e-7:
            best_state, best_loss, wait = copy.deepcopy(model.state_dict()), mean_loss, 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), time.perf_counter() - start


def train_target_only(frame, cutoff_map, seed, max_epochs, patience):
    set_seed(seed)
    calibration_rows = []
    target_rows = []
    for cell_key, group0 in frame.groupby("cell_key", sort=False):
        group = group0.sort_values("k_exp").reset_index(drop=True)
        calibration_rows.append(group.iloc[: cutoff_map[cell_key] + 1])
        target_rows.append(group.iloc[1 : cutoff_map[cell_key] + 1])
    scaler_x = StandardScaler().fit(pd.concat(calibration_rows)[SEQUENCE_FEATURES].to_numpy(float))
    scaler_y = StandardScaler().fit(pd.concat(target_rows)[TARGETS].to_numpy(float))
    model = LSTMSOH(len(SEQUENCE_FEATURES)).to(DEVICE)
    model, history, seconds = train_on_calibration(
        model, frame, cutoff_map, scaler_x, scaler_y, seed, "from_scratch", max_epochs, patience
    )
    return model, scaler_x, scaler_y, history, seconds


def rollout(frame, cutoff_map, model, scaler_x, scaler_y, model_name, seed):
    model.eval()
    rows = []
    soh_index, resistance_index = SEQUENCE_FEATURES.index("SOH"), SEQUENCE_FEATURES.index("Re_norm")
    inference_seconds = 0.0
    for cell_key, group0 in frame.groupby("cell_key", sort=False):
        group = group0.sort_values("k_exp").reset_index(drop=True)
        buffer = group[SEQUENCE_FEATURES].to_numpy(float).copy()
        for index in range(cutoff_map[cell_key] + 1, len(group)):
            window = scaler_x.transform(left_padded_window(buffer, index)).reshape(1, SEQ_LEN, len(SEQUENCE_FEATURES))
            start = time.perf_counter()
            with torch.no_grad():
                prediction_scaled = model(torch.tensor(window, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            prediction = scaler_y.inverse_transform(prediction_scaled)[0]
            soh_prediction = float(np.clip(prediction[0], 0.01, 1.2))
            resistance_prediction = float(max(prediction[1], 1e-6))
            q0 = float(group.loc[index, "Q0"])
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "cell_key": cell_key,
                    "exp": int(group.loc[index, "exp"]),
                    "k_exp": float(group.loc[index, "k_exp"]),
                    "SOH_true": float(group.loc[index, "SOH"]),
                    "SOH_pred": soh_prediction,
                    "Q_true": float(group.loc[index, "Q"]),
                    "Q_pred": soh_prediction * q0,
                    "Re_norm_true": float(group.loc[index, "Re_norm"]),
                    "Re_norm_pred": resistance_prediction,
                }
            )
            buffer[index, soh_index] = soh_prediction
            buffer[index, resistance_index] = resistance_prediction
    return pd.DataFrame(rows), inference_seconds


def control_predictions(frame, cutoff_map):
    predictions = []
    for cell_key, group0 in frame.groupby("cell_key", sort=False):
        group = group0.sort_values("k_exp").reset_index(drop=True)
        cutoff = cutoff_map[cell_key]
        calibration, evaluation = group.iloc[: cutoff + 1], group.iloc[cutoff + 1 :]
        x_calibration = calibration["k_exp"].to_numpy(float)
        x_evaluation = evaluation["k_exp"].to_numpy(float)
        y_calibration = calibration["SOH"].to_numpy(float)
        designs = {
            "SOH_BOL_persistence": np.full_like(x_evaluation, y_calibration[0]),
            "SOH_last_observation_persistence": np.full_like(x_evaluation, y_calibration[-1]),
        }
        for name, transform in (("SOH_linear_fade", lambda values: values), ("SOH_sqrt_fade", np.sqrt)):
            slope, intercept = np.polyfit(transform(x_calibration), y_calibration, 1)
            designs[name] = np.clip(intercept + min(0.0, float(slope)) * transform(x_evaluation), 0.01, 1.2)
        for name, prediction in designs.items():
            q0 = evaluation["Q0"].to_numpy(float)
            predictions.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "seed": np.nan,
                        "cell_key": cell_key,
                        "exp": evaluation["exp"].to_numpy(int),
                        "k_exp": x_evaluation,
                        "SOH_true": evaluation["SOH"].to_numpy(float),
                        "SOH_pred": prediction,
                        "Q_true": evaluation["Q"].to_numpy(float),
                        "Q_pred": prediction * q0,
                    }
                )
            )
    return pd.concat(predictions, ignore_index=True)


def metric_tables(predictions: pd.DataFrame):
    overall_rows, cell_rows, experiment_rows = [], [], []
    keys = ["model", "seed"]
    for group_key, group in predictions.groupby(keys, dropna=False, sort=False):
        model, seed = group_key
        overall_rows.append(
            {
                "model": model,
                "seed": seed,
                "Q_MAPE": mape(group["Q_true"], group["Q_pred"]),
                "SOH_MAPE": mape(group["SOH_true"], group["SOH_pred"]),
                "n_eval_rows": len(group),
                "n_eval_cells": group["cell_key"].nunique(),
            }
        )
        for cell_key, cell_group in group.groupby("cell_key"):
            cell_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "cell_key": cell_key,
                    "exp": int(cell_group["exp"].iloc[0]),
                    "Q_MAPE": mape(cell_group["Q_true"], cell_group["Q_pred"]),
                    "n_eval_rows": len(cell_group),
                }
            )
        for experiment, experiment_group in group.groupby("exp"):
            experiment_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "exp": int(experiment),
                    "Q_MAPE": mape(experiment_group["Q_true"], experiment_group["Q_pred"]),
                    "n_eval_rows": len(experiment_group),
                    "n_eval_cells": experiment_group["cell_key"].nunique(),
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(cell_rows), pd.DataFrame(experiment_rows)


def cell_level_tests(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    averaged = cell_metrics.groupby(["model", "cell_key"], as_index=False)["Q_MAPE"].mean()
    pivot = averaged.pivot(index="cell_key", columns="model", values="Q_MAPE")
    comparisons = [
        ("source_pretrained_head_adapted", "target_only_LSTM_from_scratch"),
        ("source_pretrained_head_adapted", "SOH_sqrt_fade"),
    ]
    rows = []
    rng = np.random.default_rng(20260820)
    for left, right in comparisons:
        paired = pivot[[left, right]].dropna()
        difference = paired[left].to_numpy() - paired[right].to_numpy()
        statistic, p_value = wilcoxon(difference, alternative="two-sided", method="auto")
        boot = np.asarray(
            [rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(10000)]
        )
        rows.append(
            {
                "model_a": left,
                "model_b": right,
                "mean_cell_MAPE_difference_a_minus_b_pp": float(difference.mean()),
                "median_cell_MAPE_difference_a_minus_b_pp": float(np.median(difference)),
                "bootstrap_95ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_95ci_high": float(np.quantile(boot, 0.975)),
                "wilcoxon_W": float(statistic),
                "wilcoxon_p_two_sided": float(p_value),
                "n_cells": len(difference),
            }
        )
    return pd.DataFrame(rows)


def main(smoke: bool = False) -> None:
    source_train_raw, source_validation_raw, _ = load_phase1_split()
    source_train, source_validation = enrich(source_train_raw), enrich(source_validation_raw)
    external = enrich(pd.read_csv(KIRKALDY_FILE), external=True)
    cutoff_map = cutoffs(external)
    seeds = (42,) if smoke else tuple(SEEDS)
    source_epochs, source_patience = (2, 2) if smoke else (140, 25)
    target_epochs, target_patience = (2, 2) if smoke else (180, 25)

    controls = control_predictions(external, cutoff_map)
    predictions = [controls]
    histories, timing_rows = [], []
    for seed in seeds:
        source_model, source_scaler_x, source_scaler_y, source_history, source_seconds = train_source(
            source_train, source_validation, seed, source_epochs, source_patience
        )
        source_history["seed"] = seed
        histories.append(source_history)
        adapted_model = copy.deepcopy(source_model)
        adapted_model, adaptation_history, adaptation_seconds = train_on_calibration(
            adapted_model,
            external,
            cutoff_map,
            source_scaler_x,
            source_scaler_y,
            seed,
            "head_only",
            target_epochs,
            target_patience,
        )
        adaptation_history["seed"] = seed
        histories.append(adaptation_history)
        adapted_predictions, adapted_inference = rollout(
            external,
            cutoff_map,
            adapted_model,
            source_scaler_x,
            source_scaler_y,
            "source_pretrained_head_adapted",
            seed,
        )
        predictions.append(adapted_predictions)

        target_model, target_scaler_x, target_scaler_y, target_history, target_seconds = train_target_only(
            external, cutoff_map, seed, target_epochs, target_patience
        )
        target_history["seed"] = seed
        histories.append(target_history)
        target_predictions, target_inference = rollout(
            external,
            cutoff_map,
            target_model,
            target_scaler_x,
            target_scaler_y,
            "target_only_LSTM_from_scratch",
            seed,
        )
        predictions.append(target_predictions)
        timing_rows.extend(
            [
                {
                    "model": "source_pretrained_head_adapted",
                    "seed": seed,
                    "source_pretrain_seconds": source_seconds,
                    "target_training_seconds": adaptation_seconds,
                    "inference_ms_per_trajectory": adapted_inference / 40 * 1000,
                },
                {
                    "model": "target_only_LSTM_from_scratch",
                    "seed": seed,
                    "source_pretrain_seconds": 0.0,
                    "target_training_seconds": target_seconds,
                    "inference_ms_per_trajectory": target_inference / 40 * 1000,
                },
            ]
        )
        print(f"Completed transfer audit seed {seed}", flush=True)

    prediction_table = pd.concat(predictions, ignore_index=True)
    overall, per_cell, per_experiment = metric_tables(prediction_table)
    summary = overall.groupby("model", as_index=False).agg(
        mean_Q_MAPE=("Q_MAPE", "mean"),
        std_Q_MAPE=("Q_MAPE", "std"),
        mean_SOH_MAPE=("SOH_MAPE", "mean"),
        std_SOH_MAPE=("SOH_MAPE", "std"),
        seeds=("seed", "count"),
        n_eval_rows=("n_eval_rows", "max"),
        n_eval_cells=("n_eval_cells", "max"),
    ).sort_values("mean_Q_MAPE")

    prediction_table.to_csv(OUTPUT / "transfer_predictions.csv", index=False)
    overall.to_csv(OUTPUT / "transfer_seed_metrics.csv", index=False)
    summary.to_csv(OUTPUT / "transfer_summary.csv", index=False)
    per_cell.to_csv(OUTPUT / "transfer_cell_metrics.csv", index=False)
    per_experiment.to_csv(OUTPUT / "transfer_experiment_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT / "transfer_training_history.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUTPUT / "transfer_timing.csv", index=False)
    cell_level_tests(per_cell).to_csv(OUTPUT / "transfer_cell_level_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    plot = summary.sort_values("mean_Q_MAPE", ascending=True)
    ax.barh(plot["model"].str.replace("_", " "), plot["mean_Q_MAPE"], xerr=plot["std_Q_MAPE"].fillna(0), color="#4C78A8")
    ax.set_xlabel("Later-50% restored-capacity Q-MAPE (%)")
    ax.set_title("Forty-cell Kirkaldy transfer controls")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_kirkaldy40_transfer.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_kirkaldy40_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "journal_target": "Soft Computing (Springer)",
        "protocol": "first 50% of each target trajectory used for calibration; later 50% used only for evaluation",
        "external_cells": 40,
        "external_records": int(len(external)),
        "external_evaluation_points": int(sum(len(group) - cutoff_map[key] - 1 for key, group in external.groupby("cell_key"))),
        "source_train_cells": int(source_train["cell_key"].nunique()),
        "source_validation_cells": int(source_validation["cell_key"].nunique()),
        "seeds": seeds,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "source_epochs_max": source_epochs,
        "target_epochs_max": target_epochs,
        "smoke_test": smoke,
    }
    (OUTPUT / "transfer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(smoke=arguments.smoke)
