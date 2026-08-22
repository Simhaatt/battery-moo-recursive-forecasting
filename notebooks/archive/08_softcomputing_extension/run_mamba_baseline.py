"""Official Mamba sequence baseline under the rollout-tuned protocol."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
LOCAL_CORE = HERE / "round2_core"
ROUND2_CORE = LOCAL_CORE if LOCAL_CORE.exists() else HERE.parent / "07_round2_revision"
sys.path.insert(0, str(ROUND2_CORE))

try:
    from mamba_ssm import Mamba
except Exception as exc:  # pragma: no cover - exercised on the Kaggle GPU runtime
    raise RuntimeError(
        "The official state-spaces/mamba package is required. Install it with "
        "`pip install mamba-ssm --no-build-isolation` in a Linux CUDA runtime."
    ) from exc

from run_metaheuristic_search import FULL9, scaled_rollout_segments  # noqa: E402
from run_round2_experiments import (  # noqa: E402
    DEVICE,
    SEEDS,
    TARGETS,
    build_sequences,
    load_phase1_split,
    metrics_2d,
    rollout_predict,
    set_seed,
)


OUTPUT = HERE / "outputs_mamba"
OUTPUT.mkdir(parents=True, exist_ok=True)
COMMON_EVALUATION_START = 20


def rollout_predict_common(model, scaler_x, scaler_y, frame, features, sequence_length):
    """Autoregressive prediction on a common row window for fair model comparison."""
    model.eval()
    sequence_columns = features + TARGETS
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
            truth.append(group.loc[index, TARGETS].to_numpy(np.float32))
            predictions.append(prediction)
            metadata.append({"cell_id": cell_id, "k_exp": group.loc[index, "k_exp"]})
            buffer[index, len(features) : len(features) + 2] = prediction
    return np.asarray(truth), np.asarray(predictions), pd.DataFrame(metadata), inference_seconds


class MambaMulti(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])
        self.blocks = nn.ModuleList(
            [
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, inputs):
        hidden = self.input_projection(inputs)
        for norm, block in zip(self.norms, self.blocks):
            hidden = hidden + self.dropout(block(norm(hidden)))
        return self.head(self.final_norm(hidden)[:, -1, :])


def train_one_seed(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    teacher_epochs: int,
    teacher_patience: int,
    rollout_epochs: int,
    rollout_patience: int,
    sequence_length: int = 15,
    rollout_horizon: int = 10,
    learning_rate: float = 7e-4,
):
    set_seed(seed)
    x_train, y_train = build_sequences(train, FULL9, sequence_length)
    x_validation, y_validation = build_sequences(validation, FULL9, sequence_length)
    input_dim = x_train.shape[-1]
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    x_train = scaler_x.fit_transform(x_train.reshape(-1, input_dim)).reshape(x_train.shape).astype(np.float32)
    x_validation = scaler_x.transform(x_validation.reshape(-1, input_dim)).reshape(x_validation.shape).astype(np.float32)
    y_train = scaler_y.fit_transform(y_train).astype(np.float32)
    y_validation = scaler_y.transform(y_validation).astype(np.float32)

    model = MambaMulti(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_tensor = torch.tensor(x_validation, dtype=torch.float32, device=DEVICE)
    best_state, best_direct, bad = None, np.inf, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, teacher_epochs + 1):
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
            prediction = scaler_y.inverse_transform(model(x_val_tensor).cpu().numpy())
        direct_metric = metrics_2d(scaler_y.inverse_transform(y_validation), prediction)["macro_MAPE"]
        history.append(
            {"stage": "teacher", "epoch": epoch, "train_loss": np.mean(losses), "validation_macro_MAPE": direct_metric}
        )
        if direct_metric < best_direct - 1e-7:
            best_direct, bad = direct_metric, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        if bad >= teacher_patience:
            break
    model.load_state_dict(best_state)

    initial_windows, future_exog, rollout_targets = scaled_rollout_segments(
        train,
        FULL9,
        sequence_length,
        rollout_horizon,
        scaler_x,
        scaler_y,
        stride=2,
    )
    if len(initial_windows) == 0:
        raise RuntimeError(
            f"No rollout segments for L={sequence_length}, H={rollout_horizon}. "
            "Choose L + H no larger than the longest training trajectory."
        )
    rollout_loader = DataLoader(
        TensorDataset(
            torch.tensor(initial_windows),
            torch.tensor(future_exog),
            torch.tensor(rollout_targets),
        ),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate * 0.3, weight_decay=1e-5)
    y_true, y_pred, _, _ = rollout_predict_common(
        model, scaler_x, scaler_y, validation, FULL9, sequence_length
    )
    best_rollout = metrics_2d(y_true, y_pred)["macro_MAPE"]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    bad = 0
    history.append({"stage": "rollout", "epoch": 0, "validation_macro_MAPE": best_rollout})
    for epoch in range(1, rollout_epochs + 1):
        model.train()
        losses = []
        for window, exogenous, targets in rollout_loader:
            window, exogenous, targets = window.to(DEVICE), exogenous.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            for step in range(rollout_horizon):
                prediction = model(window)
                step_losses.append(nn.functional.mse_loss(prediction, targets[:, step, :]))
                next_row = torch.cat([exogenous[:, step, :], prediction], dim=1).unsqueeze(1)
                window = torch.cat([window[:, 1:, :], next_row], dim=1)
            loss = torch.stack(step_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        y_true, y_pred, _, _ = rollout_predict_common(
            model, scaler_x, scaler_y, validation, FULL9, sequence_length
        )
        rollout_metric = metrics_2d(y_true, y_pred)["macro_MAPE"]
        history.append(
            {"stage": "rollout", "epoch": epoch, "train_loss": np.mean(losses), "validation_macro_MAPE": rollout_metric}
        )
        if rollout_metric < best_rollout - 1e-5:
            best_rollout, bad = rollout_metric, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad += 1
        if bad >= rollout_patience:
            break
    model.load_state_dict(best_state)
    return model, scaler_x, scaler_y, pd.DataFrame(history), best_rollout, time.perf_counter() - start_time


def main(smoke: bool = False) -> None:
    train, validation, test = load_phase1_split()
    seeds = (42,) if smoke else tuple(SEEDS)
    teacher_epochs, teacher_patience = (2, 2) if smoke else (220, 40)
    rollout_epochs, rollout_patience = (1, 1) if smoke else (40, 15)
    rows, histories, predictions = [], [], []
    for seed in seeds:
        model, scaler_x, scaler_y, history, validation_mape, train_seconds = train_one_seed(
            train,
            validation,
            seed,
            teacher_epochs,
            teacher_patience,
            rollout_epochs,
            rollout_patience,
        )
        y_true, y_pred, metadata, inference_seconds = rollout_predict_common(
            model, scaler_x, scaler_y, test, FULL9, sequence_length=15
        )
        result = {
            "model": "Mamba_full9_rollout_tuned",
            "seed": seed,
            "validation_macro_MAPE": validation_mape,
            **metrics_2d(y_true, y_pred),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "train_seconds": train_seconds,
            "inference_ms_per_trajectory": inference_seconds / metadata["cell_id"].nunique() * 1000.0,
            "n_eval_rows": len(y_true),
            "n_eval_cells": int(metadata["cell_id"].nunique()),
        }
        rows.append(result)
        history["seed"] = seed
        histories.append(history)
        prediction = metadata.copy()
        prediction["seed"] = seed
        prediction[["Q_true", "Re_true"]] = y_true
        prediction[["Q_pred", "Re_pred"]] = y_pred
        predictions.append(prediction)
        print(f"Mamba seed={seed}: test macro MAPE={result['macro_MAPE']:.4f}%", flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT / "mamba_seed_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT / "mamba_training_history.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / "mamba_predictions.csv", index=False)
    summary = raw.groupby("model", as_index=False).agg(
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
    summary.to_csv(OUTPUT / "mamba_summary.csv", index=False)
    manifest = {
        "journal_target": "Soft Computing (Springer)",
        "implementation": "official state-spaces/mamba Mamba block",
        "mamba_ssm_version": importlib.metadata.version("mamba-ssm"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": seeds,
        "configuration": {
            "sequence_length": 15,
            "common_evaluation_start": COMMON_EVALUATION_START,
            "rollout_horizon": 10,
            "d_model": 64,
            "layers": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "learning_rate": 7e-4,
            "teacher_epochs": teacher_epochs,
            "rollout_epochs": rollout_epochs,
        },
        "smoke_test": smoke,
    }
    (OUTPUT / "mamba_environment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(smoke=arguments.smoke)
