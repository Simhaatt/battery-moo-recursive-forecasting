"""Five-seed TCN rerun with synchronized T4 trajectory-latency measurement."""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGETS = ["Q", "Re"]
FEATURES = [
    "k_exp", "temperature", "c_rate_chg", "c_rate_dischg", "soc_window",
    "age_type", "Q0", "Re0", "Rct0",
]
SEEDS = [42, 52, 62, 72, 82]
EXPECTED_PARAMETERS = 119_234
DEVICE = torch.device("cuda")


@dataclass(frozen=True)
class Config:
    seq_len: int = 20
    hidden: int = 96
    lr: float = 7e-4
    max_epochs: int = 180
    patience: int = 35
    batch_size: int = 512
    weight_decay: float = 1e-4


def synchronize() -> None:
    torch.cuda.synchronize()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.clip(np.abs(np.asarray(y_true, dtype=float)), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    target_mapes = []
    for index, target in enumerate(TARGETS):
        value = mape(y_true[:, index], y_pred[:, index])
        target_mapes.append(value)
        result[f"{target}_MAPE"] = value
        result[f"{target}_RMSE"] = float(
            np.sqrt(np.mean((y_true[:, index] - y_pred[:, index]) ** 2))
        )
        result[f"{target}_R2"] = float(r2_score(y_true[:, index], y_pred[:, index]))
    result["macro_MAPE"] = float(np.mean(target_mapes))
    return result


def load_split(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_dir / "phase1_cv_all_rows.csv")
    validation_cells = set(
        pd.read_csv(input_dir / "fixed_split_val_predictions.csv")["cell_id"].unique()
    )
    test_cells = set(
        pd.read_csv(input_dir / "fixed_split_test_predictions.csv")["cell_id"].unique()
    )
    training = frame[~frame["cell_id"].isin(validation_cells | test_cells)].copy()
    validation = frame[frame["cell_id"].isin(validation_cells)].copy()
    test = frame[frame["cell_id"].isin(test_cells)].copy()
    if set(training.cell_id) & set(validation.cell_id):
        raise RuntimeError("Training/validation cell leakage")
    if set(training.cell_id) & set(test.cell_id):
        raise RuntimeError("Training/test cell leakage")
    if set(validation.cell_id) & set(test.cell_id):
        raise RuntimeError("Validation/test cell leakage")
    return training, validation, test


def build_sequences(
    frame: pd.DataFrame, features: list[str], sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    sequence_columns = features + TARGETS
    sequences, targets = [], []
    for _, group_unsorted in frame.groupby("cell_id", sort=False):
        group = group_unsorted.sort_values("k_exp").reset_index(drop=True)
        array = group[sequence_columns].to_numpy(np.float32)
        target_array = group[TARGETS].to_numpy(np.float32)
        for index in range(sequence_length, len(group)):
            sequences.append(array[index - sequence_length:index])
            targets.append(target_array[index])
    if not sequences:
        return (
            np.empty((0, sequence_length, len(sequence_columns)), np.float32),
            np.empty((0, len(TARGETS)), np.float32),
        )
    return np.stack(sequences), np.stack(targets)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.padding = 2 * dilation
        self.convolution = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=self.padding
        )
        self.normalization = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convolution(inputs)
        if self.padding:
            outputs = outputs[:, :, :-self.padding]
        outputs = self.dropout(torch.relu(self.normalization(outputs)))
        return torch.relu(inputs + outputs)


class TCNMulti(nn.Module):
    def __init__(self, input_dimension: int, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dimension, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *[ResidualTCNBlock(hidden, dilation, dropout) for dilation in (1, 2, 4, 8)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.input_projection(inputs.transpose(1, 2)))
        return self.head(hidden[:, :, -1])


def train_one_seed(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    config: Config,
) -> tuple[TCNMulti, StandardScaler, StandardScaler, pd.DataFrame, float]:
    set_seed(seed)
    x_train, y_train = build_sequences(training, FEATURES, config.seq_len)
    x_validation, y_validation = build_sequences(validation, FEATURES, config.seq_len)
    if not len(x_train) or not len(x_validation):
        raise RuntimeError("No training or validation sequences")

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    feature_count = x_train.shape[-1]
    x_train = scaler_x.fit_transform(x_train.reshape(-1, feature_count)).reshape(
        x_train.shape
    ).astype(np.float32)
    x_validation = scaler_x.transform(
        x_validation.reshape(-1, feature_count)
    ).reshape(x_validation.shape).astype(np.float32)
    y_train = scaler_y.fit_transform(y_train).astype(np.float32)
    y_validation = scaler_y.transform(y_validation).astype(np.float32)

    model = TCNMulti(feature_count, hidden=config.hidden).to(DEVICE)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"TCN parameter count {parameter_count:,} != expected {EXPECTED_PARAMETERS:,}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-5
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_x = torch.tensor(x_validation, device=DEVICE)
    validation_y = torch.tensor(y_validation, device=DEVICE)

    best_loss, best_state, bad_epochs = np.inf, None, 0
    history = []
    synchronize()
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            batch_y = batch_y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                nn.functional.mse_loss(model(validation_x), validation_y).item()
            )
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break
    synchronize()
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training never produced a checkpoint")
    model.load_state_dict(best_state)
    return model, scaler_x, scaler_y, pd.DataFrame(history), training_seconds


def recursive_rollout(
    model: TCNMulti,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    frame: pd.DataFrame,
    sequence_length: int,
    collect: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    sequence_columns = FEATURES + TARGETS
    truths, predictions, metadata = [], [], []
    with torch.inference_mode():
        for cell_id, group_unsorted in frame.groupby("cell_id", sort=False):
            group = group_unsorted.sort_values("k_exp").reset_index(drop=True)
            if len(group) <= sequence_length:
                continue
            buffer = group[sequence_columns].to_numpy(np.float32).copy()
            for index in range(sequence_length, len(group)):
                window = buffer[index - sequence_length:index]
                scaled = scaler_x.transform(window).reshape(
                    1, sequence_length, len(sequence_columns)
                ).astype(np.float32)
                prediction = model(torch.from_numpy(scaled).to(DEVICE)).cpu().numpy()
                prediction = scaler_y.inverse_transform(prediction)[0]
                prediction = np.clip(prediction, [1e-6, 1e-9], None)
                buffer[index, len(FEATURES):len(FEATURES) + 2] = prediction
                if collect:
                    truths.append(group.loc[index, TARGETS].to_numpy(np.float32))
                    predictions.append(prediction)
                    metadata.append({"cell_id": cell_id, "k_exp": group.loc[index, "k_exp"]})
    if not collect:
        return np.empty((0, 2)), np.empty((0, 2)), pd.DataFrame()
    return np.asarray(truths), np.asarray(predictions), pd.DataFrame(metadata)


def benchmark_trajectory_latency(
    model: TCNMulti,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    test: pd.DataFrame,
    sequence_length: int,
    trajectory_count: int,
    repeats: int,
    warmups: int = 5,
) -> tuple[float, float, list[float]]:
    """Measure synchronized end-to-end recursive rollout latency.

    Timing includes scaling, host-to-device transfer, batch-one TCN forward,
    device-to-host transfer, inverse scaling, clipping, and feedback. Dataset loading,
    model loading, and metric calculation are excluded.
    """
    for _ in range(warmups):
        recursive_rollout(model, scaler_x, scaler_y, test, sequence_length, collect=False)
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        recursive_rollout(model, scaler_x, scaler_y, test, sequence_length, collect=False)
        synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / trajectory_count)
    return float(np.mean(samples)), float(np.std(samples, ddof=1)), samples


def run(arguments: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"This evidence run requires an NVIDIA T4; detected {gpu_name}")

    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    training, validation, test = load_split(arguments.input_dir.resolve())
    config = Config(
        max_epochs=2 if arguments.smoke else 180,
        patience=2 if arguments.smoke else 35,
    )
    seeds = SEEDS[:1] if arguments.smoke else SEEDS
    latency_repeats = 3 if arguments.smoke else arguments.latency_repeats
    rows, histories, prediction_frames, latency_samples = [], [], [], []

    for seed in seeds:
        model, scaler_x, scaler_y, history, training_seconds = train_one_seed(
            training, validation, seed, config
        )
        y_true, y_pred, metadata = recursive_rollout(
            model, scaler_x, scaler_y, test, config.seq_len, collect=True
        )
        trajectory_count = int(metadata["cell_id"].nunique())
        latency_mean, latency_std, samples = benchmark_trajectory_latency(
            model, scaler_x, scaler_y, test, config.seq_len,
            trajectory_count, latency_repeats,
        )
        result = {
            "seed": seed,
            "model": "TCN_full9_rollout",
            **metrics(y_true, y_pred),
            "parameters": EXPECTED_PARAMETERS,
            "train_seconds": training_seconds,
            "inference_ms_per_trajectory": latency_mean,
            "inference_ms_per_trajectory_repeat_std": latency_std,
            "latency_repeats": latency_repeats,
            "n_eval_rows": len(y_true),
            "n_eval_cells": trajectory_count,
            "epochs": len(history),
            "device": gpu_name,
        }
        rows.append(result)
        history["seed"] = seed
        histories.append(history)
        metadata["seed"] = seed
        metadata[["Q_true", "Re_true"]] = y_true
        metadata[["Q_pred", "Re_pred"]] = y_pred
        prediction_frames.append(metadata)
        latency_samples.extend(
            {"seed": seed, "repeat": index + 1, "ms_per_trajectory": value}
            for index, value in enumerate(samples)
        )
        print(
            f"seed={seed} macro_MAPE={result['macro_MAPE']:.6f}% "
            f"latency={latency_mean:.6f} ms/trajectory train={training_seconds:.2f}s"
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "tcn_t4_seed_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(
        output / "tcn_t4_training_history.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output / "tcn_t4_predictions.csv", index=False
    )
    pd.DataFrame(latency_samples).to_csv(
        output / "tcn_t4_latency_repeats.csv", index=False
    )
    summary_columns = [
        "macro_MAPE", "Q_MAPE", "Re_MAPE", "train_seconds",
        "inference_ms_per_trajectory",
        "inference_ms_per_trajectory_repeat_std",
    ]
    raw[summary_columns].agg(["mean", "std"]).to_csv(output / "tcn_t4_summary.csv")

    manifest = {
        "experiment": "five-seed TCN T4 rerun with latency",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu_name,
        "seeds": seeds,
        "configuration": asdict(config),
        "features": FEATURES,
        "targets": TARGETS,
        "parameters": EXPECTED_PARAMETERS,
        "latency_repeats": latency_repeats,
        "latency_warmups": 5,
        "latency_unit": "milliseconds per evaluable test-cell trajectory",
        "latency_scope": (
            "synchronized end-to-end recursive rollout: scaling, H2D, batch-one "
            "forward, D2H, inverse scaling, clipping, and autoregressive feedback"
        ),
        "test_rows": int(len(test)),
        "test_cells": int(test.cell_id.nunique()),
        "evaluable_test_rows": int(raw.n_eval_rows.iloc[0]),
        "evaluable_test_cells": int(raw.n_eval_cells.iloc[0]),
        "smoke": bool(arguments.smoke),
    }
    (output / "tcn_t4_environment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    archive = arguments.archive.resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive_without_suffix = archive.with_suffix("")
    created = Path(shutil.make_archive(str(archive_without_suffix), "zip", output))
    if created != archive:
        created.replace(archive)
    print(f"Results archive: {archive} ({archive.stat().st_size / 1e6:.2f} MB)")
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
