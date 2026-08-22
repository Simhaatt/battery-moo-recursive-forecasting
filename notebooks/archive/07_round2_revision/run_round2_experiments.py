"""Round-two experiments requested for the ASC revision.

The script uses only preserved, row-level outputs from the executed notebooks.
It deliberately keeps the fixed test cells untouched during fitting and writes
machine-readable CSV files plus publication-ready PDF/PNG figures.

Examples
--------
python run_round2_experiments.py stress-ablation
python run_round2_experiments.py transfer-controls
python run_round2_experiments.py tcn
python run_round2_experiments.py statistics
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, min(8, (torch.get_num_threads() or 1))))

TARGETS = ["Q", "Re"]
FULL9 = [
    "k_exp",
    "temperature",
    "c_rate_chg",
    "c_rate_dischg",
    "soc_window",
    "age_type",
    "Q0",
    "Re0",
    "Rct0",
]
BASE4 = ["k_exp", "Re0", "Rct0", "Q0"]
SEEDS = [42, 52, 62, 72, 82]
ABLATION_SEEDS = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mape(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((yt - yp) / np.clip(np.abs(yt), 1e-8, None))) * 100)


def metrics_2d(y_true: np.ndarray, y_pred: np.ndarray, names=("Q", "Re")) -> dict:
    out = {}
    mapes = []
    for j, name in enumerate(names):
        value = mape(y_true[:, j], y_pred[:, j])
        out[f"{name}_MAPE"] = value
        out[f"{name}_RMSE"] = float(np.sqrt(np.mean((y_true[:, j] - y_pred[:, j]) ** 2)))
        out[f"{name}_R2"] = float(r2_score(y_true[:, j], y_pred[:, j]))
        mapes.append(value)
    out["macro_MAPE"] = float(np.mean(mapes))
    return out


def load_phase1_split():
    df = pd.read_csv(INPUT / "phase1_cv_all_rows.csv")
    val_cells = set(pd.read_csv(INPUT / "fixed_split_val_predictions.csv")["cell_id"].unique())
    test_cells = set(pd.read_csv(INPUT / "fixed_split_test_predictions.csv")["cell_id"].unique())
    # Cells absent from prediction CSVs have no evaluable 20-step windows and
    # therefore cannot influence sequence fitting or evaluation.
    train = df[~df["cell_id"].isin(val_cells | test_cells)].copy()
    val = df[df["cell_id"].isin(val_cells)].copy()
    test = df[df["cell_id"].isin(test_cells)].copy()
    return train, val, test


def build_sequences(df: pd.DataFrame, features: list[str], seq_len: int):
    seq_cols = features + TARGETS
    xs, ys = [], []
    for _, g0 in df.groupby("cell_id", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        arr = g[seq_cols].to_numpy(np.float32)
        targ = g[TARGETS].to_numpy(np.float32)
        for i in range(seq_len, len(g)):
            xs.append(arr[i - seq_len : i])
            ys.append(targ[i])
    if not xs:
        return np.empty((0, seq_len, len(seq_cols)), np.float32), np.empty((0, 2), np.float32)
    return np.stack(xs), np.stack(ys)


class LSTMMulti(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = 2 * dilation
        self.pad = pad
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=pad)
        self.norm = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(x)
        if self.pad:
            y = y[:, :, : -self.pad]
        y = self.drop(torch.relu(self.norm(y)))
        return torch.relu(x + y)


class TCNMulti(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualTCNBlock(hidden, d, dropout) for d in (1, 2, 4, 8)])
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))

    def forward(self, x):
        h = self.blocks(self.input(x.transpose(1, 2)))
        return self.head(h[:, :, -1])


@dataclass
class TrainConfig:
    seq_len: int = 20
    hidden: int = 192
    layers: int = 2
    lr: float = 7e-4
    max_epochs: int = 180
    patience: int = 35
    batch_size: int = 512
    weight_decay: float = 1e-4


def matched_lstm_states(seed: int, hidden: int = 192, layers: int = 2):
    """Return paired initial states for base4 and base4+stress models.

    Every common parameter is identical.  The base model's first input matrix
    is obtained by deleting the stress column from the larger model, so common
    input channels also have exactly matched initial weights.
    """
    set_seed(seed)
    large = LSTMMulti(7, hidden=hidden, layers=layers)
    large_state = copy.deepcopy(large.state_dict())
    small = LSTMMulti(6, hidden=hidden, layers=layers)
    small_state = copy.deepcopy(small.state_dict())
    for key in small_state:
        if key == "lstm.weight_ih_l0":
            # large order: base4, stress, Q, Re; small order: base4, Q, Re
            small_state[key] = large_state[key][:, [0, 1, 2, 3, 5, 6]].clone()
        elif small_state[key].shape == large_state[key].shape:
            small_state[key] = large_state[key].clone()
    return small_state, large_state


def train_sequence_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    seed: int,
    cfg: TrainConfig,
    family: str = "lstm",
    initial_state: dict | None = None,
):
    set_seed(seed)
    xtr, ytr = build_sequences(train_df, features, cfg.seq_len)
    xva, yva = build_sequences(val_df, features, cfg.seq_len)
    if not len(xtr) or not len(xva):
        raise RuntimeError("No train/validation sequences")
    sc_x, sc_y = StandardScaler(), StandardScaler()
    nf = xtr.shape[-1]
    xtr = sc_x.fit_transform(xtr.reshape(-1, nf)).reshape(xtr.shape).astype(np.float32)
    xva = sc_x.transform(xva.reshape(-1, nf)).reshape(xva.shape).astype(np.float32)
    ytr = sc_y.fit_transform(ytr).astype(np.float32)
    yva = sc_y.transform(yva).astype(np.float32)

    if family == "lstm":
        model = LSTMMulti(nf, cfg.hidden, cfg.layers).to(DEVICE)
    elif family == "tcn":
        model = TCNMulti(nf, hidden=cfg.hidden).to(DEVICE)
    else:
        raise ValueError(family)
    if initial_state is not None:
        model.load_state_dict(initial_state)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(xtr), torch.tensor(ytr)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
    )
    xv = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva, dtype=torch.float32, device=DEVICE)
    best_loss, best_state, bad, history = np.inf, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(xv), yv).item())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, sc_x, sc_y, pd.DataFrame(history), seconds


def rollout_predict(model, sc_x, sc_y, df, features, seq_len=20):
    model.eval()
    seq_cols = features + TARGETS
    truth, preds, meta = [], [], []
    infer_seconds = 0.0
    for cell_id, g0 in df.groupby("cell_id", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        buf = g[seq_cols].to_numpy(np.float32).copy()
        for i in range(seq_len, len(g)):
            win = buf[i - seq_len : i]
            x = sc_x.transform(win).reshape(1, seq_len, len(seq_cols)).astype(np.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                yp = model(torch.tensor(x, device=DEVICE)).cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - t0
            yp = sc_y.inverse_transform(yp)[0]
            yp = np.clip(yp, [1e-6, 1e-9], None)
            truth.append(g.loc[i, TARGETS].to_numpy(np.float32))
            preds.append(yp)
            meta.append({"cell_id": cell_id, "k_exp": g.loc[i, "k_exp"]})
            buf[i, len(features) : len(features) + 2] = yp
    return np.asarray(truth), np.asarray(preds), pd.DataFrame(meta), infer_seconds


def run_stress_ablation() -> None:
    train, val, test = load_phase1_split()
    cfg = TrainConfig()
    rows, pred_frames, histories = [], [], []
    for seed in ABLATION_SEEDS:
        small_state, large_state = matched_lstm_states(seed, cfg.hidden, cfg.layers)
        for variant, features, state in [
            ("base4", BASE4, small_state),
            ("base4_plus_stress", BASE4 + ["stress"], large_state),
        ]:
            model, sc_x, sc_y, history, train_seconds = train_sequence_model(
                train, val, features, seed, cfg, initial_state=state
            )
            yt, yp, meta, infer_seconds = rollout_predict(model, sc_x, sc_y, test, features, cfg.seq_len)
            result = {
                "seed": seed,
                "variant": variant,
                **metrics_2d(yt, yp),
                "train_seconds": train_seconds,
                "inference_seconds": infer_seconds,
                "inference_ms_per_trajectory": infer_seconds / max(1, meta["cell_id"].nunique()) * 1000,
                "n_eval_rows": len(yt),
                "n_eval_cells": meta["cell_id"].nunique(),
                "epochs": len(history),
            }
            rows.append(result)
            history["seed"], history["variant"] = seed, variant
            histories.append(history)
            meta["seed"], meta["variant"] = seed, variant
            meta[["Q_true", "Re_true"]] = yt
            meta[["Q_pred", "Re_pred"]] = yp
            pred_frames.append(meta)
            print(seed, variant, f"macro={result['macro_MAPE']:.4f}%", f"time={train_seconds:.1f}s")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT / "stress_ablation_seed_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT / "stress_ablation_training_history.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(OUTPUT / "stress_ablation_predictions.csv", index=False)
    summary = raw.groupby("variant", as_index=False).agg(
        mean_macro_MAPE=("macro_MAPE", "mean"),
        std_macro_MAPE=("macro_MAPE", "std"),
        mean_Q_MAPE=("Q_MAPE", "mean"),
        std_Q_MAPE=("Q_MAPE", "std"),
        mean_Re_MAPE=("Re_MAPE", "mean"),
        std_Re_MAPE=("Re_MAPE", "std"),
        mean_train_seconds=("train_seconds", "mean"),
        mean_inference_ms_per_trajectory=("inference_ms_per_trajectory", "mean"),
    )
    paired = raw.pivot(index="seed", columns="variant", values="macro_MAPE")
    delta = paired["base4_plus_stress"] - paired["base4"]
    stat, p = wilcoxon(delta, alternative="two-sided", method="exact")
    summary["paired_delta_mean_pp"] = float(delta.mean())
    summary["paired_delta_std_pp"] = float(delta.std(ddof=1))
    summary["paired_wilcoxon_p"] = float(p)
    summary.to_csv(OUTPUT / "stress_ablation_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for seed, row in paired.iterrows():
        ax.plot([0, 1], [row["base4"], row["base4_plus_stress"]], marker="o", alpha=0.7, label=f"seed {seed}")
    ax.set_xticks([0, 1], [r"$k+R_{e0}+R_{ct0}+Q_0$", "+ stress"])
    ax.set_ylabel("Rollout macro MAPE (%)")
    ax.set_title("Matched-seed stress-descriptor ablation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_stress_ablation_paired.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_stress_ablation_paired.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


KIRK_EXOG = [
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
KIRK_TARGETS = ["SOH", "Re_norm"]
KIRK_SEQ = KIRK_EXOG + KIRK_TARGETS


class LSTMSOH(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 2)
        )

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1, :])


def kirk_cutoffs(df: pd.DataFrame):
    return {key: min(max(1, int(np.floor((len(g) - 1) * 0.50))), len(g) - 2) for key, g in df.groupby("cell_key")}


def left_padded_window(arr: np.ndarray, i: int, seq_len: int):
    start = max(0, i - seq_len)
    win = arr[start:i]
    if len(win) < seq_len:
        pad = np.repeat(arr[[0]], seq_len - len(win), axis=0)
        win = np.vstack([pad, win])
    return win


def build_kirk_calibration_sequences(df, cutoffs, sc_x, sc_y, seq_len=20):
    xs, ys = [], []
    for key, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        arr, targ = g[KIRK_SEQ].to_numpy(float), g[KIRK_TARGETS].to_numpy(float)
        for i in range(1, cutoffs[key] + 1):
            xs.append(sc_x.transform(left_padded_window(arr, i, seq_len)))
            ys.append(sc_y.transform(targ[i : i + 1])[0])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def kirk_rollout(model, df, cutoffs, sc_x, sc_y, seq_len=20):
    truth, pred, meta = [], [], []
    infer_seconds = 0.0
    model.eval()
    soh_idx, re_idx = KIRK_SEQ.index("SOH"), KIRK_SEQ.index("Re_norm")
    for key, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        buf = g[KIRK_SEQ].to_numpy(float).copy()
        cutoff = cutoffs[key]
        for i in range(cutoff + 1, len(g)):
            win = left_padded_window(buf, i, seq_len)
            x = sc_x.transform(win).reshape(1, seq_len, len(KIRK_SEQ)).astype(np.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                yp = model(torch.tensor(x, device=DEVICE)).cpu().numpy()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - t0
            yp = sc_y.inverse_transform(yp)[0]
            yp = np.clip(yp, [0.01, 1e-6], None)
            truth.append(g.loc[i, KIRK_TARGETS].to_numpy(float))
            pred.append(yp)
            meta.append({"cell_key": key, "k_exp": g.loc[i, "k_exp"], "Q0": g.loc[i, "Q0"]})
            buf[i, soh_idx], buf[i, re_idx] = yp
    return np.asarray(truth), np.asarray(pred), pd.DataFrame(meta), infer_seconds


def train_target_only(df, seed, cutoffs, seq_len=20):
    set_seed(seed)
    calibration_rows = []
    target_rows = []
    for key, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        calibration_rows.append(g.iloc[: cutoffs[key] + 1])
        target_rows.append(g.iloc[1 : cutoffs[key] + 1])
    cal = pd.concat(calibration_rows)
    targ = pd.concat(target_rows)
    sc_x = StandardScaler().fit(cal[KIRK_SEQ])
    sc_y = StandardScaler().fit(targ[KIRK_TARGETS])
    xtr, ytr = build_kirk_calibration_sequences(df, cutoffs, sc_x, sc_y, seq_len)
    model = LSTMSOH(len(KIRK_SEQ)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(xtr), torch.tensor(ytr)),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history, best, best_loss, bad = [], None, np.inf, 0
    t0 = time.perf_counter()
    for epoch in range(1, 181):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if mean_loss < best_loss - 1e-6:
            best_loss = mean_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 25:
            break
    seconds = time.perf_counter() - t0
    model.load_state_dict(best)
    return model, sc_x, sc_y, pd.DataFrame(history), seconds


def run_transfer_controls() -> None:
    df = pd.read_csv(INPUT / "kirkaldy_normalized_features.csv")
    cutoffs = kirk_cutoffs(df)
    control_rows, control_predictions = [], []

    for key, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        c = cutoffs[key]
        cal, ev = g.iloc[: c + 1], g.iloc[c + 1 :].copy()
        xcal, xev = cal["k_exp"].to_numpy(float), ev["k_exp"].to_numpy(float)
        ycal, yev = cal["SOH"].to_numpy(float), ev["SOH"].to_numpy(float)
        designs = {
            "SOH_BOL_persistence": np.ones_like(xev) * ycal[0],
            "SOH_last_observation_persistence": np.ones_like(xev) * ycal[-1],
        }
        for name, transform in [("SOH_linear_fade", lambda z: z), ("SOH_sqrt_fade", np.sqrt)]:
            xc, xe = transform(xcal), transform(xev)
            slope, intercept = np.polyfit(xc, ycal, 1)
            slope = min(0.0, float(slope))
            designs[name] = np.clip(intercept + slope * xe, 0.01, 1.2)
        for name, yp in designs.items():
            q_true, q_pred = yev * ev["Q0"].to_numpy(float), yp * ev["Q0"].to_numpy(float)
            control_predictions.append(
                pd.DataFrame(
                    {
                        "cell_key": key,
                        "k_exp": xev,
                        "model": name,
                        "SOH_true": yev,
                        "SOH_pred": yp,
                        "Q_true": q_true,
                        "Q_pred": q_pred,
                    }
                )
            )
    cp = pd.concat(control_predictions, ignore_index=True)
    for model, g in cp.groupby("model"):
        control_rows.append(
            {
                "model": model,
                "seed": np.nan,
                "SOH_MAPE": mape(g["SOH_true"], g["SOH_pred"]),
                "Q_MAPE": mape(g["Q_true"], g["Q_pred"]),
                "n_eval_rows": len(g),
                "n_eval_cells": g["cell_key"].nunique(),
            }
        )

    histories = []
    for seed in SEEDS:
        model, sc_x, sc_y, hist, train_seconds = train_target_only(df, seed, cutoffs)
        yt, yp, meta, infer_seconds = kirk_rollout(model, df, cutoffs, sc_x, sc_y)
        q_true, q_pred = yt[:, 0] * meta["Q0"].to_numpy(), yp[:, 0] * meta["Q0"].to_numpy()
        row = {
            "model": "target_only_LSTM_from_scratch",
            "seed": seed,
            "SOH_MAPE": mape(yt[:, 0], yp[:, 0]),
            "Q_MAPE": mape(q_true, q_pred),
            "n_eval_rows": len(yt),
            "n_eval_cells": meta["cell_key"].nunique(),
            "train_seconds": train_seconds,
            "inference_ms_per_trajectory": infer_seconds / meta["cell_key"].nunique() * 1000,
            "epochs": len(hist),
        }
        control_rows.append(row)
        hist["seed"] = seed
        histories.append(hist)
        pred = meta.copy()
        pred["model"], pred["seed"] = row["model"], seed
        pred["SOH_true"], pred["SOH_pred"] = yt[:, 0], yp[:, 0]
        pred["Q_true"], pred["Q_pred"] = q_true, q_pred
        control_predictions.append(pred)
        print(seed, row["model"], f"Q-MAPE={row['Q_MAPE']:.4f}%")

    raw = pd.DataFrame(control_rows)
    raw.to_csv(OUTPUT / "transfer_controls_metrics.csv", index=False)
    pd.concat(control_predictions, ignore_index=True).to_csv(OUTPUT / "transfer_controls_predictions.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT / "target_only_training_history.csv", index=False)
    summary = raw.groupby("model", as_index=False).agg(
        mean_Q_MAPE=("Q_MAPE", "mean"),
        std_Q_MAPE=("Q_MAPE", "std"),
        mean_SOH_MAPE=("SOH_MAPE", "mean"),
        std_SOH_MAPE=("SOH_MAPE", "std"),
        seeds=("seed", "count"),
        n_eval_rows=("n_eval_rows", "max"),
    ).sort_values("mean_Q_MAPE")
    summary.to_csv(OUTPUT / "transfer_controls_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    plot = summary.sort_values("mean_Q_MAPE", ascending=True)
    err = plot["std_Q_MAPE"].fillna(0)
    ax.barh(plot["model"].str.replace("_", " "), plot["mean_Q_MAPE"], xerr=err, color="#4776B4")
    ax.set_xlabel("Later-window restored-capacity Q-MAPE (%)")
    ax.set_title("Kirkaldy target-domain controls (first 50% observed)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_transfer_controls.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_transfer_controls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_tcn() -> None:
    train, val, test = load_phase1_split()
    cfg = TrainConfig(hidden=96, layers=1, max_epochs=180, patience=35)
    rows, histories, preds = [], [], []
    for seed in SEEDS:
        model, sc_x, sc_y, hist, train_seconds = train_sequence_model(
            train, val, FULL9, seed, cfg, family="tcn"
        )
        yt, yp, meta, infer_seconds = rollout_predict(model, sc_x, sc_y, test, FULL9, cfg.seq_len)
        row = {
            "seed": seed,
            "model": "TCN_full9_rollout",
            **metrics_2d(yt, yp),
            "train_seconds": train_seconds,
            "inference_seconds": infer_seconds,
            "inference_ms_per_trajectory": infer_seconds / meta["cell_id"].nunique() * 1000,
            "n_eval_rows": len(yt),
            "n_eval_cells": meta["cell_id"].nunique(),
            "epochs": len(hist),
        }
        rows.append(row)
        hist["seed"] = seed
        histories.append(hist)
        meta["seed"] = seed
        meta[["Q_true", "Re_true"]], meta[["Q_pred", "Re_pred"]] = yt, yp
        preds.append(meta)
        print(seed, f"TCN macro={row['macro_MAPE']:.4f}%")
    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT / "tcn_seed_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT / "tcn_training_history.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUTPUT / "tcn_predictions.csv", index=False)
    summary = raw.agg(
        {
            "macro_MAPE": ["mean", "std"],
            "Q_MAPE": ["mean", "std"],
            "Re_MAPE": ["mean", "std"],
            "train_seconds": ["mean", "std"],
            "inference_ms_per_trajectory": ["mean", "std"],
        }
    )
    summary.to_csv(OUTPUT / "tcn_summary.csv")


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def run_statistics() -> None:
    metrics = pd.read_csv(INPUT / "phase1_grouped_cv_metrics_raw.csv")
    macro = metrics[metrics["target"].eq("macro_avg")]
    pivot = macro.pivot(index="fold", columns="variant", values="MAPE").sort_index(axis=1)
    stat, p = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot])
    ranks = pivot.rank(axis=1, method="average", ascending=True)
    mean_ranks = ranks.mean().sort_values()
    k, n = pivot.shape[1], pivot.shape[0]
    q_alpha = studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2)
    cd = float(q_alpha * math.sqrt(k * (k + 1) / (6 * n)))
    omnibus = {
        "friedman_chi_square": float(stat),
        "friedman_p": float(p),
        "n_folds": int(n),
        "n_models": int(k),
        "nemenyi_critical_difference_alpha_0_05": cd,
    }
    (OUTPUT / "statistical_omnibus.json").write_text(json.dumps(omnibus, indent=2), encoding="utf-8")
    mean_ranks.rename("mean_rank").to_csv(OUTPUT / "statistical_mean_ranks.csv")

    pair_rows = []
    for a, b in combinations(pivot.columns, 2):
        diff = pivot[a] - pivot[b]
        w, raw_p = wilcoxon(pivot[a], pivot[b], alternative="two-sided", method="exact")
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "mean_difference_pp_a_minus_b": float(diff.mean()),
                "wilcoxon_W": float(w),
                "p_raw": float(raw_p),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    pairs["p_holm"] = holm_adjust(pairs["p_raw"].tolist())
    pairs["significant_holm_0_05"] = pairs["p_holm"] < 0.05
    pairs.to_csv(OUTPUT / "statistical_pairwise_wilcoxon_holm.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    y = np.arange(len(mean_ranks))
    ax.scatter(mean_ranks.values, y, s=55, color="#163A5F", zorder=3)
    for yi, (name, rank) in enumerate(mean_ranks.items()):
        ax.text(rank + 0.05, yi, name.replace("_", " "), va="center", fontsize=8)
    ax.errorbar([1.0], [-0.8], xerr=[[0], [cd]], fmt="none", capsize=5, color="#B24A3A", lw=2)
    ax.text(1 + cd / 2, -1.15, f"CD = {cd:.2f}", ha="center", color="#B24A3A")
    ax.set_xlim(0.8, k + 0.9)
    ax.set_ylim(len(mean_ranks) - 0.4, -1.5)
    ax.set_xlabel("Average rank across five grouped folds (lower is better)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(f"Critical-difference summary (Friedman p={p:.4g})")
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_critical_difference.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_critical_difference.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(omnibus, indent=2))


def write_manifest() -> None:
    manifest = {
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "seeds": SEEDS,
        "stress_ablation_seeds": ABLATION_SEEDS,
        "inputs": {p.name: p.stat().st_size for p in sorted(INPUT.glob("*.csv"))},
    }
    (OUTPUT / "round2_environment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=["stress-ablation", "transfer-controls", "tcn", "statistics", "all"],
    )
    args = parser.parse_args()
    write_manifest()
    tasks = {
        "stress-ablation": run_stress_ablation,
        "transfer-controls": run_transfer_controls,
        "tcn": run_tcn,
        "statistics": run_statistics,
    }
    if args.task == "all":
        for fn in tasks.values():
            fn()
    else:
        tasks[args.task]()


if __name__ == "__main__":
    main()
