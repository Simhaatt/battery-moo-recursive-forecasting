"""Instrument the archived best PINN-feature hybrid checkpoint.

The benchmark replays the exact number of autoregressive forward calls in each
held-out trajectory. Standardized zero-valued feature windows are sufficient for
latency measurement because dense/LSTM execution does not depend on the values.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parents[1] / "legacy_workspace_archive" / "_phase6_pinn_feature_sparse_20260625_064713" / "phase6_pinn_feature_sparse_20260625_064713"
MODEL_DIR = ARCHIVE / "models"
STEM = "pinnfeat_sparse_k_re0_rct0_q0_eastress"


class HybridLSTM(nn.Module):
    def __init__(self, in_dim=8, hidden=192, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2),
        )
        self.raw_ea = nn.Parameter(torch.tensor(0.0))
        self.raw_q_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_re_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_q_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_re_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


def main():
    torch.set_num_threads(1)
    model = HybridLSTM()
    state = torch.load(MODEL_DIR / f"{STEM}.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    pred = pd.read_csv(ARCHIVE / f"predictions_{STEM}_test.csv")
    calls = pred.groupby("cell_id").size().astype(int).to_dict()
    window = torch.zeros((1, 20, 8), dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(20):
            model(window)
        repeats = 100
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for n in calls.values():
                for _ in range(n):
                    model(window)
            samples.append(1000.0 * (time.perf_counter() - t0) / len(calls))

    result = {
        "model": "PINNfeat_sparse_k_Re0_Rct0_Q0_EaStress",
        "n_trajectories": len(calls),
        "n_forward_calls": int(sum(calls.values())),
        "repeats": repeats,
        "mean_inference_ms_per_trajectory": float(np.mean(samples)),
        "std_inference_ms_per_trajectory": float(np.std(samples, ddof=1)),
        "training_seconds_archived": 22.745105504989624,
        "device": "CPU; torch threads=1",
    }
    out = ROOT / "outputs" / "archived_hybrid_latency.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
