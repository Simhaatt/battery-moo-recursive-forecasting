# %% [markdown]
# # Phase 4e - Few-Shot Kirkaldy Fine-Tuning
#
# **Purpose.** Test whether the best Phase 4d model can adapt to a new cell
# format using only early-life calibration points from Kirkaldy.
#
# This notebook:
# 1. Trains the normalized PINN-feature LSTM on Luh/Blank.
# 2. Uses `SOH = Q/Q0` and `Re_norm = Re/Re0`.
# 3. Fine-tunes lightly on early Kirkaldy RPT rows only.
# 4. Evaluates autoregressive rollout only on later, unseen Kirkaldy rows.
#
# Calibration protocols:
# - `BOL_only`: no fine-tuning, BOL is the rollout seed.
# - `first_2_RPT`: fine-tune on the first post-BOL target row per cell.
# - `first_3_RPT`: fine-tune on the first two post-BOL target rows per cell.
# - `first_25pct`: fine-tune on the first quarter of each cell's RPT sequence.

# %%
import copy
import json
import math
import os
import random
import re
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
if not KAGGLE_INPUT.exists():
    KAGGLE_INPUT = Path(".")
    WORKING = Path(".")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_FAST = bool(int(os.environ.get("PHASE4E_FAST", "0")))
RUN_TAG = "phase4e_fewshot_finetune_external_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORKING / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 20
SOURCE_EPOCHS = 45 if RUN_FAST else 140
SOURCE_PATIENCE = 10 if RUN_FAST else 25
FT_EPOCHS = 35 if RUN_FAST else 80
FT_PATIENCE = 8 if RUN_FAST else 15
BATCH_SIZE = 512
FT_BATCH_SIZE = 64
EPS = 1e-9

PINN_FEATS = [
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
HIST_TARGETS = ["SOH", "Re_norm"]
SEQ_FEATURES = PINN_FEATS + HIST_TARGETS
TARGETS = ["SOH", "Re_norm"]

print("Run tag:", RUN_TAG)
print("Device:", DEVICE)
print("RUN_FAST:", RUN_FAST)
print("Output:", OUT_DIR)

# %% [markdown]
# ## 1. Utilities

# %%
def print_section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def normalize_name(s):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def find_col(columns, keywords, exclude=None):
    exclude = exclude or []
    for kw in keywords:
        for col in columns:
            if kw in col and not any(ex in col for ex in exclude):
                return col
    return None


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > EPS)
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 3 or np.var(y_true) <= EPS:
        return np.nan
    return float(r2_score(y_true, y_pred))


def metric_row(model, protocol, target, y_true, y_pred, n_cells, n_calibration_rows, split):
    return {
        "Model": model,
        "Protocol": protocol,
        "Split": split,
        "Target": target,
        "MAPE": mape(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
        "n": int(len(y_true)),
        "n_cells": int(n_cells),
        "n_calibration_rows": int(n_calibration_rows),
    }


def add_normalized_columns(df):
    out = df.copy()
    for col in ["Q", "Q0", "Re", "Re0"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["SOH"] = out["Q"] / out["Q0"].replace(0, np.nan)
    out["Re_norm"] = out["Re"] / out["Re0"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def clean_model_table(df):
    out = df.copy()
    if "cell_key" not in out.columns:
        if "cell_id" in out.columns:
            out["cell_key"] = out["cell_id"].astype(str)
        else:
            out["cell_key"] = "cell_" + out.groupby("Q0").cumcount().astype(str)

    if "Ea_kJ_mol_mean" not in out.columns:
        out["Ea_kJ_mol_mean"] = 56.0
    out["Ea_kJ_mol_mean"] = pd.to_numeric(out["Ea_kJ_mol_mean"], errors="coerce").fillna(56.0)

    temp_k = pd.to_numeric(out.get("temperature", 25.0), errors="coerce").fillna(25.0) + 273.15
    k_safe = np.clip(pd.to_numeric(out.get("k_exp", 0.0), errors="coerce").fillna(0.0), 1e-6, None)
    out["stress"] = (
        pd.to_numeric(out.get("c_rate_chg", 0.3), errors="coerce").fillna(0.3).abs()
        * (pd.to_numeric(out.get("soc_window", 1.0), errors="coerce").fillna(1.0).abs() + EPS)
        * np.exp((temp_k - 298.15) / 50.0)
    )
    out["stress_logEa"] = np.log(k_safe) - out["Ea_kJ_mol_mean"].astype(float) / (8.314e-3 * temp_k)

    for col in sorted(set(PINN_FEATS + TARGETS + ["Q", "Q0", "Re", "Re0", "k_exp"])):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = add_normalized_columns(out)
    out = out[out["Q"].notna() & out["Q0"].notna() & out["SOH"].notna()].copy()
    out["Re_norm"] = out.groupby("cell_key")["Re_norm"].transform(lambda s: s.interpolate(limit_direction="both").fillna(s.median()))
    out["Re_norm"] = out["Re_norm"].fillna(out["Re_norm"].median()).fillna(1.0)
    out[PINN_FEATS] = out[PINN_FEATS].fillna(out[PINN_FEATS].median(numeric_only=True)).fillna(0.0)
    return out.sort_values(["cell_key", "k_exp"]).reset_index(drop=True)

# %% [markdown]
# ## 2. Load Luh/Blank

# %%
def discover_luh_pack_dir():
    for root in [KAGGLE_INPUT, WORKING, Path(".")]:
        if not root.exists():
            continue
        for p in [root] + [x for x in root.rglob("*") if x.is_dir()]:
            if (p / "phase2_train.csv").exists() and (p / "phase2_val.csv").exists() and (p / "phase2_test.csv").exists():
                return p
    raise FileNotFoundError("Attach Luh/Blank processed dataset with phase2_train/val/test.csv")


def attach_ea_if_available(df):
    out = df.copy()
    if "Ea_kJ_mol_mean" in out.columns:
        return out
    paths = (
        list(KAGGLE_INPUT.rglob("diag_pinn_param_stability_per_cell.csv"))
        + list(KAGGLE_INPUT.rglob("*pinn*param*.csv"))
        + list(Path(".").rglob("diag_pinn_param_stability_per_cell.csv"))
    )
    ea_df = None
    for p in paths:
        try:
            d = pd.read_csv(p)
            if {"cell_id", "Ea_kJ_mol_mean"}.issubset(d.columns):
                ea_df = d[["cell_id", "Ea_kJ_mol_mean"]].copy()
                break
            if {"cell_id", "Ea_kJ_mol_extracted"}.issubset(d.columns):
                ea_df = d[["cell_id", "Ea_kJ_mol_extracted"]].rename(columns={"Ea_kJ_mol_extracted": "Ea_kJ_mol_mean"})
                break
        except Exception:
            continue
    if ea_df is None or "cell_id" not in out.columns:
        out["Ea_kJ_mol_mean"] = 56.0
        return out
    return out.merge(ea_df.drop_duplicates("cell_id"), on="cell_id", how="left")


PACK_DIR = discover_luh_pack_dir()
train_df = attach_ea_if_available(pd.read_csv(PACK_DIR / "phase2_train.csv"))
val_df = attach_ea_if_available(pd.read_csv(PACK_DIR / "phase2_val.csv"))
test_df = attach_ea_if_available(pd.read_csv(PACK_DIR / "phase2_test.csv"))

train_df = clean_model_table(train_df)
val_df = clean_model_table(val_df)
test_df = clean_model_table(test_df)

print_section("Luh/Blank")
print("Pack dir:", PACK_DIR)
print("Train/Val/Test:", train_df.shape, val_df.shape, test_df.shape)
print("Ea range:", float(train_df["Ea_kJ_mol_mean"].min()), "to", float(train_df["Ea_kJ_mol_mean"].max()))
display(train_df[["cell_key", "k_exp", "Q", "Q0", "SOH", "Ea_kJ_mol_mean", "stress_logEa"]].head())

# %% [markdown]
# ## 3. Build Kirkaldy Table

# %%
EXP_SOC_MAP = {
    1: (0.00, 0.30),
    2: (0.70, 0.85),
    3: (0.85, 1.00),
    4: (0.00, 1.00),
    5: (0.00, 1.00),
}


def count_expt_dirs(path):
    try:
        return len([p for p in path.iterdir() if p.is_dir() and "expt" in p.name.lower()])
    except Exception:
        return 0


def discover_kirkaldy_root():
    perf_dirs = [p for p in KAGGLE_INPUT.rglob("*") if p.is_dir() and p.name == "Performance Summary"]
    if perf_dirs:
        candidates = [p.parent.parent for p in perf_dirs]
        return max(candidates + [p.parent for p in candidates], key=count_expt_dirs)
    candidates = [
        p for p in KAGGLE_INPUT.rglob("*")
        if p.is_dir() and any(token in p.name.lower() for token in ["kirkald", "kirkal"])
    ]
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Attach Kirkaldy dataset containing Performance Summary CSVs")


def parse_kirkaldy_filename(path):
    name = Path(path).name
    exp_m = re.search(r"[Ee]xp[t]?\.?\s*(\d+)", name)
    temp_m = re.search(r"(\d+)\s*deg", name, re.I)
    cell_m = re.search(r"[Cc]ell\s+([A-Za-z])", name)
    return {
        "exp": int(exp_m.group(1)) if exp_m else None,
        "temp_c": int(temp_m.group(1)) if temp_m else None,
        "cell_id": str(cell_m.group(1)).upper() if cell_m else None,
    }


def build_kirkaldy_table():
    root = discover_kirkaldy_root()
    perf_csvs = sorted(root.rglob("Performance Summary/*.csv"))
    perf_csvs = [p for p in perf_csvs if not any(x in p.name.lower() for x in ["charge_data", "discharge_data", "voltage", "timeseries"])]
    rows = []
    errors = []
    for path in perf_csvs:
        meta = parse_kirkaldy_filename(path)
        if meta["exp"] is None or meta["cell_id"] is None:
            errors.append({"path": str(path), "reason": "filename_parse_failed"})
            continue
        soc_lo, soc_hi = EXP_SOC_MAP.get(meta["exp"], (0.0, 1.0))
        try:
            d = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            d = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
        d.columns = [normalize_name(c) for c in d.columns]
        c_rpt = find_col(d.columns, ["ageing_sets", "ageing_set", "rpt"])
        c_cycles = find_col(d.columns, ["ageing_cycles", "ageing_cycle", "cycles"])
        c_cap10 = find_col(d.columns, ["c_10", "c10"], exclude=["c_2", "c2"])
        c_cap2 = find_col(d.columns, ["c_2", "c2"])
        c_res = find_col(d.columns, ["0_1s_resist", "0_1s_res", "resistance"])
        c_temp = find_col(d.columns, ["age_set_av_temperature", "temperature", "temp"], exclude=["min", "max"])
        cap_col = c_cap10 or c_cap2
        if c_cycles is None or cap_col is None:
            errors.append({"path": str(path), "reason": "missing_capacity_or_cycle"})
            continue

        cell_key = f"exp{meta['exp']}_cell{meta['cell_id']}"
        for j, r in d.iterrows():
            q_mah = to_num(r.get(cap_col))
            cycles = to_num(r.get(c_cycles))
            if not np.isfinite(q_mah) or not np.isfinite(cycles):
                continue
            rpt = to_num(r.get(c_rpt)) if c_rpt else j
            temp = to_num(r.get(c_temp)) if c_temp else meta["temp_c"]
            re_ohm = to_num(r.get(c_res)) if c_res else np.nan
            rows.append({
                "cell_key": cell_key,
                "exp": meta["exp"],
                "cell_id": meta["cell_id"],
                "rpt_idx": float(rpt) if np.isfinite(rpt) else float(j),
                "ageing_cycles": float(cycles),
                "k_exp": np.nan,
                "temperature": float(temp) if np.isfinite(temp) else float(meta["temp_c"] or 25.0),
                "c_rate_chg": 0.3,
                "c_rate_dischg": 1.0,
                "soc_window": float(soc_hi - soc_lo),
                "age_type": 3.0 if meta["exp"] == 4 else 2.0,
                "Q": float(q_mah) / 1000.0,
                "Re": float(re_ohm) * 1000.0 if np.isfinite(re_ohm) else np.nan,
                "source_path": str(path),
            })

    pd.DataFrame(errors).to_csv(OUT_DIR / "phase4e_kirkaldy_parse_errors.csv", index=False)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No Kirkaldy rows parsed.")
    out = out.sort_values(["cell_key", "rpt_idx", "ageing_cycles"]).drop_duplicates(["cell_key", "rpt_idx"], keep="last")
    out = out[out["Q"].notna() & (out["Q"] > 0.1)].copy()
    out["Re"] = out.groupby("cell_key")["Re"].transform(lambda s: s.interpolate(limit_direction="both").fillna(s.median()))
    out["Re"] = out["Re"].fillna(out["Re"].median())
    out["k_exp_raw"] = out.groupby("cell_key").cumcount()
    out["k_exp"] = out.groupby("cell_key")["k_exp_raw"].transform(lambda s: s / max(float(s.max()), 1.0))
    out["Q0"] = out.groupby("cell_key")["Q"].transform("first")
    out["Re0"] = out.groupby("cell_key")["Re"].transform("first")
    out["Ea_kJ_mol_mean"] = float(train_df["Ea_kJ_mol_mean"].mean())
    return root, clean_model_table(out)


KIRK_ROOT, kirk_df = build_kirkaldy_table()
kirk_df.to_csv(OUT_DIR / "phase4e_kirkaldy_normalized_features.csv", index=False)

print_section("Kirkaldy")
print("Kirk root:", KIRK_ROOT)
print("Rows/cells:", kirk_df.shape, kirk_df["cell_key"].nunique())
print("Q range:", float(kirk_df["Q"].min()), "to", float(kirk_df["Q"].max()))
print("SOH range:", float(kirk_df["SOH"].min()), "to", float(kirk_df["SOH"].max()))
display(kirk_df[["cell_key", "k_exp", "Q", "Q0", "SOH", "Ea_kJ_mol_mean", "stress_logEa"]].head())

# %% [markdown]
# ## 4. Sequence and Model Functions

# %%
class LSTMSOH(nn.Module):
    def __init__(self, input_dim, hidden=96, layers=2, output_dim=2, dropout=0.1):
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
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1, :])


def inverse_y(sc_y, y_scaled):
    return sc_y.inverse_transform(np.asarray(y_scaled, dtype=float))


def fit_scalers(train):
    sc_x = StandardScaler().fit(train[SEQ_FEATURES].to_numpy(dtype=float))
    sc_y = StandardScaler().fit(train[TARGETS].to_numpy(dtype=float))
    return sc_x, sc_y


def build_source_sequences(df, sc_x, sc_y, seq_len=SEQ_LEN):
    xs, ys = [], []
    for _, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        x = g[SEQ_FEATURES].to_numpy(dtype=float)
        y = g[TARGETS].to_numpy(dtype=float)
        for i in range(seq_len, len(g)):
            xs.append(sc_x.transform(x[i - seq_len:i]).reshape(seq_len, len(SEQ_FEATURES)))
            ys.append(sc_y.transform(y[i:i + 1])[0])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def calibration_cutoffs(df, protocol):
    cutoffs = {}
    for cell_key, g0 in df.groupby("cell_key", sort=False):
        n = len(g0)
        if protocol == "BOL_only":
            cutoff = 0
        elif protocol == "first_2_RPT":
            cutoff = min(1, n - 2)
        elif protocol == "first_3_RPT":
            cutoff = min(2, n - 2)
        elif protocol == "first_25pct":
            cutoff = min(max(1, int(np.floor((n - 1) * 0.25))), n - 2)
        else:
            raise ValueError(protocol)
        cutoffs[cell_key] = max(0, cutoff)
    return cutoffs


def build_kirk_sequences(df, sc_x, sc_y, cutoffs, mode, seq_len=SEQ_LEN):
    xs, ys, meta = [], [], []
    for cell_key, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        cutoff = int(cutoffs[cell_key])
        if mode == "calibration":
            target_indices = range(1, cutoff + 1)
        elif mode == "eval":
            target_indices = range(cutoff + 1, len(g))
        else:
            raise ValueError(mode)
        x = g[SEQ_FEATURES].to_numpy(dtype=float)
        y = g[TARGETS].to_numpy(dtype=float)
        for i in target_indices:
            start = max(0, i - seq_len)
            win = x[start:i].copy()
            if len(win) < seq_len:
                win = np.vstack([np.repeat(win[:1], seq_len - len(win), axis=0), win])
            xs.append(sc_x.transform(win).reshape(seq_len, len(SEQ_FEATURES)))
            ys.append(sc_y.transform(y[i:i + 1])[0])
            meta.append({
                "cell_key": cell_key,
                "row_pos": i,
                "k_exp": float(g.loc[i, "k_exp"]),
                "Q_true": float(g.loc[i, "Q"]),
                "Q0": float(g.loc[i, "Q0"]),
                "SOH_true": float(g.loc[i, "SOH"]),
                "Re_norm_true": float(g.loc[i, "Re_norm"]),
            })
    if not xs:
        return (
            np.empty((0, seq_len, len(SEQ_FEATURES)), np.float32),
            np.empty((0, len(TARGETS)), np.float32),
            pd.DataFrame(meta),
        )
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), pd.DataFrame(meta)


def rollout_with_cutoff(df, model, sc_x, sc_y, cutoffs, seq_len=SEQ_LEN):
    model.eval()
    pred_rows = []
    for cell_key, g0 in df.groupby("cell_key", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True).copy()
        cutoff = int(cutoffs[cell_key])
        buf = g[SEQ_FEATURES].to_numpy(dtype=float).copy()
        soh_idx = SEQ_FEATURES.index("SOH")
        re_idx = SEQ_FEATURES.index("Re_norm")
        for i in range(1, len(g)):
            if i <= cutoff:
                buf[i, soh_idx] = float(g.loc[i, "SOH"])
                buf[i, re_idx] = float(g.loc[i, "Re_norm"])
                continue
            start = max(0, i - seq_len)
            win = buf[start:i].copy()
            if len(win) < seq_len:
                win = np.vstack([np.repeat(win[:1], seq_len - len(win), axis=0), win])
            x = sc_x.transform(win).reshape(1, seq_len, len(SEQ_FEATURES))
            with torch.no_grad():
                yp_scaled = model(torch.tensor(x, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            yp = inverse_y(sc_y, yp_scaled)[0]
            soh_pred = float(np.clip(yp[0], 0.0, 1.2))
            re_pred = float(max(yp[1], 0.0))
            q0 = float(g.loc[i, "Q0"])
            pred_rows.append({
                "cell_key": cell_key,
                "row_pos": i,
                "k_exp": float(g.loc[i, "k_exp"]),
                "SOH_true": float(g.loc[i, "SOH"]),
                "SOH_pred": soh_pred,
                "Q_true": float(g.loc[i, "Q"]),
                "Q_pred": soh_pred * q0,
                "Q0": q0,
                "Re_norm_true": float(g.loc[i, "Re_norm"]),
                "Re_norm_pred": re_pred,
                "calibration_cutoff": cutoff,
            })
            buf[i, soh_idx] = soh_pred
            buf[i, re_idx] = re_pred
    return pd.DataFrame(pred_rows)

# %% [markdown]
# ## 5. Source Training on Luh/Blank

# %%
source_train = pd.concat([train_df, val_df], ignore_index=True)
sc_x, sc_y = fit_scalers(source_train)
Xtr, ytr = build_source_sequences(source_train, sc_x, sc_y, SEQ_LEN)

if len(Xtr) == 0:
    raise RuntimeError("No source sequences. Check Luh/Blank data.")

model = LSTMSOH(input_dim=len(SEQ_FEATURES)).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.SmoothL1Loss()
loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=BATCH_SIZE, shuffle=True)

# Use Luh/Blank test teacher-forced sequences only for early stopping.
Xstop, ystop = build_source_sequences(test_df, sc_x, sc_y, SEQ_LEN)
best_state = None
best_loss = float("inf")
wait = 0
source_history = []

for epoch in range(1, SOURCE_EPOCHS + 1):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * len(xb)
        n += len(xb)
    train_loss = total / max(n, 1)
    model.eval()
    with torch.no_grad():
        pred_stop = model(torch.tensor(Xstop, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    y_true = inverse_y(sc_y, ystop)
    y_pred = inverse_y(sc_y, pred_stop)
    stop_mape = mape(y_true[:, 0], y_pred[:, 0])
    source_history.append({"epoch": epoch, "train_loss": train_loss, "luh_test_teacher_SOH_MAPE": stop_mape})

    if stop_mape < best_loss:
        best_loss = stop_mape
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
    if epoch % 10 == 0 or epoch == 1:
        print(f"source epoch {epoch:03d} train_loss={train_loss:.5f} luh_test_teacher_SOH_MAPE={stop_mape:.3f}")
    if wait >= SOURCE_PATIENCE:
        print("Source early stop at epoch", epoch)
        break

model.load_state_dict(best_state)
source_state = copy.deepcopy(model.state_dict())
pd.DataFrame(source_history).to_csv(OUT_DIR / "phase4e_source_training_history.csv", index=False)
print_section("Source Training Complete")
print("Best Luh/Blank teacher-forced SOH MAPE:", best_loss)

# %% [markdown]
# ## 6. Few-Shot Fine-Tuning and Evaluation

# %%
PROTOCOLS = ["BOL_only", "first_2_RPT", "first_3_RPT", "first_25pct"]


def finetune_model(protocol, cutoffs):
    ft_model = LSTMSOH(input_dim=len(SEQ_FEATURES)).to(DEVICE)
    ft_model.load_state_dict(copy.deepcopy(source_state))

    Xcal, ycal, cal_meta = build_kirk_sequences(kirk_df, sc_x, sc_y, cutoffs, mode="calibration", seq_len=SEQ_LEN)
    if protocol == "BOL_only" or len(Xcal) == 0:
        return ft_model, cal_meta, pd.DataFrame()

    # Freeze most recurrent dynamics and adapt the prediction head plus final recurrent layer.
    for name, param in ft_model.named_parameters():
        param.requires_grad = name.startswith("head") or "lstm.weight_hh_l1" in name or "lstm.weight_ih_l1" in name or "lstm.bias" in name

    loader = DataLoader(TensorDataset(torch.tensor(Xcal), torch.tensor(ycal)), batch_size=min(FT_BATCH_SIZE, len(Xcal)), shuffle=True)
    opt = torch.optim.AdamW([p for p in ft_model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_state = copy.deepcopy(ft_model.state_dict())
    best_loss = float("inf")
    wait = 0
    hist = []
    for epoch in range(1, FT_EPOCHS + 1):
        ft_model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = ft_model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in ft_model.parameters() if p.requires_grad], 0.5)
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        cal_loss = total / max(n, 1)
        hist.append({"protocol": protocol, "epoch": epoch, "calibration_loss": cal_loss, "calibration_rows": len(Xcal)})
        if cal_loss < best_loss - 1e-7:
            best_loss = cal_loss
            best_state = copy.deepcopy(ft_model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= FT_PATIENCE:
            break
    ft_model.load_state_dict(best_state)
    return ft_model, cal_meta, pd.DataFrame(hist)


metrics = []
all_preds = []
all_ft_history = []
calibration_manifest = []

for protocol in PROTOCOLS:
    cutoffs = calibration_cutoffs(kirk_df, protocol)
    model_for_protocol, cal_meta, ft_history = finetune_model(protocol, cutoffs)
    preds = rollout_with_cutoff(kirk_df, model_for_protocol, sc_x, sc_y, cutoffs, SEQ_LEN)
    preds["Protocol"] = protocol
    preds["Model"] = "NormSOH_PINNfeat_LSTM_finetuned" if protocol != "BOL_only" else "NormSOH_PINNfeat_LSTM_frozen"
    all_preds.append(preds)
    if not ft_history.empty:
        all_ft_history.append(ft_history)

    n_cal = int(len(cal_meta))
    calibration_manifest.append({
        "Protocol": protocol,
        "n_calibration_rows": n_cal,
        "n_eval_rows": int(len(preds)),
        "mean_cutoff_row_pos": float(np.mean(list(cutoffs.values()))),
    })
    model_name = "NormSOH_PINNfeat_LSTM_finetuned" if protocol != "BOL_only" else "NormSOH_PINNfeat_LSTM_frozen"
    metrics.append(metric_row(model_name, protocol, "SOH", preds["SOH_true"], preds["SOH_pred"], preds["cell_key"].nunique(), n_cal, "Kirkaldy_later_rows"))
    metrics.append(metric_row(model_name, protocol + "_restored", "Q", preds["Q_true"], preds["Q_pred"], preds["cell_key"].nunique(), n_cal, "Kirkaldy_later_rows"))

preds_df = pd.concat(all_preds, ignore_index=True)
preds_df.to_csv(OUT_DIR / "phase4e_fewshot_predictions.csv", index=False)
if all_ft_history:
    pd.concat(all_ft_history, ignore_index=True).to_csv(OUT_DIR / "phase4e_finetune_history.csv", index=False)
else:
    pd.DataFrame().to_csv(OUT_DIR / "phase4e_finetune_history.csv", index=False)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT_DIR / "phase4e_fewshot_metrics_summary.csv", index=False)
pd.DataFrame(calibration_manifest).to_csv(OUT_DIR / "phase4e_calibration_manifest.csv", index=False)

print_section("Few-Shot Metrics")
display(metrics_df.sort_values(["Target", "MAPE"]))

# %% [markdown]
# ## 7. Baselines and Paper Table

# %%
# Add BOL persistence over the same later rows for each protocol.
baseline_rows = []
for protocol in PROTOCOLS:
    p = preds_df[preds_df["Protocol"].eq(protocol)].copy()
    p["SOH_pred"] = 1.0
    p["Q_pred"] = p["Q0"]
    n_cal = int(pd.DataFrame(calibration_manifest).set_index("Protocol").loc[protocol, "n_calibration_rows"])
    baseline_rows.append(metric_row("BOL_persistence", protocol, "SOH", p["SOH_true"], p["SOH_pred"], p["cell_key"].nunique(), n_cal, "Kirkaldy_later_rows"))
    baseline_rows.append(metric_row("BOL_persistence", protocol + "_restored", "Q", p["Q_true"], p["Q_pred"], p["cell_key"].nunique(), n_cal, "Kirkaldy_later_rows"))

paper = pd.concat([metrics_df, pd.DataFrame(baseline_rows)], ignore_index=True)
paper["Rank_in_target_protocol"] = paper.groupby(["Target", "Protocol"])["MAPE"].rank(method="min")
paper.to_csv(OUT_DIR / "phase4e_paper_fewshot_table.csv", index=False)

best_q = paper[paper["Target"].eq("Q")].sort_values("MAPE").reset_index(drop=True)
claims = pd.DataFrame([
    {
        "Claim": "Few-shot fine-tuning tests deployable domain adaptation, not full Kirkaldy retraining.",
        "Evidence": "Only early RPT rows are used for calibration; all reported metrics are on later rows.",
        "Paper_Use": "Use this as an optional external adaptation experiment after the frozen Phase 4d result.",
    },
    {
        "Claim": "The frozen normalized PINN-feature LSTM remains the zero-calibration reference.",
        "Evidence": "BOL_only uses no fine-tuning and evaluates post-BOL rollout.",
        "Paper_Use": "Compare all fine-tuning protocols against the BOL_only frozen row.",
    },
    {
        "Claim": "A fine-tuning result is publishable only if it beats both the frozen model and BOL persistence on the same later rows.",
        "Evidence": "The notebook exports matched protocol-specific BOL persistence rows.",
        "Paper_Use": "Avoid leakage by not evaluating on calibration rows.",
    },
])
claims.to_csv(OUT_DIR / "phase4e_fewshot_claims.csv", index=False)

print_section("Paper Table")
display(best_q)
display(claims)

# %% [markdown]
# ## 8. Figures

# %%
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print("Saved:", path)


q_plot = paper[paper["Target"].eq("Q")].copy()
q_plot["Label"] = q_plot["Protocol"] + " | " + q_plot["Model"]
q_plot = q_plot.sort_values("MAPE")
fig, ax = plt.subplots(figsize=(9.0, max(4.2, 0.38 * len(q_plot) + 1.5)))
colors = np.where(q_plot["Model"].str.contains("persistence"), "#9A9A9A", "#4C78A8")
ax.barh(q_plot["Label"], q_plot["MAPE"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("Later-row restored Q-MAPE [%]")
ax.set_title("Phase 4e Few-Shot Kirkaldy Adaptation")
for i, v in enumerate(q_plot["MAPE"]):
    ax.text(v + 0.15, i, f"{v:.2f}", va="center")
ax.grid(axis="x", alpha=0.25)
savefig("figure_phase4e_fewshot_q_mape.png")
plt.show()

best_protocol = best_q.iloc[0]["Protocol"].replace("_restored", "")
best_preds = preds_df[preds_df["Protocol"].eq(best_protocol)].copy()
cells = list(best_preds["cell_key"].drop_duplicates())[:8]
fig, axes = plt.subplots(math.ceil(len(cells) / 2), 2, figsize=(10, max(4, 2.2 * math.ceil(len(cells) / 2))), squeeze=False)
for ax, cell in zip(axes.ravel(), cells):
    g = best_preds[best_preds["cell_key"].eq(cell)].sort_values("k_exp")
    ax.plot(g["k_exp"], g["SOH_true"], marker="o", color="#1F4E79", label="true")
    ax.plot(g["k_exp"], g["SOH_pred"], marker="s", color="#B85C00", label=best_protocol)
    ax.set_title(cell)
    ax.set_xlabel("k_exp")
    ax.set_ylabel("SOH")
    ax.grid(alpha=0.2)
for ax in axes.ravel()[len(cells):]:
    ax.axis("off")
axes.ravel()[0].legend(frameon=False)
savefig("figure_phase4e_best_fewshot_soh_trajectories.png")
plt.show()

# %% [markdown]
# ## 9. Export

# %%
manifest = {
    "run_tag": RUN_TAG,
    "pack_dir": str(PACK_DIR),
    "kirk_root": str(KIRK_ROOT),
    "seq_len": SEQ_LEN,
    "features": PINN_FEATS,
    "targets": TARGETS,
    "protocols": PROTOCOLS,
    "source_epochs_requested": SOURCE_EPOCHS,
    "finetune_epochs_requested": FT_EPOCHS,
    "run_fast": RUN_FAST,
    "outputs": sorted([p.name for p in OUT_DIR.iterdir()]),
}
(OUT_DIR / "phase4e_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

zip_path = WORKING / f"{RUN_TAG}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            z.write(p, arcname=p.relative_to(OUT_DIR))

print_section("Done")
print("Output directory:", OUT_DIR)
print("Zip:", zip_path)
print("Files:")
for p in sorted(OUT_DIR.iterdir()):
    print(" -", p.name)
