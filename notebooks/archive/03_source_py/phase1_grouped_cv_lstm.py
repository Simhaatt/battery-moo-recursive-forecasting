# %% [markdown]
# # Phase 1d - Grouped Cross-Validation for Luh & Blank Main Models
#
# **Purpose.** Strengthen Phase 1 after review by showing that the LSTM result is
# not a lucky fixed train/val/test split.
#
# This notebook uses **Luh & Blank only**. Kirkaldy remains reserved for Phase 4
# external validation.
#
# The cross-validation split is grouped by `cell_id`, so no cell can appear in
# both training and held-out fold data.
#
# Models/evaluations included:
# - Main full-feature LSTM with teacher-forced held-out evaluation.
# - Main full-feature LSTM with deployment rollout held-out evaluation.
# - Sparse `k_exp + Re0` LSTM rollout ablation.
# - PINN_pred pointwise baseline.
# - PINN_phys pointwise baseline with monotonic/Arrhenius-style regularization.

# %% [markdown]
# ## 0. Imports and Config

# %%
import os
import re
import json
import time
import random
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORKING.mkdir(parents=True, exist_ok=True)

RUN_FAST = bool(int(os.environ.get("PHASE1_CV_FAST", "0")))
N_FOLDS = int(os.environ.get("PHASE1_CV_FOLDS", "5"))
RUN_TAG = "phase1_grouped_cv_main_models_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORKING / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_FEATS_9 = [
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
TARGETS = ["Q", "Re"]
SEQ_LEN = 20

LSTM_MAX_EPOCHS = 45 if RUN_FAST else 180
LSTM_PATIENCE = 12 if RUN_FAST else 35
PINN_MAX_EPOCHS = 45 if RUN_FAST else 180
PINN_PATIENCE = 12 if RUN_FAST else 35
BATCH_SIZE = 512
PINN_LAMBDA_MONO = 0.08
PINN_LAMBDA_ARR = 0.02

LSTM_VARIANTS = {
    "LSTM_main_full9": RAW_FEATS_9,
    "LSTM_sparse_k_Re0": ["k_exp", "Re0"],
    "PINNfeat_sparse_k_Re0_Q0_EaStress": ["k_exp", "Re0", "Q0", "stress", "Ea_kJ_mol_mean"],
    "PINNfeat_sparse_k_Re0_Rct0_Q0_EaStress": ["k_exp", "Re0", "Rct0", "Q0", "stress", "Ea_kJ_mol_mean"],
}

PINN_VARIANTS = {
    "PINN_pred_full9": {"features": RAW_FEATS_9, "physics": False},
    "PINN_phys_full9": {"features": RAW_FEATS_9, "physics": True},
}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(SEED)

print("=" * 72)
print("Phase 1d - Grouped Cross-Validation Main Models")
print("=" * 72)
print("Device:", DEVICE)
print("RUN_FAST:", RUN_FAST)
print("N_FOLDS:", N_FOLDS)
print("Output:", OUT_DIR)

# %% [markdown]
# ## 1. Utilities

# %%
def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def safe_r2(y_true, y_pred):
    if len(y_true) < 3 or np.nanvar(y_true) <= 1e-12:
        return np.nan
    return float(r2_score(y_true, y_pred))


def per_target_metrics(y_true, y_pred):
    rows = []
    for j, target in enumerate(TARGETS):
        rows.append({
            "target": target,
            "MAPE": mape(y_true[:, j], y_pred[:, j]),
            "RMSE": float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))),
            "R2": safe_r2(y_true[:, j], y_pred[:, j]),
        })
    rows.append({
        "target": "macro_avg",
        "MAPE": float(np.mean([r["MAPE"] for r in rows])),
        "RMSE": float(np.mean([r["RMSE"] for r in rows])),
        "R2": float(np.nanmean([r["R2"] for r in rows])),
    })
    return pd.DataFrame(rows)


def slugify(text):
    text = str(text).lower()
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch)
        elif ch in [" ", "_", "-", "+"]:
            keep.append("_")
    return "_".join("".join(keep).split("_")).strip("_")


def find_first_dir(name, roots=None):
    roots = roots or [Path("/kaggle/input"), WORKING, Path(".")]
    for root in roots:
        if not root.exists():
            continue
        hits = [p for p in root.rglob(name) if p.is_dir()]
        if hits:
            return hits[0]
    return None


def read_semicolon_csv(path, **kwargs):
    return pd.read_csv(path, sep=";", **kwargs)


def raw_cell_id_from_name(path):
    match = re.search(r"(P\d+_\d+_S\d+_C\d+)", Path(path).name)
    return match.group(1) if match else Path(path).stem

# %% [markdown]
# ## 2. Data Loading
#
# The preferred input is the processed Phase 1 dataset:
#
# - `phase2_train.csv`
# - `phase2_val.csv`
# - `phase2_test.csv`
#
# If those files are not attached, this notebook builds a Phase 1-like table from
# raw Luh & Blank folders: `cell_eocv2`, `cell_eisv2`, `cell_plsv2`.

# %%
def build_raw_luh_all_table():
    eoc_dir = find_first_dir("cell_eocv2")
    eis_dir = find_first_dir("cell_eisv2")
    pls_dir = find_first_dir("cell_plsv2")
    if eoc_dir is None or eis_dir is None:
        return None

    print_section("Raw Luh & Blank Fallback")
    print("EOC:", eoc_dir)
    print("EIS:", eis_dir)
    print("PLS:", pls_dir if pls_dir is not None else "not found")

    rows = []
    issues = []
    for eoc_path in sorted(eoc_dir.glob("cell_eocv2_*.csv")):
        cid = raw_cell_id_from_name(eoc_path)
        try:
            eoc = read_semicolon_csv(eoc_path)
        except Exception as exc:
            issues.append((cid, "eoc_read_fail", str(exc)))
            continue
        if len(eoc) == 0 or "cap_aged_est_Ah" not in eoc.columns:
            issues.append((cid, "empty_or_no_capacity", len(eoc)))
            continue

        d = eoc.copy()
        d["cell_id"] = cid
        d["Q"] = pd.to_numeric(d["cap_aged_est_Ah"], errors="coerce")
        d["cyc_charged"] = pd.to_numeric(d.get("cyc_charged"), errors="coerce")
        d["timestamp_s"] = pd.to_numeric(d.get("timestamp_s"), errors="coerce")
        d = d[d["Q"].notna() & (d["Q"] > 0.1) & d["cyc_charged"].notna()].copy()
        if len(d) < SEQ_LEN + 5:
            issues.append((cid, "too_few_eoc_rows", len(d)))
            continue
        d = d.sort_values(["cyc_charged", "timestamp_s"]).drop_duplicates("cyc_charged", keep="last").reset_index(drop=True)

        eis_path = eis_dir / eoc_path.name.replace("cell_eocv2_", "cell_eisv2_")
        if eis_path.exists():
            try:
                eis = read_semicolon_csv(eis_path)
                eis["cyc_charged"] = pd.to_numeric(eis.get("cyc_charged"), errors="coerce")
                eis["Re"] = pd.to_numeric(eis.get("z_ref_now_mOhm"), errors="coerce")
                eis_agg = eis[eis["Re"].notna()].groupby("cyc_charged", as_index=False)["Re"].median()
            except Exception as exc:
                issues.append((cid, "eis_read_fail", str(exc)))
                eis_agg = pd.DataFrame(columns=["cyc_charged", "Re"])
        else:
            eis_agg = pd.DataFrame(columns=["cyc_charged", "Re"])

        if pls_dir is not None:
            pls_path = pls_dir / eoc_path.name.replace("cell_eocv2_", "cell_plsv2_")
        else:
            pls_path = None
        if pls_path is not None and pls_path.exists():
            try:
                pls = read_semicolon_csv(pls_path)
                pls["cyc_charged"] = pd.to_numeric(pls.get("cyc_charged"), errors="coerce")
                pls["r10"] = pd.to_numeric(pls.get("r_ref_10ms_mOhm"), errors="coerce")
                pls["r1s"] = pd.to_numeric(pls.get("r_ref_1s_mOhm"), errors="coerce")
                pls["Rct_proxy"] = pls["r1s"] - pls["r10"]
                pls_agg = pls[pls["Rct_proxy"].notna()].groupby("cyc_charged", as_index=False)["Rct_proxy"].median()
                pls_re = pls[pls["r10"].notna()].groupby("cyc_charged", as_index=False)["r10"].median()
            except Exception as exc:
                issues.append((cid, "pls_read_fail", str(exc)))
                pls_agg = pd.DataFrame(columns=["cyc_charged", "Rct_proxy"])
                pls_re = pd.DataFrame(columns=["cyc_charged", "r10"])
        else:
            pls_agg = pd.DataFrame(columns=["cyc_charged", "Rct_proxy"])
            pls_re = pd.DataFrame(columns=["cyc_charged", "r10"])

        d = d.merge(eis_agg, on="cyc_charged", how="left")
        d = d.merge(pls_agg, on="cyc_charged", how="left")
        if d["Re"].isna().all() and len(pls_re):
            d = d.merge(pls_re, on="cyc_charged", how="left")
            d["Re"] = d["r10"]
        d["Re"] = d["Re"].interpolate(limit_direction="both")
        d["Rct_proxy"] = d["Rct_proxy"].interpolate(limit_direction="both")
        if d["Re"].notna().sum() < SEQ_LEN + 5:
            issues.append((cid, "too_few_re_rows", int(d["Re"].notna().sum())))
            continue

        max_cyc = d["cyc_charged"].max()
        d["k_exp"] = d["cyc_charged"] / max_cyc if max_cyc and max_cyc > 0 else np.linspace(0, 1, len(d))
        d["temperature"] = pd.to_numeric(d.get("age_temp"), errors="coerce")
        if "t_end_degC" in d.columns:
            d["temperature"] = d["temperature"].replace(0, np.nan).fillna(pd.to_numeric(d["t_end_degC"], errors="coerce"))
        d["temperature"] = d["temperature"].fillna(25.0)
        d["c_rate_chg"] = pd.to_numeric(d.get("age_chg_rate"), errors="coerce").fillna(0.0)
        d["c_rate_dischg"] = pd.to_numeric(d.get("age_dischg_rate"), errors="coerce").fillna(0.0)
        d["age_type"] = pd.to_numeric(d.get("age_type"), errors="coerce").fillna(2.0)

        if "soc_est_start" in d.columns and "soc_est_end" in d.columns:
            d["soc_window"] = (pd.to_numeric(d["soc_est_end"], errors="coerce") - pd.to_numeric(d["soc_est_start"], errors="coerce")).abs() / 100.0
        else:
            d["soc_window"] = np.nan
        d["soc_window"] = d["soc_window"].replace(0, np.nan).fillna(1.0)
        d["Q0"] = float(d["Q"].iloc[0])
        d["Re0"] = float(d["Re"].iloc[0])
        d["Rct0"] = float(d["Rct_proxy"].dropna().iloc[0]) if d["Rct_proxy"].notna().any() else float(d["Re0"] * 0.35)
        rows.append(d[["cell_id"] + RAW_FEATS_9 + TARGETS])

    pd.DataFrame(issues, columns=["cell_id", "issue", "detail"]).to_csv(OUT_DIR / "raw_build_issues.csv", index=False)
    if not rows:
        return None
    all_df = pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    return all_df.sort_values(["cell_id", "k_exp"]).reset_index(drop=True)


def discover_processed_dir():
    fixed = [
        "/kaggle/input/datasets/simhaatt/outputs-of-luh-and-blank",
        "/kaggle/input/outputs-of-luh-and-blank",
        "/kaggle/input/battery-outputs/outputs-of-luh-and-blank",
        "/kaggle/working",
        ".",
    ]
    for c in fixed:
        p = Path(c)
        if (p / "phase2_train.csv").exists() and (p / "phase2_val.csv").exists() and (p / "phase2_test.csv").exists():
            return p
    for root in [Path("/kaggle/input"), WORKING, Path(".")]:
        if not root.exists():
            continue
        for hit in sorted(root.rglob("phase2_train.csv")):
            p = hit.parent
            if (p / "phase2_val.csv").exists() and (p / "phase2_test.csv").exists():
                return p
    return None


pack_dir = discover_processed_dir()
if pack_dir is not None:
    print_section("Processed Phase 1 CSVs")
    print("Pack dir:", pack_dir)
    train_df = pd.read_csv(pack_dir / "phase2_train.csv")
    val_df = pd.read_csv(pack_dir / "phase2_val.csv")
    test_df = pd.read_csv(pack_dir / "phase2_test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
else:
    all_df = build_raw_luh_all_table()
    if all_df is None:
        raise FileNotFoundError("Could not find processed Phase 1 CSVs or raw Luh & Blank folders.")

if "cell_key" in all_df.columns and "cell_id" not in all_df.columns:
    all_df = all_df.rename(columns={"cell_key": "cell_id"})

required = ["cell_id"] + RAW_FEATS_9 + TARGETS
missing = sorted(set(required) - set(all_df.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

for c in RAW_FEATS_9 + TARGETS:
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
all_df = all_df.sort_values(["cell_id", "k_exp"]).reset_index(drop=True)


def find_ea_table():
    candidates = []
    for root in [Path("/kaggle/input"), WORKING, Path(".")]:
        if root.exists():
            candidates.extend(root.rglob("diag_pinn_param_stability_per_cell.csv"))
            candidates.extend(root.rglob("pinn_extracted_ea.csv"))
    for p in candidates:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if {"cell_id", "Ea_kJ_mol_mean"}.issubset(d.columns):
            return d[["cell_id", "Ea_kJ_mol_mean"]].copy(), p
        if {"cell_id", "Ea_kJ_mol_extracted"}.issubset(d.columns):
            out = d[["cell_id", "Ea_kJ_mol_extracted"]].copy()
            out = out.rename(columns={"Ea_kJ_mol_extracted": "Ea_kJ_mol_mean"})
            return out, p
    return None, None


ea_df, ea_path = find_ea_table()
if ea_df is None:
    ea_df = pd.DataFrame({"cell_id": all_df["cell_id"].unique(), "Ea_kJ_mol_mean": 56.0})
    print("Ea source: fallback constant 56.0 kJ/mol")
else:
    print("Ea source:", ea_path)

ea_global = float(ea_df["Ea_kJ_mol_mean"].mean())
all_df = all_df.merge(ea_df, on="cell_id", how="left")
all_df["Ea_kJ_mol_mean"] = all_df["Ea_kJ_mol_mean"].fillna(ea_global)
temp_k = all_df["temperature"].astype(float) + 273.15
k_safe = np.clip(all_df["k_exp"].astype(float), 1e-6, None)
all_df["stress"] = np.log(k_safe) - all_df["Ea_kJ_mol_mean"].astype(float) / (8.314e-3 * temp_k)

all_df.to_csv(OUT_DIR / "phase1_cv_all_rows.csv", index=False)

print("Rows:", all_df.shape)
print("Cells:", all_df["cell_id"].nunique())
print("Ea range:", float(all_df["Ea_kJ_mol_mean"].min()), "to", float(all_df["Ea_kJ_mol_mean"].max()))
print("Rows per cell:")
display(all_df.groupby("cell_id").size().describe().to_frame("rows_per_cell").T)

# %% [markdown]
# ## 3. LSTM Rollout Model

# %%
class LSTMMulti(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden=192, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


def build_teacher_sequences(df, feature_cols):
    X, y, meta = [], [], []
    seq_cols = feature_cols + TARGETS
    for cid, grp in df.groupby("cell_id", sort=False):
        g = grp.sort_values("k_exp").reset_index(drop=True)
        arr = g[seq_cols].to_numpy(np.float32)
        targ = g[TARGETS].to_numpy(np.float32)
        for i in range(SEQ_LEN, len(g)):
            X.append(arr[i - SEQ_LEN:i])
            y.append(targ[i])
            meta.append({"cell_id": cid, "k_exp": float(g.loc[i, "k_exp"])})
    if not X:
        return np.empty((0, SEQ_LEN, len(seq_cols)), np.float32), np.empty((0, 2), np.float32), pd.DataFrame(meta)
    return np.stack(X), np.stack(y), pd.DataFrame(meta)


def rollout_predict(model, sc_x, sc_y, df, feature_cols):
    model.eval()
    seq_cols = feature_cols + TARGETS
    preds, truth, meta_rows = [], [], []
    for cid, grp in df.groupby("cell_id", sort=False):
        g = grp.sort_values("k_exp").reset_index(drop=True).copy()
        if len(g) <= SEQ_LEN:
            continue
        buf = g[seq_cols].to_numpy(np.float32).copy()
        for i in range(SEQ_LEN, len(g)):
            win = buf[i - SEQ_LEN:i]
            nf = win.shape[1]
            X = sc_x.transform(win.reshape(-1, nf)).reshape(1, SEQ_LEN, nf)
            with torch.no_grad():
                yp_s = model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            yp = sc_y.inverse_transform(yp_s)[0]
            yp = np.clip(yp, [1e-6, 1e-9], None)
            preds.append(yp)
            truth.append(g.loc[i, TARGETS].to_numpy(np.float32))
            meta_rows.append({"cell_id": cid, "k_exp": float(g.loc[i, "k_exp"])})
            buf[i, len(feature_cols):len(feature_cols) + 2] = yp.astype(np.float32)
    if not preds:
        return np.empty((0, 2)), np.empty((0, 2)), pd.DataFrame(meta_rows)
    return np.asarray(truth), np.asarray(preds), pd.DataFrame(meta_rows)


def teacher_forced_predict(model, sc_x, sc_y, df, feature_cols):
    model.eval()
    X, y_true, meta = build_teacher_sequences(df, feature_cols)
    if len(X) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), meta
    nfeat = X.shape[-1]
    Xs = sc_x.transform(X.reshape(-1, nfeat)).reshape(X.shape).astype(np.float32)
    preds = []
    for start in range(0, len(Xs), 4096):
        xb = torch.tensor(Xs[start:start + 4096], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            yp_s = model(xb).cpu().numpy()
        preds.append(sc_y.inverse_transform(yp_s))
    y_pred = np.clip(np.vstack(preds), [1e-6, 1e-9], None)
    return y_true, y_pred, meta


def train_lstm_for_fold(train_df, val_df, feature_cols, seed):
    set_all_seeds(seed)
    Xtr, ytr, _ = build_teacher_sequences(train_df, feature_cols)
    Xvl, yvl, _ = build_teacher_sequences(val_df, feature_cols)
    if len(Xtr) == 0 or len(Xvl) == 0:
        raise RuntimeError("Not enough sequence rows for this fold. Reduce SEQ_LEN or check data.")

    sc_x = StandardScaler()
    sc_y = StandardScaler()
    nfeat = Xtr.shape[-1]
    Xtr_s = sc_x.fit_transform(Xtr.reshape(-1, nfeat)).reshape(Xtr.shape).astype(np.float32)
    Xvl_s = sc_x.transform(Xvl.reshape(-1, nfeat)).reshape(Xvl.shape).astype(np.float32)
    ytr_s = sc_y.fit_transform(ytr).astype(np.float32)
    yvl_s = sc_y.transform(yvl).astype(np.float32)

    model = LSTMMulti(in_dim=nfeat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr_s), torch.tensor(ytr_s)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    Xvl_t = torch.tensor(Xvl_s, dtype=torch.float32, device=DEVICE)
    yvl_t = torch.tensor(yvl_s, dtype=torch.float32, device=DEVICE)

    best_loss = np.inf
    best_state = None
    bad = 0
    history = []
    for ep in range(1, LSTM_MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(Xvl_t), yvl_t).item())
        sch.step(val_loss)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= LSTM_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, sc_x, sc_y, pd.DataFrame(history)


class PINNRegressor(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=4, dropout=0.08):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden)]
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out = self.head(self.net(x))
        return torch.cat([
            nn.functional.softplus(out[:, 0:1]) + 1e-6,
            nn.functional.softplus(out[:, 1:2]) + 1e-9,
        ], dim=1)


def build_pointwise(df, feature_cols):
    d = df.dropna(subset=feature_cols + TARGETS).copy()
    X = d[feature_cols].to_numpy(np.float32)
    y = d[TARGETS].to_numpy(np.float32)
    meta = d[["cell_id", "k_exp"]].copy()
    return X, y, meta


def pinn_physics_loss(model, xb_scaled, yp_raw, feature_cols, sc_x):
    losses = []
    if "k_exp" in feature_cols:
        k_idx = feature_cols.index("k_exp")
        x_plus = xb_scaled.clone()
        k_delta = 0.02 / max(float(sc_x.scale_[k_idx]), 1e-8)
        x_plus[:, k_idx] = x_plus[:, k_idx] + k_delta
        yp_plus = model(x_plus)
        # Capacity should not increase with ageing; Re should not decrease.
        losses.append(torch.relu(yp_plus[:, 0] - yp_raw[:, 0] + 5e-5).mean())
        losses.append(torch.relu(yp_raw[:, 1] - yp_plus[:, 1] + 5e-5).mean())
    if "temperature" in feature_cols and "k_exp" in feature_cols:
        t_idx = feature_cols.index("temperature")
        x_hot = xb_scaled.clone()
        t_delta = 5.0 / max(float(sc_x.scale_[t_idx]), 1e-8)
        x_hot[:, t_idx] = x_hot[:, t_idx] + t_delta
        yp_hot = model(x_hot)
        # Higher temperature should not predict less degradation at same cycle fraction.
        losses.append(torch.relu(yp_hot[:, 0] - yp_raw[:, 0] + 2e-5).mean())
        losses.append(torch.relu(yp_raw[:, 1] - yp_hot[:, 1] + 2e-5).mean())
    if not losses:
        return torch.tensor(0.0, device=xb_scaled.device)
    return torch.stack(losses).mean()


def train_pinn_for_fold(train_df, val_df, feature_cols, use_physics, seed):
    set_all_seeds(seed)
    Xtr_raw, ytr, _ = build_pointwise(train_df, feature_cols)
    Xvl_raw, yvl, _ = build_pointwise(val_df, feature_cols)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    Xtr = sc_x.fit_transform(Xtr_raw).astype(np.float32)
    Xvl = sc_x.transform(Xvl_raw).astype(np.float32)
    ytr_s = sc_y.fit_transform(ytr).astype(np.float32)
    yvl_s = sc_y.transform(yvl).astype(np.float32)

    model = PINNRegressor(in_dim=len(feature_cols)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr_s)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    Xvl_t = torch.tensor(Xvl, dtype=torch.float32, device=DEVICE)
    yvl_t = torch.tensor(yvl_s, dtype=torch.float32, device=DEVICE)
    best_loss = np.inf
    best_state = None
    bad = 0
    history = []

    y_mean = torch.tensor(sc_y.mean_, dtype=torch.float32, device=DEVICE)
    y_scale = torch.tensor(sc_y.scale_, dtype=torch.float32, device=DEVICE)

    for ep in range(1, PINN_MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            yp_raw = model(xb)
            yp_s = (yp_raw - y_mean) / y_scale
            data_loss = nn.functional.mse_loss(yp_s, yb)
            phys_loss = pinn_physics_loss(model, xb, yp_raw, feature_cols, sc_x) if use_physics else torch.tensor(0.0, device=DEVICE)
            loss = data_loss + PINN_LAMBDA_MONO * phys_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            yp_val_raw = model(Xvl_t)
            yp_val_s = (yp_val_raw - y_mean) / y_scale
            val_loss = float(nn.functional.mse_loss(yp_val_s, yvl_t).item())
        sch.step(val_loss)
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PINN_PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, sc_x, sc_y, pd.DataFrame(history)


def pinn_predict(model, sc_x, df, feature_cols):
    model.eval()
    X_raw, y_true, meta = build_pointwise(df, feature_cols)
    if len(X_raw) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), meta
    X = sc_x.transform(X_raw).astype(np.float32)
    preds = []
    for start in range(0, len(X), 4096):
        xb = torch.tensor(X[start:start + 4096], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            preds.append(model(xb).cpu().numpy())
    y_pred = np.clip(np.vstack(preds), [1e-6, 1e-9], None)
    return y_true, y_pred, meta

# %% [markdown]
# ## 4. Grouped Cross-Validation

# %%
cells = np.array(sorted(all_df["cell_id"].unique()))
if len(cells) < N_FOLDS:
    raise ValueError(f"Need at least {N_FOLDS} cells for grouped CV; found {len(cells)}")

gkf = GroupKFold(n_splits=N_FOLDS)
groups_by_cell = pd.DataFrame({"cell_id": cells, "group": cells})

metrics_rows = []
pred_frames = []
hist_frames = []
fold_manifest = []


def append_eval_result(fold, variant_name, protocol, feature_cols, y_true, y_pred, meta, n_heldout_cells, seconds):
    if len(y_true) == 0:
        print(f"{variant_name} {protocol}: no held-out rows.")
        return
    pred = meta.copy()
    pred["fold"] = fold
    pred["variant"] = variant_name
    pred["protocol"] = protocol
    pred["Q"] = y_true[:, 0]
    pred["Re"] = y_true[:, 1]
    pred["pred_Q"] = y_pred[:, 0]
    pred["pred_Re"] = y_pred[:, 1]
    pred_frames.append(pred)

    mdf = per_target_metrics(y_true, y_pred)
    for _, row in mdf.iterrows():
        metrics_rows.append({
            "fold": fold,
            "variant": variant_name,
            "protocol": protocol,
            "target": row["target"],
            "MAPE": row["MAPE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"],
            "n_eval_rows": int(len(y_true)),
            "n_heldout_cells": int(n_heldout_cells),
            "seconds": float(seconds),
            "features": ",".join(feature_cols),
        })
    macro = mdf[mdf["target"] == "macro_avg"].iloc[0]
    print(f"{variant_name} {protocol} fold {fold}: macro MAPE={macro['MAPE']:.4f}% R2={macro['R2']:.4f}")

for fold, (tr_cell_idx, te_cell_idx) in enumerate(gkf.split(cells, groups=cells), start=1):
    train_cells = set(cells[tr_cell_idx])
    test_cells = set(cells[te_cell_idx])
    fold_train_all = all_df[all_df["cell_id"].isin(train_cells)].copy()
    fold_test = all_df[all_df["cell_id"].isin(test_cells)].copy()

    # Inner validation split is cell-grouped inside the training cells.
    inner_cells = np.array(sorted(train_cells))
    rng = np.random.default_rng(SEED + fold)
    rng.shuffle(inner_cells)
    n_val = max(1, int(round(0.15 * len(inner_cells))))
    val_cells = set(inner_cells[:n_val])
    train_cells_inner = set(inner_cells[n_val:])
    fold_train = fold_train_all[fold_train_all["cell_id"].isin(train_cells_inner)].copy()
    fold_val = fold_train_all[fold_train_all["cell_id"].isin(val_cells)].copy()

    fold_manifest.append({
        "fold": fold,
        "train_cells": len(train_cells_inner),
        "val_cells": len(val_cells),
        "heldout_cells": len(test_cells),
        "train_rows": len(fold_train),
        "val_rows": len(fold_val),
        "heldout_rows": len(fold_test),
    })
    print_section(f"Fold {fold}/{N_FOLDS}")
    print("Train/val/heldout cells:", len(train_cells_inner), len(val_cells), len(test_cells))
    print("Rows:", len(fold_train), len(fold_val), len(fold_test))

    for variant_name, feature_cols in LSTM_VARIANTS.items():
        t0 = time.time()
        print(f"\nTraining {variant_name} with features: {feature_cols}")
        model, sc_x, sc_y, hist = train_lstm_for_fold(fold_train, fold_val, feature_cols, seed=SEED + fold)
        hist["fold"] = fold
        hist["variant"] = variant_name
        hist["model_family"] = "LSTM"
        hist_frames.append(hist)

        if variant_name == "LSTM_main_full9":
            y_true, y_pred, meta = teacher_forced_predict(model, sc_x, sc_y, fold_test, feature_cols)
            append_eval_result(
                fold, "LSTM_main_full9_teacher_forced", "teacher_forced",
                feature_cols, y_true, y_pred, meta, len(test_cells), time.time() - t0,
            )

        y_true, y_pred, meta = rollout_predict(model, sc_x, sc_y, fold_test, feature_cols)
        append_eval_result(
            fold, f"{variant_name}_rollout", "autoregressive_rollout",
            feature_cols, y_true, y_pred, meta, len(test_cells), time.time() - t0,
        )

    for variant_name, cfg in PINN_VARIANTS.items():
        t0 = time.time()
        feature_cols = cfg["features"]
        print(f"\nTraining {variant_name} with features: {feature_cols}")
        model, sc_x, sc_y, hist = train_pinn_for_fold(
            fold_train, fold_val, feature_cols, use_physics=cfg["physics"], seed=SEED + 100 + fold,
        )
        hist["fold"] = fold
        hist["variant"] = variant_name
        hist["model_family"] = "PINN"
        hist_frames.append(hist)
        y_true, y_pred, meta = pinn_predict(model, sc_x, fold_test, feature_cols)
        append_eval_result(
            fold, variant_name, "pointwise",
            feature_cols, y_true, y_pred, meta, len(test_cells), time.time() - t0,
        )

metrics = pd.DataFrame(metrics_rows)
predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
history = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
folds = pd.DataFrame(fold_manifest)

metrics.to_csv(OUT_DIR / "phase1_grouped_cv_metrics_raw.csv", index=False)
predictions.to_csv(OUT_DIR / "phase1_grouped_cv_predictions_long.csv", index=False)
history.to_csv(OUT_DIR / "phase1_grouped_cv_training_history.csv", index=False)
folds.to_csv(OUT_DIR / "phase1_grouped_cv_fold_manifest.csv", index=False)

display(metrics[metrics["target"] == "macro_avg"].sort_values(["variant", "protocol", "fold"]))

# %% [markdown]
# ## 5. Summary Tables and Figures

# %%
summary = (
    metrics.groupby(["variant", "protocol", "target"], as_index=False)
    .agg(
        mean_MAPE=("MAPE", "mean"),
        std_MAPE=("MAPE", "std"),
        mean_RMSE=("RMSE", "mean"),
        std_RMSE=("RMSE", "std"),
        mean_R2=("R2", "mean"),
        std_R2=("R2", "std"),
        folds=("fold", "nunique"),
        mean_eval_rows=("n_eval_rows", "mean"),
    )
    .sort_values(["target", "mean_MAPE"])
)
summary.to_csv(OUT_DIR / "phase1_grouped_cv_metrics_summary.csv", index=False)
display(summary)

macro = summary[summary["target"] == "macro_avg"].copy()
macro["label"] = macro["variant"] + "\n" + macro["protocol"].str.replace("_", " ")
plt.figure(figsize=(9.5, 4.8))
plt.bar(
    macro["label"],
    macro["mean_MAPE"],
    yerr=macro["std_MAPE"],
    capsize=5,
    color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"][: len(macro)],
    edgecolor="black",
    linewidth=0.5,
)
plt.ylabel("Grouped CV macro MAPE (%)")
plt.title("Phase 1 Luh & Blank grouped cross-validation")
plt.grid(axis="y", alpha=0.25)
plt.xticks(rotation=20, ha="right")
for i, (_, r) in enumerate(macro.iterrows()):
    plt.text(i, r["mean_MAPE"], f"{r['mean_MAPE']:.2f}±{r['std_MAPE']:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "phase1_grouped_cv_macro_mape.png", dpi=250, bbox_inches="tight")
plt.show()

if len(predictions):
    for variant in predictions["variant"].unique():
        sub = predictions[predictions["variant"] == variant]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(sub["Q"], sub["pred_Q"], s=10, alpha=0.45)
        lo, hi = sub[["Q", "pred_Q"]].min().min(), sub[["Q", "pred_Q"]].max().max()
        axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
        axes[0].set_xlabel("Measured Q")
        axes[0].set_ylabel("Predicted Q")
        axes[0].set_title(f"{variant}: Q")
        axes[0].grid(alpha=0.25)

        axes[1].scatter(sub["Re"], sub["pred_Re"], s=10, alpha=0.45, color="#F58518")
        lo, hi = sub[["Re", "pred_Re"]].min().min(), sub[["Re", "pred_Re"]].max().max()
        axes[1].plot([lo, hi], [lo, hi], color="black", lw=1)
        axes[1].set_xlabel("Measured Re")
        axes[1].set_ylabel("Predicted Re")
        axes[1].set_title(f"{variant}: Re")
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"phase1_grouped_cv_scatter_{slugify(variant)}.png", dpi=220, bbox_inches="tight")
        plt.show()

print("Primary table for paper:")
display(macro[["variant", "protocol", "mean_MAPE", "std_MAPE", "mean_R2", "std_R2", "folds"]])

# %% [markdown]
# ## 6. Export Zip

# %%
manifest = {
    "run_tag": RUN_TAG,
    "created_at": datetime.now().isoformat(),
    "seed": SEED,
    "n_folds": N_FOLDS,
    "run_fast": RUN_FAST,
    "seq_len": SEQ_LEN,
    "lstm_max_epochs": LSTM_MAX_EPOCHS,
    "lstm_patience": LSTM_PATIENCE,
    "pinn_max_epochs": PINN_MAX_EPOCHS,
    "pinn_patience": PINN_PATIENCE,
    "lstm_variants": LSTM_VARIANTS,
    "pinn_variants": PINN_VARIANTS,
    "interpretation": [
        "Luh & Blank only.",
        "Grouped by cell_id: no cell appears in both training and held-out fold.",
        "Kirkaldy is not used here; it remains Phase 4 external validation.",
        "LSTM main full9 is evaluated in teacher-forced and autoregressive rollout modes.",
        "Sparse LSTM is evaluated in autoregressive rollout mode.",
        "PINN-feature sparse LSTM hybrids use Ea_kJ_mol_mean and Arrhenius stress, then are evaluated in autoregressive rollout mode.",
        "PINN_pred and PINN_phys are pointwise held-out cell evaluations.",
    ],
}
(OUT_DIR / "phase1_grouped_cv_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

zip_path = WORKING / f"{RUN_TAG}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in OUT_DIR.glob("*"):
        z.write(f, arcname=f.name)

print("Zip:", zip_path)
print("Files:")
for f in sorted(OUT_DIR.glob("*")):
    print(" -", f.name)

try:
    from IPython.display import FileLink
    display(FileLink(str(zip_path)))
except Exception:
    pass
