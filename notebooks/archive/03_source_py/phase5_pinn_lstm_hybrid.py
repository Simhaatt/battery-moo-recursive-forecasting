# %% [markdown]
# # Phase 5 - PINN-LSTM Hybrid Rollout Forecasting
#
# **Goal.** Build a deployment-realistic hybrid that keeps the strong Phase 1
# LSTM rollout accuracy while adding physics consistency from the PINN side.
#
# This notebook intentionally follows the Phase 1 Luh and Blank structure:
#
# - same train/val/test CSV discovery
# - same cell-level split already present in the CSVs
# - same targets: `Q` and `Re`
# - same rollout protocol: only the first `SEQ_LEN` measured targets seed the
#   sequence; later target history is replaced by model predictions
# - new candidates: physics-feature LSTM and physics-regularized PINN-LSTM
#
# The main baseline to beat is the Phase 1 deployment-realistic LSTM v4:
# **1.551% test macro MAPE**.

# %% [markdown]
# ## 0. Imports and Config

# %%
import os
import glob
import json
import random
import re
import shutil
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORKING.mkdir(parents=True, exist_ok=True)

RUN_FAST = bool(int(os.environ.get("PHASE5_FAST", "0")))
RUN_TAG = "phase5_pinn_lstm_hybrid_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORKING / RUN_TAG
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
LSTM_HIDDEN = 192
LSTM_LAYERS = 2
LSTM_BATCH_SIZE = 512
LSTM_MAX_EPOCHS = 60 if RUN_FAST else 220
LSTM_FINE_TUNE_EPOCHS = 25 if RUN_FAST else 80
LSTM_PATIENCE = 20 if RUN_FAST else 40

R_GAS = 8.314  # J/(mol K)
PHASE1_REFERENCE = {
    "LSTM_v1_teacher_forced_macro_mape": 1.405,
    "LSTM_v4_rollout_tuned_macro_mape": 1.551,
    "PINN_phys_macro_mape": 6.830,
    "PINN_phys_Ea_kJ": 57.49,
}


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(SEED)

print("=" * 72)
print("Phase 5 - PINN-LSTM Hybrid")
print("=" * 72)
print(f"Device     : {DEVICE}")
print(f"Torch      : {torch.__version__}")
print(f"RUN_FAST   : {RUN_FAST}")
print(f"Output dir : {OUT_DIR}")
print(f"Targets    : {TARGETS}")

# %% [markdown]
# ## 1. Utilities

# %%
metrics_rows = []
train_times = {}


def print_section(title, char="="):
    print("\n" + char * 72)
    print(title)
    print(char * 72)


def print_kv(label, value, width=28):
    print(f"{label:<{width}} : {value}")


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.clip(np.max(y_true) - np.min(y_true), 1e-8, None)
    return float(rmse / denom)


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.nanvar(y_true) <= 1e-12:
        return np.nan
    return float(r2_score(y_true, y_pred))


def per_target_metrics(y_true, y_pred):
    rows = []
    for j, target in enumerate(TARGETS):
        rows.append(
            {
                "target": target,
                "MAPE": mape(y_true[:, j], y_pred[:, j]),
                "RMSE": float(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))),
                "NRMSE": nrmse(y_true[:, j], y_pred[:, j]),
                "R2": safe_r2(y_true[:, j], y_pred[:, j]),
            }
        )
    rows.append(
        {
            "target": "macro_avg",
            "MAPE": float(np.mean([r["MAPE"] for r in rows])),
            "RMSE": float(np.mean([r["RMSE"] for r in rows])),
            "NRMSE": float(np.mean([r["NRMSE"] for r in rows])),
            "R2": float(np.nanmean([r["R2"] for r in rows])),
        }
    )
    return pd.DataFrame(rows)


def record_metrics(model_name, split, y_true, y_pred, extra=None):
    dfm = per_target_metrics(y_true, y_pred)
    for _, row in dfm.iterrows():
        rec = {
            "model": model_name,
            "split": split,
            "target": row["target"],
            "MAPE": row["MAPE"],
            "RMSE": row["RMSE"],
            "NRMSE": row["NRMSE"],
            "R2": row["R2"],
        }
        if extra:
            rec.update(extra)
        metrics_rows.append(rec)
    macro = dfm[dfm["target"] == "macro_avg"].iloc[0]
    print(
        f"{model_name:<30s} {split:<5s} "
        f"macro MAPE={macro['MAPE']:.4f}%  macro R2={macro['R2']:.4f}"
    )
    return dfm


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


def build_raw_luh_phase_tables():
    """Fallback for Kaggle mounts that contain raw Luh/Blank folders only."""
    eoc_dir = find_first_dir("cell_eocv2")
    eis_dir = find_first_dir("cell_eisv2")
    pls_dir = find_first_dir("cell_plsv2")
    if eoc_dir is None or eis_dir is None:
        return None

    print_section("Raw Luh/Blank Fallback")
    print_kv("EOC dir", eoc_dir)
    print_kv("EIS dir", eis_dir)
    print_kv("PLS dir", pls_dir if pls_dir is not None else "not found")

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

        d = (
            d.sort_values(["cyc_charged", "timestamp_s"])
            .drop_duplicates("cyc_charged", keep="last")
            .reset_index(drop=True)
        )

        eis_path = eis_dir / eoc_path.name.replace("cell_eocv2_", "cell_eisv2_")
        if eis_path.exists():
            try:
                eis = read_semicolon_csv(eis_path)
                eis["cyc_charged"] = pd.to_numeric(eis.get("cyc_charged"), errors="coerce")
                # Phase 1 model scale is mOhm.
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
            t_fallback = pd.to_numeric(d["t_end_degC"], errors="coerce")
            d["temperature"] = d["temperature"].replace(0, np.nan).fillna(t_fallback)
        d["temperature"] = d["temperature"].fillna(25.0)

        d["c_rate_chg"] = pd.to_numeric(d.get("age_chg_rate"), errors="coerce").fillna(0.0)
        d["c_rate_dischg"] = pd.to_numeric(d.get("age_dischg_rate"), errors="coerce").fillna(0.0)
        d["age_type"] = pd.to_numeric(d.get("age_type"), errors="coerce").fillna(2.0)

        if "soc_est_start" in d.columns and "soc_est_end" in d.columns:
            d["soc_window"] = (
                pd.to_numeric(d["soc_est_end"], errors="coerce")
                - pd.to_numeric(d["soc_est_start"], errors="coerce")
            ).abs() / 100.0
        else:
            d["soc_window"] = np.nan
        d["soc_window"] = d["soc_window"].replace(0, np.nan).fillna(1.0)

        d["Q0"] = float(d["Q"].iloc[0])
        d["Re0"] = float(d["Re"].iloc[0])
        if d["Rct_proxy"].notna().any():
            d["Rct0"] = float(d["Rct_proxy"].dropna().iloc[0])
        else:
            d["Rct0"] = float(d["Re0"].iloc[0] * 0.35) if hasattr(d["Re0"], "iloc") else float(d["Re0"] * 0.35)

        keep = ["cell_id"] + RAW_FEATS_9 + TARGETS
        rows.append(d[keep])

    if not rows:
        print("Raw fallback found folders but produced no usable rows.")
        if issues:
            print(pd.DataFrame(issues, columns=["cell_id", "issue", "detail"]).head(20).to_string(index=False))
        return None

    features = pd.concat(rows, ignore_index=True)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    features = features.sort_values(["cell_id", "k_exp"]).reset_index(drop=True)

    cells = np.array(sorted(features["cell_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(cells)
    if len(cells) >= 228:
        n_test, n_val = 21, 27
    else:
        n_test = max(1, int(round(0.10 * len(cells))))
        n_val = max(1, int(round(0.12 * len(cells))))
    test_cells = set(cells[:n_test])
    val_cells = set(cells[n_test:n_test + n_val])
    train_cells = set(cells[n_test + n_val:])

    out = WORKING / "phase5_raw_luh_processed"
    out.mkdir(parents=True, exist_ok=True)
    train = features[features["cell_id"].isin(train_cells)].copy()
    val = features[features["cell_id"].isin(val_cells)].copy()
    test = features[features["cell_id"].isin(test_cells)].copy()

    train.to_csv(out / "phase2_train.csv", index=False)
    val.to_csv(out / "phase2_val.csv", index=False)
    test.to_csv(out / "phase2_test.csv", index=False)
    features.to_csv(out / "raw_phase1_like_all.csv", index=False)
    pd.DataFrame(issues, columns=["cell_id", "issue", "detail"]).to_csv(out / "raw_build_issues.csv", index=False)

    print_kv("Raw rows", features.shape)
    print_kv("Raw cells", features["cell_id"].nunique())
    print_kv("Split cells", f"train={len(train_cells)}, val={len(val_cells)}, test={len(test_cells)}")
    print_kv("Processed dir", out)
    return out

# %% [markdown]
# ## 2. Data Loading
#
# The Kaggle dataset only needs to contain:
#
# - `phase2_train.csv`
# - `phase2_val.csv`
# - `phase2_test.csv`
#
# Optional but useful:
#
# - `diag_pinn_param_stability_per_cell.csv` with `Ea_kJ_mol_mean`

# %%
def discover_pack_dir():
    fixed_candidates = [
        "/kaggle/input/datasets/simhaatt/outputs-of-luh-and-blank",
        "/kaggle/input/outputs-of-luh-and-blank",
        "/kaggle/input/battery-outputs/outputs-of-luh-and-blank",
        "/kaggle/working",
        ".",
    ]
    for c in fixed_candidates:
        p = Path(c)
        if (p / "phase2_train.csv").exists() and (p / "phase2_val.csv").exists():
            return p

    roots = [Path("/kaggle/input"), WORKING, Path(".")]
    for root in roots:
        if not root.exists():
            continue
        hits = sorted(root.glob("**/phase2_train.csv"))
        for hit in hits:
            p = hit.parent
            if (p / "phase2_val.csv").exists() and (p / "phase2_test.csv").exists():
                return p

    # Some Kaggle datasets are uploaded as zip files rather than expanded folders.
    # Extract only the archive that contains the expected train/val/test trio.
    extract_root = WORKING / "phase5_input_extract"
    for root in roots:
        if not root.exists():
            continue
        for zpath in sorted(root.glob("**/*.zip")):
            try:
                with zipfile.ZipFile(zpath) as zf:
                    names = zf.namelist()
                    has_train = any(name.endswith("phase2_train.csv") for name in names)
                    has_val = any(name.endswith("phase2_val.csv") for name in names)
                    has_test = any(name.endswith("phase2_test.csv") for name in names)
                    if not (has_train and has_val and has_test):
                        continue
                    target = extract_root / zpath.stem
                    target.mkdir(parents=True, exist_ok=True)
                    zf.extractall(target)
                    hits = sorted(target.glob("**/phase2_train.csv"))
                    for hit in hits:
                        p = hit.parent
                        if (p / "phase2_val.csv").exists() and (p / "phase2_test.csv").exists():
                            print(f"Extracted input archive: {zpath}")
                            return p
            except Exception as exc:
                print(f"[warn] Could not inspect zip {zpath}: {exc}")

    raw_pack = build_raw_luh_phase_tables()
    if raw_pack is not None:
        return raw_pack

    if Path("/kaggle/input").exists():
        sample = []
        for fp in sorted(Path("/kaggle/input").glob("**/*"))[:80]:
            sample.append(str(fp))
        print("\nFirst /kaggle/input paths visible to this notebook:")
        for fp in sample:
            print(" -", fp)

    raise FileNotFoundError(
        "Could not find phase2_train/val/test CSVs. Attach the Luh and Blank dataset "
        "that contains phase2_train.csv, phase2_val.csv, and phase2_test.csv, or upload "
        "a zip containing those files."
    )


pack_dir = discover_pack_dir()
print_section("Data Loading")
print_kv("Pack dir", pack_dir)

train_df = pd.read_csv(pack_dir / "phase2_train.csv")
val_df = pd.read_csv(pack_dir / "phase2_val.csv")
test_df = pd.read_csv(pack_dir / "phase2_test.csv")

required_cols = set(["cell_id"] + RAW_FEATS_9 + TARGETS)
missing = sorted(required_cols - set(train_df.columns))
if missing:
    raise ValueError(f"Training CSV is missing required columns: {missing}")

print_kv("Train", f"{train_df.shape} ({train_df['cell_id'].nunique()} cells)")
print_kv("Val", f"{val_df.shape} ({val_df['cell_id'].nunique()} cells)")
print_kv("Test", f"{test_df.shape} ({test_df['cell_id'].nunique()} cells)")

ea_path = pack_dir / "diag_pinn_param_stability_per_cell.csv"
if not ea_path.exists():
    hits = sorted(Path("/kaggle/input").glob("**/diag_pinn_param_stability_per_cell.csv")) if Path("/kaggle/input").exists() else []
    ea_path = hits[0] if hits else None

if ea_path and Path(ea_path).exists():
    params_df = pd.read_csv(ea_path)
    if {"cell_id", "Ea_kJ_mol_mean"}.issubset(params_df.columns):
        ea_df = params_df[["cell_id", "Ea_kJ_mol_mean"]].copy()
        print_kv("Ea source", ea_path)
    else:
        ea_df = pd.DataFrame({"cell_id": train_df["cell_id"].unique(), "Ea_kJ_mol_mean": 56.0})
        print_kv("Ea source", "fallback constant 56 kJ/mol (columns not found)")
else:
    ea_df = pd.DataFrame({"cell_id": train_df["cell_id"].unique(), "Ea_kJ_mol_mean": 56.0})
    print_kv("Ea source", "fallback constant 56 kJ/mol")

print_kv(
    "Ea range",
    f"{ea_df['Ea_kJ_mol_mean'].min():.2f} to {ea_df['Ea_kJ_mol_mean'].max():.2f} kJ/mol",
)

# %% [markdown]
# ## 3. Physics Features
#
# Phase 5 adds two optional physics descriptors:
#
# - `Ea_kJ_mol_mean`: per-cell activation energy when available, otherwise 56 kJ/mol
# - `stress`: log-Arrhenius stress, `log(k_exp) - Ea/(R T)`
#
# These are used by the hybrid candidates, not by the frozen Phase 1 baseline.

# %%
EA_GLOBAL_KJ = float(ea_df["Ea_kJ_mol_mean"].mean())


def add_physics_features(df_in, ea_df_in, ea_global=EA_GLOBAL_KJ):
    df = df_in.copy()
    merged = df.merge(ea_df_in[["cell_id", "Ea_kJ_mol_mean"]], on="cell_id", how="left")
    merged["Ea_kJ_mol_mean"] = merged["Ea_kJ_mol_mean"].fillna(ea_global)
    temp_k = merged["temperature"].astype(float).values + 273.15
    ea_kj = merged["Ea_kJ_mol_mean"].astype(float).values
    k = np.clip(merged["k_exp"].astype(float).values, 1e-10, None)
    merged["stress"] = np.log(k) - ea_kj / (8.314e-3 * temp_k)
    return merged


train_df = add_physics_features(train_df, ea_df)
val_df = add_physics_features(val_df, ea_df)
test_df = add_physics_features(test_df, ea_df)

PHYS_FEATS = RAW_FEATS_9 + ["stress", "Ea_kJ_mol_mean"]
MINIMAL_PHYS_FEATS = ["k_exp", "temperature", "Re0", "Rct0", "Q0", "stress", "Ea_kJ_mol_mean"]

print_section("Physics Feature Summary")
for col in ["stress", "Ea_kJ_mol_mean"]:
    print_kv(
        col,
        f"train mean={train_df[col].mean():.4f}, std={train_df[col].std():.4f}, "
        f"min={train_df[col].min():.4f}, max={train_df[col].max():.4f}",
    )

# %% [markdown]
# ## 4. Sequence Builder and Rollout Helpers

# %%
META_COLS = ["k_prev", "k_cur", "T_K", "c_rate_chg", "Q_prev", "Re_prev"]


def build_seq_samples(df_in, feats, seq_len=SEQ_LEN, include_target_history=True):
    seq_features = list(feats) + TARGETS if include_target_history else list(feats)
    xs, ys, metas = [], [], []
    for _, g in df_in.groupby("cell_id"):
        g = g.sort_values("k_exp").reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        xvals = g[seq_features].values.astype(np.float32)
        yvals = g[TARGETS].values.astype(np.float32)
        kvals = g["k_exp"].values.astype(np.float32)
        temps = (g["temperature"].values.astype(np.float32) + 273.15)
        crates = g["c_rate_chg"].values.astype(np.float32)
        for i in range(seq_len, len(g)):
            xs.append(xvals[i - seq_len : i])
            ys.append(yvals[i])
            metas.append(
                [
                    float(kvals[i - 1]),
                    float(kvals[i]),
                    float(temps[i]),
                    float(crates[i]),
                    float(yvals[i - 1, 0]),
                    float(yvals[i - 1, 1]),
                ]
            )
    return (
        np.asarray(xs, np.float32),
        np.asarray(ys, np.float32),
        np.asarray(metas, np.float32),
    )


def rollout_truth(df_in, seq_len=SEQ_LEN):
    y_parts = []
    id_parts = []
    for cid, g in df_in.groupby("cell_id"):
        g = g.sort_values("k_exp").reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        y_parts.append(g[TARGETS].values.astype(np.float32)[seq_len:])
        ids = g[["cell_id", "k_exp", "temperature", "c_rate_chg"]].iloc[seq_len:].copy()
        ids["step_in_cell"] = np.arange(seq_len, len(g))
        id_parts.append(ids)
    if not y_parts:
        return np.empty((0, len(TARGETS)), dtype=np.float32), pd.DataFrame()
    return np.vstack(y_parts).astype(np.float32), pd.concat(id_parts, ignore_index=True)


class LSTMMulti(nn.Module):
    def __init__(self, in_dim, out_dim=2, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS):
        super().__init__()
        dropout = 0.2 if layers > 1 else 0.0
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

        # Physics head parameters. They are harmless for non-physics candidates.
        self.raw_ea = nn.Parameter(torch.tensor(0.0))
        self.raw_q_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_re_gain = nn.Parameter(torch.tensor(0.0))
        self.raw_q_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_re_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])

    def physics_params(self):
        ea_j = 40000.0 + 40000.0 * torch.sigmoid(self.raw_ea)
        q_gain = F.softplus(self.raw_q_gain) + 1e-6
        re_gain = F.softplus(self.raw_re_gain) + 1e-6
        q_beta = 0.5 + 1.5 * torch.sigmoid(self.raw_q_beta)
        re_beta = 0.5 + 1.5 * torch.sigmoid(self.raw_re_beta)
        return ea_j, q_gain, re_gain, q_beta, re_beta


def infer_scaled(model, sc_x, sc_y, x_seq):
    n_feat = x_seq.shape[2]
    xs = sc_x.transform(x_seq.reshape(-1, n_feat)).reshape(x_seq.shape)
    with torch.no_grad():
        yp_s = model(torch.tensor(xs, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return np.clip(sc_y.inverse_transform(yp_s), [1e-6, 1e-9], None)


def rollout_predict(model, sc_x, sc_y, df_in, feats, seq_len=SEQ_LEN):
    seq_features = list(feats) + TARGETS
    n_raw = len(feats)
    preds = []
    for _, g in df_in.groupby("cell_id"):
        g = g.sort_values("k_exp").reset_index(drop=True)
        if len(g) <= seq_len:
            continue
        buf = g[seq_features].values.astype(np.float32).copy()
        for i in range(seq_len, len(g)):
            win = buf[i - seq_len : i]
            n_feat = win.shape[1]
            ws = sc_x.transform(win.reshape(-1, n_feat)).reshape(1, seq_len, n_feat)
            with torch.no_grad():
                yp = model(torch.tensor(ws, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            yp = np.clip(sc_y.inverse_transform(yp), [1e-6, 1e-9], None)[0]
            preds.append(yp)
            buf[i, n_raw : n_raw + len(TARGETS)] = yp
    return np.asarray(preds, dtype=np.float32)


def prediction_frame(df_in, y_true, y_pred, model_name, split):
    _, ids = rollout_truth(df_in, SEQ_LEN)
    out = ids.copy()
    out["model"] = model_name
    out["split"] = split
    out["Q_true"] = y_true[:, 0]
    out["Re_true"] = y_true[:, 1]
    out["Q_pred"] = y_pred[:, 0]
    out["Re_pred"] = y_pred[:, 1]
    return out


def physics_diagnostics(pred_df):
    if len(pred_df) == 0:
        return {"Q_monotonic_violations": 0, "Re_monotonic_violations": 0}
    q_bad, re_bad, n_steps = 0, 0, 0
    for _, g in pred_df.groupby("cell_id"):
        g = g.sort_values("k_exp")
        if len(g) < 2:
            continue
        dq = np.diff(g["Q_pred"].values)
        dre = np.diff(g["Re_pred"].values)
        q_bad += int(np.sum(dq > 1e-6))
        re_bad += int(np.sum(dre < -1e-8))
        n_steps += len(dq)
    return {
        "Q_monotonic_violations": q_bad,
        "Re_monotonic_violations": re_bad,
        "monotonic_steps_checked": n_steps,
        "Q_violation_rate": q_bad / max(n_steps, 1),
        "Re_violation_rate": re_bad / max(n_steps, 1),
    }

# %% [markdown]
# ## 5. Physics-Regularized Training
#
# The hybrid loss is:
#
# `data_loss + lambda_mono * monotonic_loss + lambda_phys * Arrhenius_rate_residual`
#
# The Arrhenius term is relative to 298.15 K, avoiding tiny absolute rates:
#
# `rate(T) = rate_298 * exp(-Ea/R * (1/T - 1/298.15))`

# %%
def estimate_rate_scales(y_train, meta_train):
    dk = np.clip(meta_train[:, 1] - meta_train[:, 0], 1e-6, None)
    dqdk = (y_train[:, 0] - meta_train[:, 4]) / dk
    dredk = (y_train[:, 1] - meta_train[:, 5]) / dk
    q_decay = np.maximum(-dqdk, 0.0)
    re_growth = np.maximum(dredk, 0.0)

    q_pos = q_decay[q_decay > 1e-9]
    re_pos = re_growth[re_growth > 1e-12]
    q_scale = float(np.median(q_pos)) if len(q_pos) else float(np.std(dqdk) + 1e-6)
    re_scale = float(np.median(re_pos)) if len(re_pos) else float(np.std(dredk) + 1e-6)
    return max(q_scale, 1e-6), max(re_scale, 1e-8)


def physics_loss_terms(model, y_pred, meta, rate_scales):
    k_prev = meta[:, 0:1]
    k_cur = meta[:, 1:2]
    t_k = meta[:, 2:3].clamp(min=240.0, max=380.0)
    crate = torch.abs(meta[:, 3:4]) + 1e-3
    q_prev = meta[:, 4:5].clamp(min=1e-6)
    re_prev = meta[:, 5:6].clamp(min=1e-6)
    dk = (k_cur - k_prev).clamp(min=1e-6)

    q_pred = y_pred[:, 0:1].clamp(min=1e-6)
    re_pred = y_pred[:, 1:2].clamp(min=1e-9)

    dqdk = (q_pred - q_prev) / dk
    dredk = (re_pred - re_prev) / dk

    mono = torch.relu(dqdk + 1e-5).mean() + torch.relu(1e-7 - dredk).mean()

    ea_j, q_gain, re_gain, q_beta, re_beta = model.physics_params()
    temp_factor = torch.exp(-ea_j / R_GAS * (1.0 / t_k - 1.0 / 298.15)).clamp(0.05, 20.0)

    q_scale, re_scale = rate_scales
    q_rate = q_scale * q_gain * temp_factor * torch.pow(q_prev / q_prev.mean().clamp(min=1e-6), q_beta)
    q_rate = q_rate * torch.pow(crate / crate.mean().clamp(min=1e-6), 0.25)

    re_rate = re_scale * re_gain * temp_factor * torch.pow(re_prev / re_prev.mean().clamp(min=1e-6), re_beta)
    re_rate = re_rate * torch.pow(crate / crate.mean().clamp(min=1e-6), 0.25)

    q_res = ((-dqdk - q_rate) / q_scale).pow(2).mean()
    re_res = ((dredk - re_rate) / re_scale).pow(2).mean()
    phys = q_res + 0.3 * re_res

    ea_kj = ea_j / 1000.0
    ea_prior = ((ea_kj - 56.0) / 7.0).pow(2)
    return mono, phys, ea_prior, ea_kj.detach()


def q_only_physics_loss_terms(model, y_pred, meta, rate_scales):
    k_prev = meta[:, 0:1]
    k_cur = meta[:, 1:2]
    t_k = meta[:, 2:3].clamp(min=240.0, max=380.0)
    crate = torch.abs(meta[:, 3:4]) + 1e-3
    q_prev = meta[:, 4:5].clamp(min=1e-6)
    dk = (k_cur - k_prev).clamp(min=1e-6)

    q_pred = y_pred[:, 0:1].clamp(min=1e-6)
    dqdk = (q_pred - q_prev) / dk

    mono = torch.relu(dqdk + 1e-5).mean()

    ea_j, q_gain, _, q_beta, _ = model.physics_params()
    temp_factor = torch.exp(-ea_j / R_GAS * (1.0 / t_k - 1.0 / 298.15)).clamp(0.05, 20.0)

    q_scale = rate_scales[0]
    q_rate = q_scale * q_gain * temp_factor * torch.pow(q_prev / q_prev.mean().clamp(min=1e-6), q_beta)
    q_rate = q_rate * torch.pow(crate / crate.mean().clamp(min=1e-6), 0.25)

    phys = ((-dqdk - q_rate) / q_scale).pow(2).mean()
    ea_kj = ea_j / 1000.0
    ea_prior = ((ea_kj - 56.0) / 7.0).pow(2)
    return mono, phys, ea_prior, ea_kj.detach()


def train_candidate(candidate):
    name = candidate["name"]
    feats = candidate["features"]
    seed = int(candidate.get("seed", SEED))
    phys_weight = float(candidate.get("phys_weight", 0.0))
    mono_weight = float(candidate.get("mono_weight", 0.0))
    ea_weight = float(candidate.get("ea_weight", 0.001))
    q_only_physics = bool(candidate.get("q_only_physics", False))

    print_section(f"Training {name}")
    print_kv("Features", feats)
    print_kv("Physics weight", phys_weight)
    print_kv("Monotonic weight", mono_weight)

    Xtr, ytr, mtr = build_seq_samples(train_df, feats, SEQ_LEN, include_target_history=True)
    Xvl, yvl, mvl = build_seq_samples(val_df, feats, SEQ_LEN, include_target_history=True)
    Xte, yte, mte = build_seq_samples(test_df, feats, SEQ_LEN, include_target_history=True)

    if len(Xtr) == 0 or len(Xvl) == 0:
        raise ValueError(f"Insufficient sequence samples for {name}")

    n_feat = Xtr.shape[2]
    sc_x = StandardScaler()
    Xtr_s = sc_x.fit_transform(Xtr.reshape(-1, n_feat)).reshape(Xtr.shape).astype(np.float32)
    Xvl_s = sc_x.transform(Xvl.reshape(-1, n_feat)).reshape(Xvl.shape).astype(np.float32)

    sc_y = StandardScaler()
    ytr_s = sc_y.fit_transform(ytr).astype(np.float32)
    yvl_s = sc_y.transform(yvl).astype(np.float32)

    y_mean_t = torch.tensor(sc_y.mean_, dtype=torch.float32, device=DEVICE).view(1, 2)
    y_std_t = torch.tensor(np.clip(sc_y.scale_, 1e-8, None), dtype=torch.float32, device=DEVICE).view(1, 2)
    rate_scales = estimate_rate_scales(ytr, mtr)
    print_kv("Rate scales", f"Q={rate_scales[0]:.6g}, Re={rate_scales[1]:.6g}")

    set_all_seeds(seed)
    model = LSTMMulti(in_dim=n_feat).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=12, min_lr=1e-5)

    ds = TensorDataset(
        torch.tensor(Xtr_s, dtype=torch.float32),
        torch.tensor(ytr_s, dtype=torch.float32),
        torch.tensor(mtr, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, drop_last=False)
    Xvl_t = torch.tensor(Xvl_s, dtype=torch.float32, device=DEVICE)

    best, best_state, no_imp = float("inf"), None, 0
    warmup_end = max(1, int(0.2 * LSTM_MAX_EPOCHS))
    ramp_len = max(10, int(0.2 * LSTM_MAX_EPOCHS))
    t0 = time.time()

    for ep in range(1, LSTM_MAX_EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        last_ea = np.nan
        for xb, yb_s, mb in dl:
            xb = xb.to(DEVICE)
            yb_s = yb_s.to(DEVICE)
            mb = mb.to(DEVICE)

            opt.zero_grad()
            yp_s = model(xb)
            data_loss = F.smooth_l1_loss(yp_s, yb_s)
            loss = data_loss

            if phys_weight > 0 or mono_weight > 0:
                yp = yp_s * y_std_t + y_mean_t
                if q_only_physics:
                    mono, phys, ea_prior, ea_kj = q_only_physics_loss_terms(model, yp, mb, rate_scales)
                else:
                    mono, phys, ea_prior, ea_kj = physics_loss_terms(model, yp, mb, rate_scales)
                ramp = 0.0 if ep <= warmup_end else min(1.0, (ep - warmup_end) / float(ramp_len))
                loss = loss + ramp * (mono_weight * mono + phys_weight * phys + ea_weight * ea_prior)
                last_ea = float(ea_kj.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item()) * xb.shape[0]

        model.eval()
        with torch.no_grad():
            yv_pred = np.clip(sc_y.inverse_transform(model(Xvl_t).cpu().numpy()), [1e-6, 1e-9], None)
        val_direct = float(per_target_metrics(yvl, yv_pred).query("target == 'macro_avg'")["MAPE"].iloc[0])
        sch.step(val_direct)

        if val_direct < best:
            best = val_direct
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep == 1 or ep % 20 == 0:
            ea_msg = "" if np.isnan(last_ea) else f" Ea={last_ea:.2f}kJ"
            print(f"{name} ep {ep:3d}/{LSTM_MAX_EPOCHS} val_direct={val_direct:.4f}% best={best:.4f}%{ea_msg}")

        if no_imp >= LSTM_PATIENCE and not candidate.get("force_full", True):
            break

    model.load_state_dict(best_state)

    # Phase 1 v4-style lower-learning-rate fine-tune.
    opt_ft = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_ft, best_state_ft, no_imp = best, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
    for ep in range(1, LSTM_FINE_TUNE_EPOCHS + 1):
        model.train()
        for xb, yb_s, mb in dl:
            xb = xb.to(DEVICE)
            yb_s = yb_s.to(DEVICE)
            mb = mb.to(DEVICE)
            opt_ft.zero_grad()
            yp_s = model(xb)
            loss = F.smooth_l1_loss(yp_s, yb_s)
            if phys_weight > 0 or mono_weight > 0:
                yp = yp_s * y_std_t + y_mean_t
                if q_only_physics:
                    mono, phys, ea_prior, _ = q_only_physics_loss_terms(model, yp, mb, rate_scales)
                else:
                    mono, phys, ea_prior, _ = physics_loss_terms(model, yp, mb, rate_scales)
                loss = loss + mono_weight * mono + phys_weight * phys + ea_weight * ea_prior
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_ft.step()

        model.eval()
        with torch.no_grad():
            yv_pred = np.clip(sc_y.inverse_transform(model(Xvl_t).cpu().numpy()), [1e-6, 1e-9], None)
        val_direct = float(per_target_metrics(yvl, yv_pred).query("target == 'macro_avg'")["MAPE"].iloc[0])
        if val_direct < best_ft:
            best_ft = val_direct
            best_state_ft = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= 20:
            break

    model.load_state_dict(best_state_ft)
    model.eval()
    elapsed = time.time() - t0
    train_times[name] = elapsed

    # Rollout evaluation.
    yvl_roll_true, _ = rollout_truth(val_df, SEQ_LEN)
    yte_roll_true, _ = rollout_truth(test_df, SEQ_LEN)
    yvl_roll_pred = rollout_predict(model, sc_x, sc_y, val_df, feats, SEQ_LEN)
    yte_roll_pred = rollout_predict(model, sc_x, sc_y, test_df, feats, SEQ_LEN)

    val_pred_df = prediction_frame(val_df, yvl_roll_true, yvl_roll_pred, name, "val")
    test_pred_df = prediction_frame(test_df, yte_roll_true, yte_roll_pred, name, "test")
    val_diag = physics_diagnostics(val_pred_df)
    test_diag = physics_diagnostics(test_pred_df)

    record_metrics(name, "val", yvl_roll_true, yvl_roll_pred, extra=val_diag)
    record_metrics(name, "test", yte_roll_true, yte_roll_pred, extra=test_diag)

    # Save artifacts.
    model_path = MODEL_DIR / f"{slugify(name)}.pth"
    bundle_path = MODEL_DIR / f"{slugify(name)}_scalers_meta.pt"
    torch.save(model.state_dict(), model_path)
    torch.save(
        {
            "features": feats,
            "seq_len": SEQ_LEN,
            "sc_x": sc_x,
            "sc_y": sc_y,
            "candidate": candidate,
            "seed": seed,
            "rate_scales": rate_scales,
            "reference": PHASE1_REFERENCE,
        },
        bundle_path,
    )
    test_pred_df.to_csv(OUT_DIR / f"predictions_{slugify(name)}_test.csv", index=False)
    val_pred_df.to_csv(OUT_DIR / f"predictions_{slugify(name)}_val.csv", index=False)

    return {
        "name": name,
        "features": feats,
        "model_path": str(model_path),
        "bundle_path": str(bundle_path),
        "train_time_s": elapsed,
        "seed": seed,
        "val_diag": val_diag,
        "test_diag": test_diag,
    }

# %% [markdown]
# ## 6. Candidate Set
#
# The candidate set is deliberately small and paper-oriented:
#
# - `LSTM_v4_rebuilt_full`: rebuilt Phase 1 rollout baseline
# - `Hybrid_A_physics_features`: adds Ea and stress features
# - `Hybrid_B_PINN_LSTM_light`: adds light PINN-style physics loss
# - `Hybrid_B_PINN_LSTM_strong`: stronger physics regularization
# - `Hybrid_C_minimal_phys`: tests the cleaner cross-dataset feature subset

# %%
CANDIDATES = [
    {
        "name": "LSTM_v4_rebuilt_full",
        "features": RAW_FEATS_9,
        "phys_weight": 0.0,
        "mono_weight": 0.0,
        "force_full": True,
    },
    {
        "name": "Hybrid_A_physics_features",
        "features": PHYS_FEATS,
        "phys_weight": 0.0,
        "mono_weight": 0.0,
        "force_full": True,
    },
    {
        "name": "Hybrid_B_PINN_LSTM_light",
        "features": PHYS_FEATS,
        "phys_weight": 0.015,
        "mono_weight": 0.04,
        "ea_weight": 0.001,
        "force_full": True,
    },
    {
        "name": "Hybrid_B_PINN_LSTM_strong",
        "features": PHYS_FEATS,
        "phys_weight": 0.04,
        "mono_weight": 0.10,
        "ea_weight": 0.002,
        "force_full": True,
    },
    {
        "name": "Hybrid_C_minimal_phys",
        "features": MINIMAL_PHYS_FEATS,
        "phys_weight": 0.025,
        "mono_weight": 0.06,
        "ea_weight": 0.001,
        "force_full": True,
    },
    {
        "name": "Hybrid_D_Qphys_light",
        "features": PHYS_FEATS,
        "phys_weight": 0.005,
        "mono_weight": 0.015,
        "ea_weight": 0.0005,
        "q_only_physics": True,
        "force_full": True,
    },
    {
        "name": "Hybrid_E_Qphys_minimal",
        "features": MINIMAL_PHYS_FEATS,
        "phys_weight": 0.005,
        "mono_weight": 0.015,
        "ea_weight": 0.0005,
        "q_only_physics": True,
        "force_full": True,
    },
]

if RUN_FAST:
    CANDIDATES = CANDIDATES[:3]

print_section("Candidates")
for c in CANDIDATES:
    print(f"{c['name']:<32s} n_features={len(c['features'])} phys={c['phys_weight']} mono={c['mono_weight']}")

# %% [markdown]
# ## 7. Train and Evaluate

# %%
run_manifest = {
    "run_tag": RUN_TAG,
    "created_at": datetime.now().isoformat(),
    "device": str(DEVICE),
    "run_fast": RUN_FAST,
    "seq_len": SEQ_LEN,
    "phase1_reference": PHASE1_REFERENCE,
    "candidates": CANDIDATES,
    "artifacts": [],
}

for candidate in CANDIDATES:
    artifact = train_candidate(candidate)
    run_manifest["artifacts"].append(artifact)

with open(OUT_DIR / "phase5_manifest.json", "w", encoding="utf-8") as f:
    json.dump(run_manifest, f, indent=2)

print_section("Training Complete")
print(pd.DataFrame(train_times.items(), columns=["model", "train_time_s"]).to_string(index=False))

# %% [markdown]
# ## 8. Leaderboard and Diagnostics

# %%
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / "phase5_metrics_raw.csv", index=False)

summary = (
    metrics_df.groupby(["model", "split", "target"], as_index=False)[["MAPE", "RMSE", "NRMSE", "R2"]]
    .mean()
    .sort_values(["split", "target", "MAPE"])
)
summary.to_csv(OUT_DIR / "phase5_metrics_summary.csv", index=False)

test_macro = (
    summary[(summary["split"] == "test") & (summary["target"] == "macro_avg")]
    .sort_values("MAPE")
    .reset_index(drop=True)
)
test_macro["Rank"] = np.arange(1, len(test_macro) + 1)
test_macro["Delta_vs_Phase1_LSTM_v4"] = test_macro["MAPE"] - PHASE1_REFERENCE["LSTM_v4_rollout_tuned_macro_mape"]
test_macro = test_macro[["Rank", "model", "MAPE", "RMSE", "NRMSE", "R2", "Delta_vs_Phase1_LSTM_v4"]]
test_macro.to_csv(OUT_DIR / "phase5_test_leaderboard.csv", index=False)

diag_cols = [
    "model",
    "split",
    "target",
    "Q_monotonic_violations",
    "Re_monotonic_violations",
    "Q_violation_rate",
    "Re_violation_rate",
]
diag_df = metrics_df[[c for c in diag_cols if c in metrics_df.columns]].drop_duplicates()
diag_df.to_csv(OUT_DIR / "phase5_physics_diagnostics.csv", index=False)

print_section("Test Leaderboard")
print(test_macro.to_string(index=False))

print_section("Physics Diagnostics")
print(diag_df[diag_df["split"] == "test"].to_string(index=False))

# %% [markdown]
# ## 9. Figures

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
plot_df = test_macro.sort_values("MAPE", ascending=True)
ax.barh(plot_df["model"], plot_df["MAPE"], color="#2f6f9f", alpha=0.85)
ax.axvline(PHASE1_REFERENCE["LSTM_v4_rollout_tuned_macro_mape"], color="#c0392b", linestyle="--", linewidth=1.5)
ax.set_xlabel("Test macro MAPE (%)")
ax.set_title("Phase 5 Hybrid Leaderboard")
ax.grid(axis="x", alpha=0.3)
for i, v in enumerate(plot_df["MAPE"]):
    ax.text(v + 0.03, i, f"{v:.3f}%", va="center", fontsize=8)

ax = axes[1]
target_df = summary[(summary["split"] == "test") & (summary["target"].isin(TARGETS))].copy()
pivot = target_df.pivot(index="model", columns="target", values="MAPE").loc[plot_df["model"]]
x = np.arange(len(pivot))
w = 0.35
ax.bar(x - w / 2, pivot["Q"], width=w, label="Q", color="#1f77b4", alpha=0.85)
ax.bar(x + w / 2, pivot["Re"], width=w, label="Re", color="#ff7f0e", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("MAPE (%)")
ax.set_title("Target-wise Test Error")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "phase5_hybrid_leaderboard.png"
plt.savefig(fig_path, dpi=180, bbox_inches="tight")
plt.show()
print(f"Saved figure: {fig_path}")

# %% [markdown]
# ## 10. Export Zip

# %%
train_times_df = pd.DataFrame(
    [{"model": k, "train_time_s": v} for k, v in train_times.items()]
).sort_values("model")
train_times_df.to_csv(OUT_DIR / "phase5_train_times.csv", index=False)

zip_path = WORKING / f"{RUN_TAG}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fp in sorted(OUT_DIR.rglob("*")):
        if fp.is_file():
            zf.write(fp, fp.relative_to(OUT_DIR.parent))

print_section("Export")
print_kv("Output dir", OUT_DIR)
print_kv("Zip", zip_path)
print_kv("Zip MB", f"{zip_path.stat().st_size / 1024 / 1024:.2f}")
print("\nKey files:")
for name in [
    "phase5_test_leaderboard.csv",
    "phase5_metrics_summary.csv",
    "phase5_physics_diagnostics.csv",
    "phase5_hybrid_leaderboard.png",
    "phase5_manifest.json",
]:
    print(" -", OUT_DIR / name)

# %% [markdown]
# ## 11. Manuscript Interpretation Template
#
# Use this after the run completes:
#
# - If a hybrid beats 1.551% macro MAPE, the revised paper can claim that
#   physics regularization improves deployment-realistic rollout forecasting.
# - If the hybrid does not beat LSTM v4 but reduces monotonicity violations, the
#   claim should be weaker: physics improves physical coherence at little or no
#   accuracy cost.
# - If the physics-feature candidate wins, the story is that PINN-derived
#   activation-energy/stress features help LSTM forecast degradation without
#   needing a heavy PINN predictor at inference.
#
# Do not use teacher-forced LSTM v1 as the main comparison. It remains an upper
# bound, not the deployment baseline.
