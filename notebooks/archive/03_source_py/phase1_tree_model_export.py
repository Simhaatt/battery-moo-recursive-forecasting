# %% [markdown]
# # Phase 1c - Export RF/XGBoost Models for Phase 4b
#
# Run this notebook on Kaggle with the Luh & Blank processed dataset attached.
# It trains frozen Phase 1 tree baselines and exports `phase1_tree_model_bundle.joblib`.
#
# Upload the resulting zip as a Kaggle input dataset for Phase 4b.

# %%
import json
import os
import time
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception as exc:
    HAS_XGB = False
    XGB_IMPORT_ERROR = repr(exc)

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
if not KAGGLE_INPUT.exists():
    KAGGLE_INPUT = Path(".")
    WORKING = Path(".")

RUN_TAG = "phase1_tree_models_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORKING / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGETS = ["Q", "Re"]

print("Run tag:", RUN_TAG)
print("Output:", OUT_DIR)
print("XGBoost available:", HAS_XGB)

# %% [markdown]
# ## 1. Load Processed Phase 1 Splits

# %%
def discover_pack_dir():
    candidates = [
        KAGGLE_INPUT / "datasets" / "simhaatt" / "outputs-of-luh-and-blank",
        KAGGLE_INPUT / "outputs-of-luh-and-blank",
        KAGGLE_INPUT / "battery-outputs" / "outputs-of-luh-and-blank",
    ]
    for p in candidates:
        if (p / "phase2_train.csv").exists() and (p / "phase2_val.csv").exists() and (p / "phase2_test.csv").exists():
            return p
    for p in [KAGGLE_INPUT, WORKING]:
        for q in p.rglob("phase2_train.csv"):
            root = q.parent
            if (root / "phase2_val.csv").exists() and (root / "phase2_test.csv").exists():
                return root
    raise FileNotFoundError("Attach the Luh & Blank processed dataset containing phase2_train/val/test.csv")


PACK_DIR = discover_pack_dir()
train_df = pd.read_csv(PACK_DIR / "phase2_train.csv")
val_df = pd.read_csv(PACK_DIR / "phase2_val.csv")
test_df = pd.read_csv(PACK_DIR / "phase2_test.csv")

for d in [train_df, val_df, test_df]:
    if "cell_key" not in d.columns and "cell_id" in d.columns:
        d["cell_key"] = d["cell_id"]

print("Pack dir:", PACK_DIR)
print("Train/val/test:", train_df.shape, val_df.shape, test_df.shape)
print("Cells:", train_df["cell_key"].nunique(), val_df["cell_key"].nunique(), test_df["cell_key"].nunique())

# %% [markdown]
# ## 2. Feature Sets and Metrics

# %%
POINTWISE_FEATURE_SETS = {
    "full9": [
        "k_exp", "temperature", "c_rate_chg", "c_rate_dischg",
        "soc_window", "age_type", "Q0", "Re0", "Rct0",
    ],
    "sparse_k_Re0": ["k_exp", "Re0"],
    "sparse_k_Re0_Q0": ["k_exp", "Re0", "Q0"],
    "sparse_k_Re0_Rct0_Q0": ["k_exp", "Re0", "Rct0", "Q0"],
}

AUTOREG_FEATURE_SETS = {
    "full9_prevQR": POINTWISE_FEATURE_SETS["full9"] + ["Q_prev", "Re_prev"],
    "sparse_k_Re0_prevQR": POINTWISE_FEATURE_SETS["sparse_k_Re0"] + ["Q_prev", "Re_prev"],
    "sparse_k_Re0_Q0_prevQR": POINTWISE_FEATURE_SETS["sparse_k_Re0_Q0"] + ["Q_prev", "Re_prev"],
    "sparse_k_Re0_Rct0_Q0_prevQR": POINTWISE_FEATURE_SETS["sparse_k_Re0_Rct0_Q0"] + ["Q_prev", "Re_prev"],
}


def clean_frame(df, feature_cols):
    d = df.copy()
    for c in feature_cols + TARGETS:
        if c not in d.columns:
            raise KeyError(f"Missing column {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.dropna(subset=feature_cols + TARGETS).copy()


def add_prev_targets(df):
    d = df.sort_values(["cell_key", "k_exp"]).copy()
    d["Q_prev"] = d.groupby("cell_key")["Q"].shift(1)
    d["Re_prev"] = d.groupby("cell_key")["Re"].shift(1)
    return d.dropna(subset=["Q_prev", "Re_prev"]).copy()


def macro_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = []
    for j in range(y_true.shape[1]):
        denom = np.maximum(np.abs(y_true[:, j]), 1e-9)
        out.append(float(np.mean(np.abs((y_true[:, j] - y_pred[:, j]) / denom)) * 100.0))
    return float(np.mean(out))


def metric_row(model_name, protocol, feature_set, split, y_true, y_pred):
    return {
        "model": model_name,
        "protocol": protocol,
        "feature_set": feature_set,
        "split": split,
        "macro_mape": macro_mape(y_true, y_pred),
        "Q_mape": macro_mape(y_true[:, [0]], y_pred[:, [0]]),
        "Re_mape": macro_mape(y_true[:, [1]], y_pred[:, [1]]),
        "Q_rmse": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "Re_rmse": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "Q_r2": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "Re_r2": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "n": int(len(y_true)),
    }


def rollout_predict(model, df, feature_cols):
    rows = []
    for cell_key, grp in df.sort_values(["cell_key", "k_exp"]).groupby("cell_key", sort=False):
        g = grp.reset_index(drop=True).copy()
        if len(g) < 2:
            continue
        q_prev = float(g.loc[0, "Q"])
        re_prev = float(g.loc[0, "Re"])
        for i in range(1, len(g)):
            r = g.loc[i].copy()
            r["Q_prev"] = q_prev
            r["Re_prev"] = re_prev
            X = pd.DataFrame([r[feature_cols].astype(float).to_dict()])
            yp = np.asarray(model.predict(X), dtype=float).reshape(-1, 2)[0]
            yp = np.clip(yp, [1e-6, 1e-9], None)
            rows.append({
                "cell_key": cell_key,
                "k_exp": float(r["k_exp"]),
                "Q": float(r["Q"]),
                "Re": float(r["Re"]),
                "pred_Q": float(yp[0]),
                "pred_Re": float(yp[1]),
            })
            q_prev, re_prev = float(yp[0]), float(yp[1])
    return pd.DataFrame(rows)

# %% [markdown]
# ## 3. Train and Save Frozen Models

# %%
def make_models():
    models = {
        "RF": RandomForestRegressor(
            n_estimators=700, max_depth=None, min_samples_leaf=1,
            max_features=0.8, random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=900, max_depth=None, min_samples_leaf=1,
            max_features=0.9, random_state=SEED, n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = MultiOutputRegressor(XGBRegressor(
            n_estimators=700, max_depth=4, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
            objective="reg:squarederror", tree_method="hist",
            random_state=SEED, n_jobs=2,
        ))
    return models


bundle_models = {}
metrics = []
fit_times = []

for protocol, feature_sets in [
    ("pointwise", POINTWISE_FEATURE_SETS),
    ("autoregressive_rollout", AUTOREG_FEATURE_SETS),
]:
    for feature_set, feature_cols in feature_sets.items():
        if protocol == "pointwise":
            tr = clean_frame(train_df, feature_cols)
            vl = clean_frame(val_df, feature_cols)
            te = clean_frame(test_df, feature_cols)
            eval_payload = {
                "val": (vl[TARGETS].to_numpy(float), vl[feature_cols]),
                "test": (te[TARGETS].to_numpy(float), te[feature_cols]),
            }
        else:
            tr = clean_frame(add_prev_targets(train_df), feature_cols)
            vl_full = clean_frame(val_df, [c for c in feature_cols if c not in ["Q_prev", "Re_prev"]])
            te_full = clean_frame(test_df, [c for c in feature_cols if c not in ["Q_prev", "Re_prev"]])
            eval_payload = {"val": vl_full, "test": te_full}

        Xtr = tr[feature_cols]
        ytr = tr[TARGETS].to_numpy(float)

        for model_name, model in make_models().items():
            key = f"{model_name}__{protocol}__{feature_set}"
            t0 = time.time()
            model.fit(Xtr, ytr)
            fit_times.append({"key": key, "seconds": time.time() - t0, "n_train": int(len(Xtr))})

            bundle_models[key] = {
                "model": model,
                "model_name": model_name,
                "protocol": protocol,
                "feature_set": feature_set,
                "feature_cols": feature_cols,
                "target_cols": TARGETS,
                "seed": SEED,
            }

            if protocol == "pointwise":
                for split, (y_true, X_eval) in eval_payload.items():
                    y_pred = np.asarray(model.predict(X_eval), dtype=float)
                    metrics.append(metric_row(model_name, protocol, feature_set, split, y_true, y_pred))
            else:
                for split, df_eval in eval_payload.items():
                    pred = rollout_predict(model, df_eval, feature_cols)
                    if len(pred):
                        y_true = pred[TARGETS].to_numpy(float)
                        y_pred = pred[["pred_Q", "pred_Re"]].to_numpy(float)
                        metrics.append(metric_row(model_name, protocol, feature_set, split, y_true, y_pred))

            print("trained", key)

metrics_df = pd.DataFrame(metrics).sort_values(["split", "macro_mape"])
fit_df = pd.DataFrame(fit_times)
metrics_df.to_csv(OUT_DIR / "phase1_tree_model_metrics.csv", index=False)
fit_df.to_csv(OUT_DIR / "phase1_tree_model_fit_times.csv", index=False)
display(metrics_df[metrics_df["split"] == "test"].head(20))

# %% [markdown]
# ## 4. Export Bundle Zip

# %%
manifest = {
    "run_tag": RUN_TAG,
    "created_at": datetime.now().isoformat(),
    "pack_dir": str(PACK_DIR),
    "has_xgboost": HAS_XGB,
    "xgboost_import_error": None if HAS_XGB else XGB_IMPORT_ERROR,
    "target_cols": TARGETS,
    "pointwise_feature_sets": POINTWISE_FEATURE_SETS,
    "autoregressive_feature_sets": AUTOREG_FEATURE_SETS,
    "model_keys": sorted(bundle_models.keys()),
    "required_for_phase4b": [
        "phase1_tree_model_bundle.joblib",
        "phase1_tree_model_manifest.json",
    ],
}

joblib_path = OUT_DIR / "phase1_tree_model_bundle.joblib"
joblib.dump({"models": bundle_models, "manifest": manifest}, joblib_path, compress=3)

(OUT_DIR / "phase1_tree_model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

zip_path = WORKING / f"{RUN_TAG}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in OUT_DIR.glob("*"):
        z.write(f, arcname=f.name)

print("Bundle:", joblib_path)
print("Zip:", zip_path)
print("Upload this zip as a Kaggle dataset/input for Phase 4b.")

try:
    from IPython.display import FileLink
    display(FileLink(str(zip_path)))
except Exception:
    pass
