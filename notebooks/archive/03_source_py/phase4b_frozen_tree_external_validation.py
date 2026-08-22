# %% [markdown]
# # Phase 4b - Frozen Phase 1 Tree Models on Kirkaldy External Data
#
# **Kaggle inputs to attach**
#
# 1. Kirkaldy battery dataset, containing `Performance Summary/*.csv`.
# 2. Phase 1 tree model dataset produced by `phase1_tree_model_export.ipynb`.
#    It must contain:
#    - `phase1_tree_model_bundle.joblib`
#    - `phase1_tree_model_manifest.json`
#
# This notebook does not train on Kirkaldy. It evaluates frozen Phase 1
# RF/ExtraTrees/XGBoost models on the external 21700 data and exports a zip.

# %%
import json
import re
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
if not KAGGLE_INPUT.exists():
    KAGGLE_INPUT = Path(".")
    WORKING = Path(".")

RUN_TAG = "phase4b_frozen_tree_external_" + datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = WORKING / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9
print("Run tag:", RUN_TAG)
print("Output:", OUT_DIR)

# %% [markdown]
# ## 1. Input Discovery

# %%
def find_file(name):
    hits = list(KAGGLE_INPUT.rglob(name)) + list(WORKING.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found. Attach the Phase 1 tree model zip as a Kaggle input.")
    return hits[0]


def normalize_name(s):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def count_expts(p):
    try:
        return len([d for d in p.iterdir() if d.is_dir() and "expt" in d.name.lower()])
    except Exception:
        return 0


def discover_kirk_root():
    perf_dirs = [p for p in KAGGLE_INPUT.rglob("*") if p.is_dir() and p.name == "Performance Summary"]
    if perf_dirs:
        candidates = [p.parent.parent for p in perf_dirs]
        extended = candidates + [p.parent for p in candidates]
        return max(extended, key=count_expts)
    candidates = [p for p in KAGGLE_INPUT.rglob("*") if p.is_dir() and any(t in p.name.lower() for t in ["kirkal", "kirkald"])]
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Kirkaldy dataset root not found. Attach kirkaldt-battery-dataset.")


TREE_BUNDLE_PATH = find_file("phase1_tree_model_bundle.joblib")
TREE_MANIFEST_PATH = find_file("phase1_tree_model_manifest.json")
KIRK_ROOT = discover_kirk_root()

bundle = joblib.load(TREE_BUNDLE_PATH)
tree_models = bundle["models"]
tree_manifest = bundle.get("manifest", {})

print("Kirkaldy root:", KIRK_ROOT)
print("Tree bundle:", TREE_BUNDLE_PATH)
print("Model count:", len(tree_models))
print("First models:", list(tree_models.keys())[:5])

# %% [markdown]
# ## 2. Build Kirkaldy Phase 1-Compatible Feature Table

# %%
EXP_SOC_MAP = {
    1: (0.00, 0.30),
    2: (0.70, 0.85),
    3: (0.85, 1.00),
    4: (0.00, 1.00),
    5: (0.00, 1.00),
}


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


def find_col(columns, keywords, exclude=None):
    for kw in keywords:
        for c in columns:
            if kw in c and not (exclude and any(ex in c for ex in exclude)):
                return c
    return None


perf_csvs = sorted(KIRK_ROOT.rglob("Performance Summary/*.csv"))
perf_csvs = [p for p in perf_csvs if not any(x in p.name.lower() for x in ["charge_data", "discharge_data", "voltage", "timeseries"])]
print("Performance Summary CSVs:", len(perf_csvs))

rows = []
parse_errors = []

for path in perf_csvs:
    meta = parse_kirkaldy_filename(path)
    if meta["exp"] is None or meta["cell_id"] is None:
        parse_errors.append({"path": str(path), "reason": "filename_parse_failed"})
        continue

    cell_key = f"exp{meta['exp']}_cell{meta['cell_id']}"
    soc_lo, soc_hi = EXP_SOC_MAP.get(meta["exp"], (0.0, 1.0))

    try:
        d = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        d = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    d.columns = [normalize_name(c) for c in d.columns]

    c_rpt = find_col(d.columns, ["ageing_sets", "ageing_set", "rpt"])
    c_cycles = find_col(d.columns, ["ageing_cycles", "ageing_cycle", "cycles"])
    c_temp = find_col(d.columns, ["age_set_av_temperature", "temperature"])
    c_cap10 = find_col(d.columns, ["c_10", "c10"], exclude=["c_2", "c2"])
    c_cap2 = find_col(d.columns, ["c_2", "c2"])
    c_res = find_col(d.columns, ["0_1s_resist", "0_1s_res", "resistance"])
    c_days = find_col(d.columns, ["days", "time"])

    if c_cycles is None or (c_cap10 is None and c_cap2 is None):
        parse_errors.append({"path": str(path), "reason": "missing_capacity_or_cycle_columns", "columns": "|".join(d.columns)})
        continue

    for j, r in d.iterrows():
        q_mah = pd.to_numeric(r.get(c_cap10 if c_cap10 else c_cap2), errors="coerce")
        cyc = pd.to_numeric(r.get(c_cycles), errors="coerce")
        if not np.isfinite(q_mah) or not np.isfinite(cyc):
            continue
        re_ohm = pd.to_numeric(r.get(c_res), errors="coerce") if c_res else np.nan
        rpt = pd.to_numeric(r.get(c_rpt), errors="coerce") if c_rpt else j
        temp = pd.to_numeric(r.get(c_temp), errors="coerce") if c_temp else meta["temp_c"]
        days = pd.to_numeric(r.get(c_days), errors="coerce") if c_days else np.nan
        rows.append({
            "cell_key": cell_key,
            "exp": meta["exp"],
            "cell_id": meta["cell_id"],
            "rpt_idx": float(rpt) if np.isfinite(rpt) else float(j),
            "ageing_cycles": float(cyc),
            "days": float(days) if np.isfinite(days) else np.nan,
            "temperature": float(temp) if np.isfinite(temp) else float(meta["temp_c"] or 25.0),
            "c_rate_chg": 0.3,
            "c_rate_dischg": 1.0,
            "soc_window": float(soc_hi - soc_lo),
            "age_type": 3.0 if meta["exp"] == 4 else 2.0,
            "Q": float(q_mah) / 1000.0,
            "Re": float(re_ohm) * 1000.0 if np.isfinite(re_ohm) else np.nan,
            "q_src": "c10" if c_cap10 else "c2",
            "path": str(path),
        })

raw = pd.DataFrame(rows)
pd.DataFrame(parse_errors).to_csv(OUT_DIR / "parse_errors.csv", index=False)
if raw.empty:
    raise RuntimeError("No Kirkaldy rows parsed. Check parse_errors.csv.")

raw = raw.sort_values(["cell_key", "rpt_idx", "ageing_cycles"]).drop_duplicates(["cell_key", "rpt_idx"], keep="last")
raw = raw[raw["Q"].notna() & (raw["Q"] > 0.1)].copy()
raw["Re"] = raw.groupby("cell_key")["Re"].transform(lambda s: s.interpolate(limit_direction="both").fillna(s.median()))
raw["Re"] = raw["Re"].fillna(raw["Re"].median())

raw["k_exp_raw"] = raw.groupby("cell_key").cumcount()
raw["k_exp"] = raw.groupby("cell_key")["k_exp_raw"].transform(lambda s: s / max(float(s.max()), 1.0))
raw["Q0"] = raw.groupby("cell_key")["Q"].transform("first")
raw["Re0"] = raw.groupby("cell_key")["Re"].transform("first")
raw["Rct0"] = raw["Re0"] * 1.5

temp_k = raw["temperature"] + 273.15
raw["stress"] = raw["c_rate_chg"].abs() * (raw["soc_window"].abs() + 1e-6) * np.exp((temp_k - 298.15) / 50.0)

phase4b_df = raw.sort_values(["cell_key", "k_exp"]).reset_index(drop=True)
phase4b_df.to_csv(OUT_DIR / "phase4b_kirkaldy_features.csv", index=False)

print("Feature table:", phase4b_df.shape)
print("Cells:", phase4b_df["cell_key"].nunique())
display(phase4b_df.head())

# %% [markdown]
# ## 3. Frozen External Inference

# %%
def metric_block(y_true, y_pred, prefix):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) == 0:
        return {f"{prefix}_n": 0, f"{prefix}_mae": np.nan, f"{prefix}_rmse": np.nan, f"{prefix}_mape": np.nan, f"{prefix}_r2": np.nan}
    denom = np.maximum(np.abs(y_true), 1e-9)
    return {
        f"{prefix}_n": int(len(y_true)),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mape": float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 3 and np.var(y_true) > 1e-12 else np.nan,
    }


def eval_pointwise(entry, df):
    feature_cols = entry["feature_cols"]
    d = df.copy()
    d = d.groupby("cell_key", group_keys=False).apply(lambda g: g.iloc[1:]).reset_index(drop=True)
    for c in feature_cols:
        if c not in d.columns:
            d[c] = 0.0
    d = d.dropna(subset=feature_cols + ["Q", "Re"]).copy()
    if d.empty:
        return pd.DataFrame()
    yp = np.asarray(entry["model"].predict(d[feature_cols]), dtype=float).reshape(-1, 2)
    yp = np.clip(yp, [1e-6, 1e-9], None)
    out = d[["cell_key", "exp", "cell_id", "k_exp", "ageing_cycles", "Q", "Re"]].copy()
    out["pred_Q"] = yp[:, 0]
    out["pred_Re"] = yp[:, 1]
    return out


def eval_rollout(entry, df):
    feature_cols = entry["feature_cols"]
    rows = []
    base_cols = [c for c in feature_cols if c not in ["Q_prev", "Re_prev"]]
    for _, grp in df.dropna(subset=base_cols + ["Q", "Re"]).sort_values(["cell_key", "k_exp"]).groupby("cell_key", sort=False):
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
            yp = np.asarray(entry["model"].predict(X), dtype=float).reshape(-1, 2)[0]
            yp = np.clip(yp, [1e-6, 1e-9], None)
            rows.append({
                "cell_key": r["cell_key"],
                "exp": int(r["exp"]),
                "cell_id": r["cell_id"],
                "k_exp": float(r["k_exp"]),
                "ageing_cycles": float(r["ageing_cycles"]),
                "Q": float(r["Q"]),
                "Re": float(r["Re"]),
                "pred_Q": float(yp[0]),
                "pred_Re": float(yp[1]),
            })
            q_prev, re_prev = float(yp[0]), float(yp[1])
    return pd.DataFrame(rows)


pred_frames = []
metrics_rows = []

for key, entry in tree_models.items():
    protocol = entry.get("protocol", "")
    pred = eval_rollout(entry, phase4b_df) if protocol == "autoregressive_rollout" else eval_pointwise(entry, phase4b_df)
    if pred.empty:
        metrics_rows.append({"model_key": key, "status": "empty_predictions"})
        continue
    pred["model_key"] = key
    pred["model_name"] = entry.get("model_name")
    pred["protocol"] = protocol
    pred["feature_set"] = entry.get("feature_set")
    pred_frames.append(pred)

    q = metric_block(pred["Q"], pred["pred_Q"], "Q")
    re_diag = metric_block(pred["Re"], pred["pred_Re"], "Re_diagnostic")
    metrics_rows.append({
        "model_key": key,
        "model_name": entry.get("model_name"),
        "protocol": protocol,
        "feature_set": entry.get("feature_set"),
        "status": "ok",
        "n_cells": int(pred["cell_key"].nunique()),
        **q,
        **re_diag,
    })
    print("evaluated", key, "Q_MAPE", f"{q['Q_mape']:.2f}%")

predictions_long = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
metrics = pd.DataFrame(metrics_rows).sort_values("Q_mape", na_position="last")

predictions_long.to_csv(OUT_DIR / "phase4b_frozen_tree_predictions_long.csv", index=False)
metrics.to_csv(OUT_DIR / "phase4b_frozen_tree_metrics.csv", index=False)

display(metrics.head(30))

# %% [markdown]
# ## 4. Figures and Diagnostics

# %%
best = metrics[metrics["status"] == "ok"].head(12).copy()
if len(best):
    plt.figure(figsize=(9, 4.8))
    labels = best["model_name"] + "\n" + best["protocol"].str.replace("_", " ") + "\n" + best["feature_set"]
    plt.bar(range(len(best)), best["Q_mape"], color="#4C78A8", edgecolor="black", linewidth=0.5)
    plt.xticks(range(len(best)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("External capacity Q-MAPE (%)")
    plt.title("Phase 4b frozen tree-model external transfer")
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(best["Q_mape"]):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase4b_best_tree_external_qmape.png", dpi=250, bbox_inches="tight")
    plt.show()

    best_key = best.iloc[0]["model_key"]
    cells = sorted(predictions_long["cell_key"].dropna().unique())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.ravel()
    for ax, cell in zip(axes, cells):
        truth = phase4b_df[phase4b_df["cell_key"] == cell].sort_values("ageing_cycles")
        sub = predictions_long[(predictions_long["model_key"] == best_key) & (predictions_long["cell_key"] == cell)].sort_values("ageing_cycles")
        ax.plot(truth["ageing_cycles"], truth["Q"], color="black", lw=2.0, marker="o", label="Measured")
        ax.plot(sub["ageing_cycles"], sub["pred_Q"], color="#4C78A8", lw=2.0, marker=".", label="Predicted")
        ax.set_title(str(cell), fontsize=10)
        ax.set_xlabel("Ageing cycles")
        ax.set_ylabel("Q (Ah)")
        ax.grid(alpha=0.25)
    for ax in axes[len(cells):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Best Phase 4b external trajectories: {best_key}", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase4b_best_tree_external_trajectories.png", dpi=250, bbox_inches="tight")
    plt.show()

print("Interpretation:")
print("- Q metrics are the primary external-transfer endpoint.")
print("- Re metrics are diagnostic only because Kirkaldy resistance is pulse resistance, not Phase 1 EIS Re.")
print("- No Kirkaldy labels were used for training or calibration.")

# %% [markdown]
# ## 5. Export Zip

# %%
manifest = {
    "run_tag": RUN_TAG,
    "created_at": datetime.now().isoformat(),
    "kirk_root": str(KIRK_ROOT),
    "tree_bundle_path": str(TREE_BUNDLE_PATH),
    "tree_manifest": tree_manifest,
    "files": [
        "phase4b_kirkaldy_features.csv",
        "phase4b_frozen_tree_predictions_long.csv",
        "phase4b_frozen_tree_metrics.csv",
        "phase4b_best_tree_external_qmape.png",
        "phase4b_best_tree_external_trajectories.png",
        "parse_errors.csv",
    ],
}
(OUT_DIR / "phase4b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

zip_path = WORKING / f"{RUN_TAG}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in OUT_DIR.glob("*"):
        z.write(f, arcname=f.name)

print("Zip:", zip_path)
for f in sorted(OUT_DIR.glob("*")):
    print(" -", f.name)

try:
    from IPython.display import FileLink
    display(FileLink(str(zip_path)))
except Exception:
    pass
