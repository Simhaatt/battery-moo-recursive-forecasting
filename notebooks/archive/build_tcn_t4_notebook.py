"""Build the self-contained Kaggle T4 TCN latency notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_INPUTS = REPO.parent / "paper_revision_20260625" / "07_round2_revision" / "inputs"
OUTPUT = REPO / "notebooks" / "kaggle" / "ASC_TCN_T4_five_seed_latency.ipynb"


def cell(kind: str, source: str) -> dict:
    cell_id = hashlib.sha1(f"{kind}\0{source}".encode("utf-8")).hexdigest()[:12]
    result = {
        "id": cell_id,
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def packed_assets_cell() -> str:
    assets = {
        "run_tcn_t4_latency.py": HERE / "run_tcn_t4_latency.py",
        "inputs/phase1_cv_all_rows.csv": SOURCE_INPUTS / "phase1_cv_all_rows.csv",
        "inputs/fixed_split_val_predictions.csv": (
            SOURCE_INPUTS / "fixed_split_val_predictions.csv"
        ),
        "inputs/fixed_split_test_predictions.csv": (
            SOURCE_INPUTS / "fixed_split_test_predictions.csv"
        ),
    }
    missing = [str(path) for path in assets.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing notebook assets: " + ", ".join(missing))
    packed = {
        name: base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode("ascii")
        for name, path in assets.items()
    }
    payload = json.dumps(packed, separators=(",", ":"))
    return f'''from pathlib import Path
import base64, gzip, json

ROOT = Path("/kaggle/working/asc_tcn_t4_suite")
ROOT.mkdir(parents=True, exist_ok=True)
packed = json.loads({payload!r})
for name, encoded in packed.items():
    destination = ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.decompress(base64.b64decode(encoded)))
print(f"Materialized {{len(packed)}} files ({{sum((ROOT / name).stat().st_size for name in packed):,}} bytes)")
'''


def build() -> None:
    notebook = {
        "cells": [
            cell("markdown", """# ASC TCN five-seed T4 rerun with trajectory latency

This notebook reruns the manuscript TCN baseline using seeds **42, 52, 62, 72, and 82** on an
NVIDIA Tesla T4. It preserves the original nine descriptors, sequence length 20, recursive test
protocol, optimizer, early stopping, and 119,234-parameter architecture.

The latency column is a synchronized end-to-end recursive measurement in milliseconds per
evaluable test-cell trajectory. It includes input scaling, host-to-device transfer, batch-one TCN
forward passes, device-to-host transfer, inverse scaling, clipping, and autoregressive feedback.
Each seed uses five warm-up rollouts and 100 timed repetitions.

No Kaggle dataset needs to be attached: the three processed input CSVs are embedded. In Kaggle,
select **Settings > Accelerator > GPU T4 x2** (the code uses GPU 0) and run all cells. Download
`/kaggle/working/asc_tcn_t4_latency_results.zip` when finished.
"""),
            cell("code", """import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook.")
gpu = torch.cuda.get_device_name(0)
print("GPU:", gpu)
if "T4" not in gpu.upper():
    raise RuntimeError(f"Select a Kaggle Tesla T4 accelerator; detected {gpu}")

SMOKE = False  # Keep False for the publishable five-seed run.
LATENCY_REPEATS = 100
"""),
            cell("code", packed_assets_cell()),
            cell("code", """import subprocess, sys, time
root = "/kaggle/working/asc_tcn_t4_suite"
command = [
    sys.executable,
    f"{root}/run_tcn_t4_latency.py",
    "--input-dir", f"{root}/inputs",
    "--output-dir", f"{root}/outputs",
    "--archive", "/kaggle/working/asc_tcn_t4_latency_results.zip",
    "--latency-repeats", str(LATENCY_REPEATS),
]
if SMOKE:
    command.append("--smoke")
started = time.time()
subprocess.run(command, check=True)
print("Elapsed minutes:", (time.time() - started) / 60)
"""),
            cell("code", """from pathlib import Path
import json, pandas as pd
from IPython.display import FileLink, display

output = Path("/kaggle/working/asc_tcn_t4_suite/outputs")
seed_metrics = pd.read_csv(output / "tcn_t4_seed_metrics.csv")
summary = pd.read_csv(output / "tcn_t4_summary.csv", index_col=0)
display(seed_metrics[[
    "seed", "macro_MAPE", "Q_MAPE", "Re_MAPE", "parameters",
    "train_seconds", "inference_ms_per_trajectory",
    "inference_ms_per_trajectory_repeat_std", "n_eval_rows", "n_eval_cells", "epochs",
]])
display(summary)
print(json.dumps(json.loads((output / "tcn_t4_environment_manifest.json").read_text()), indent=2))
archive = Path("/kaggle/working/asc_tcn_t4_latency_results.zip")
print("DOWNLOAD THIS FILE:", archive, f"({archive.stat().st_size / 1e6:.2f} MB)")
display(FileLink(str(archive)))
"""),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(OUTPUT)
    print(f"Notebook size: {OUTPUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    build()
