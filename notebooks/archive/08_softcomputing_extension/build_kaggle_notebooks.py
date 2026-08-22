"""Build self-contained Kaggle notebooks for the Soft Computing GPU runs."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROUND2 = ROOT.parent / "07_round2_revision"
NOTEBOOKS = ROOT / "kaggle_notebooks"
NOTEBOOKS.mkdir(parents=True, exist_ok=True)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def packed_assets(paths: dict[str, Path]) -> str:
    packed = {
        destination: base64.b64encode(gzip.compress(source.read_bytes(), compresslevel=9)).decode("ascii")
        for destination, source in paths.items()
    }
    payload = json.dumps(packed, separators=(",", ":"))
    return f'''from pathlib import Path
import base64, gzip, json

ROOT = Path("/kaggle/working/softcomputing_suite")
ROOT.mkdir(parents=True, exist_ok=True)
packed = json.loads({payload!r})
for relative_path, encoded in packed.items():
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.decompress(base64.b64decode(encoded)))
print("Materialized", len(packed), "files in", ROOT)
print("Total bytes:", sum((ROOT / name).stat().st_size for name in packed))
'''


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def core_assets() -> dict[str, Path]:
    return {
        "round2_core/run_round2_experiments.py": ROUND2 / "run_round2_experiments.py",
        "round2_core/run_metaheuristic_search.py": ROUND2 / "run_metaheuristic_search.py",
        "round2_core/inputs/phase1_cv_all_rows.csv": ROUND2 / "inputs" / "phase1_cv_all_rows.csv",
        "round2_core/inputs/fixed_split_val_predictions.csv": ROUND2 / "inputs" / "fixed_split_val_predictions.csv",
        "round2_core/inputs/fixed_split_test_predictions.csv": ROUND2 / "inputs" / "fixed_split_test_predictions.csv",
    }


def build_search_notebook() -> None:
    assets = core_assets() | {"run_softcomputing_search.py": ROOT / "run_softcomputing_search.py"}
    cells = [
        markdown_cell(
            """# Soft Computing revision: matched-budget multi-objective search

This notebook runs NSGA-II and random search with **32 unique trained configurations each**.
Both methods use the same LSTM search space, data split, seed, training schedule, and three
minimization objectives: validation rollout macro MAPE, trainable parameters, and measured
batch-1 forward latency. Five-seed confirmation touches the fixed test trajectories only after
three-seed stochastic re-screening selects one candidate per search method. Every candidate is
evaluated on the same 76 test points beginning at row 20. The search uses `L in {10,15,20}` and
enforces `L + H <= 29`, so every rollout horizon has at least one valid training segment; the
legacy manual `L=20, H=10` reference is reported as the nearest feasible `L=20, H=8` reference.
"""
        ),
        code_cell(
            """import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook.")
print("GPU:", torch.cuda.get_device_name(0))
"""
        ),
        code_cell(packed_assets(assets)),
        code_cell(
            """import subprocess, sys, time
start = time.time()
subprocess.run(
    [sys.executable, "/kaggle/working/softcomputing_suite/run_softcomputing_search.py"],
    check=True,
)
print("Total notebook experiment hours:", (time.time() - start) / 3600)
"""
        ),
        code_cell(
            """from pathlib import Path
import shutil
output = Path("/kaggle/working/softcomputing_suite/outputs_softcomputing_search")
archive = shutil.make_archive("/kaggle/working/softcomputing_nsga2_random_results", "zip", output)
print("Created:", archive)
print("Output files:")
for path in sorted(output.iterdir()):
    print(path.name, path.stat().st_size)
"""
        ),
    ]
    path = NOTEBOOKS / "softcomputing_nsga2_random_search_FINAL_v3.ipynb"
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(path)


def build_mamba_notebook() -> None:
    assets = core_assets() | {"run_mamba_baseline.py": ROOT / "run_mamba_baseline.py"}
    cells = [
        markdown_cell(
            """# Soft Computing revision: official Mamba rollout baseline

This notebook installs the official `state-spaces/mamba` package and evaluates a two-block
Mamba model with the feasible `L=15, H=10` setting under the same grouped split, input set,
and five seeds used for the rollout-tuned LSTM comparison. Autoregressive scoring starts at
row 20, giving the same 76 test points used by the LSTM and TCN. The implementation and package
version are written to the output manifest.
"""
        ),
        code_cell(
            """import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook.")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA runtime:", torch.version.cuda)
print("CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI)
"""
        ),
        code_cell(
            """import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ninja", "packaging", "wheel"], check=True)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "mamba-ssm", "--no-build-isolation"],
    check=True,
)
from mamba_ssm import Mamba
import importlib.metadata
print("mamba-ssm:", importlib.metadata.version("mamba-ssm"))
"""
        ),
        code_cell(packed_assets(assets)),
        code_cell(
            """import subprocess, sys, time
start = time.time()
subprocess.run(
    [sys.executable, "/kaggle/working/softcomputing_suite/run_mamba_baseline.py"],
    check=True,
)
print("Total notebook experiment hours:", (time.time() - start) / 3600)
"""
        ),
        code_cell(
            """from pathlib import Path
import shutil
output = Path("/kaggle/working/softcomputing_suite/outputs_mamba")
archive = shutil.make_archive("/kaggle/working/softcomputing_mamba_results", "zip", output)
print("Created:", archive)
for path in sorted(output.iterdir()):
    print(path.name, path.stat().st_size)
"""
        ),
    ]
    path = NOTEBOOKS / "softcomputing_mamba_baseline_FINAL_v3.ipynb"
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(path)


def build_transfer_notebook() -> None:
    assets = core_assets() | {
        "run_kirkaldy40_transfer_audit.py": ROOT / "run_kirkaldy40_transfer_audit.py",
        "kirkaldy_40_normalized_features.csv": ROOT / "kirkaldy_40_normalized_features.csv",
        "kirkaldy_40_dataset_summary.json": ROOT / "kirkaldy_40_dataset_summary.json",
    }
    cells = [
        markdown_cell(
            """# Soft Computing revision: complete 40-cell Kirkaldy transfer audit

This notebook evaluates all 40 public LG M50T 21700 cells (511 checkup records). The first
50% of each trajectory is used for target calibration and the later 50% (246 points) is held
out for evaluation. It compares a source-pretrained/head-adapted LSTM, the same LSTM trained
from scratch on target calibration data, and four SOH-space persistence/fade controls. Results
include five seeds, per-cell and per-experiment errors, a paired 40-cell Wilcoxon test, and a
10,000-resample cell bootstrap confidence interval.
"""
        ),
        code_cell(
            """import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this notebook.")
print("GPU:", torch.cuda.get_device_name(0))
"""
        ),
        code_cell(packed_assets(assets)),
        code_cell(
            """import subprocess, sys, time
start = time.time()
subprocess.run(
    [sys.executable, "/kaggle/working/softcomputing_suite/run_kirkaldy40_transfer_audit.py"],
    check=True,
)
print("Total notebook experiment hours:", (time.time() - start) / 3600)
"""
        ),
        code_cell(
            """from pathlib import Path
import shutil
output = Path("/kaggle/working/softcomputing_suite/outputs_kirkaldy40_transfer")
archive = shutil.make_archive("/kaggle/working/softcomputing_kirkaldy40_transfer_results", "zip", output)
print("Created:", archive)
for path in sorted(output.iterdir()):
    print(path.name, path.stat().st_size)
"""
        ),
    ]
    path = NOTEBOOKS / "softcomputing_kirkaldy40_transfer_audit.ipynb"
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(path)


def main() -> None:
    build_search_notebook()
    build_mamba_notebook()
    build_transfer_notebook()


if __name__ == "__main__":
    main()
