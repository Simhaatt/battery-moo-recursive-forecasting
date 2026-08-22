"""Build three self-contained Kaggle notebooks for the repeated ASC optimizer study."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REVISION = ROOT.parent
ROUND2 = REVISION / "07_round2_revision"
EXTENSION = REVISION / "08_softcomputing_extension"
OUT = ROOT / "kaggle_notebooks"
OUT.mkdir(parents=True, exist_ok=True)


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code": result.update({"execution_count": None, "outputs": []})
    return result


def packed_assets(extra):
    assets = {
        "round2_core/run_round2_experiments.py": ROUND2 / "run_round2_experiments.py",
        "round2_core/run_metaheuristic_search.py": ROUND2 / "run_metaheuristic_search.py",
        "round2_core/inputs/phase1_cv_all_rows.csv": ROUND2 / "inputs/phase1_cv_all_rows.csv",
        "round2_core/inputs/fixed_split_val_predictions.csv": ROUND2 / "inputs/fixed_split_val_predictions.csv",
        "round2_core/inputs/fixed_split_test_predictions.csv": ROUND2 / "inputs/fixed_split_test_predictions.csv",
        "run_softcomputing_search.py": EXTENSION / "run_softcomputing_search.py",
        "run_repeated_optimizer.py": ROOT / "run_repeated_optimizer.py",
    } | extra
    packed = {name: base64.b64encode(gzip.compress(path.read_bytes(), 9)).decode() for name, path in assets.items()}
    payload = json.dumps(packed, separators=(",", ":"))
    return f'''from pathlib import Path
import base64, gzip, json
ROOT = Path("/kaggle/working/asc_optimizer_suite")
ROOT.mkdir(parents=True, exist_ok=True)
packed = json.loads({payload!r})
for name, encoded in packed.items():
    destination = ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(gzip.decompress(base64.b64decode(encoded)))
print(f"Materialized {{len(packed)}} files ({{sum((ROOT/n).stat().st_size for n in packed):,}} bytes)")
'''


GPU = '''import platform, torch
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Kaggle Settings > Accelerator must be GPU T4 x2 (or another GPU).")
print("GPU:", torch.cuda.get_device_name(0))
SMOKE = False  # Keep False for publishable results; True only tests the pipeline.
'''


def make_notebook(cells):
    return {"cells": cells, "metadata": {"accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}


def write(name, cells):
    path = OUT / name
    path.write_text(json.dumps(make_notebook(cells), indent=1), encoding="utf-8")
    print(path, path.stat().st_size)


def first():
    write("ASC_optimizer_01_noise_NS​​GA2.ipynb".replace("​​", ""), [
        cell("markdown", """# ASC optimizer study 1/3 — noise floor and NSGA-II

This self-contained notebook first repeats eight fixed LSTM configurations over ten matched
seeds to quantify training and latency noise. It then runs **15 independent NSGA-II searches**.
Each search uses 28 candidates and ten generations including initialization: **280 unique trained
configurations per run; 4,200 total**. The three minimized objectives are validation rollout macro
MAPE, trainable parameters, and measured batch-one latency. `L + H <= 29` is repaired to the
closest allowed feasible horizon. Test trajectories are never used in optimizer search.

Runtime on a Kaggle T4 is approximately 6.5–7.5 hours. A rolling ZIP is rewritten after the noise
study and every completed repetition, so an interrupted session retains all completed runs if you
save a notebook version. The final file is `/kaggle/working/asc_nsga2_noise_results.zip`.
"""), cell("code", GPU), cell("code", packed_assets({})),
        cell("code", '''import subprocess, sys, time
command = [sys.executable, "/kaggle/working/asc_optimizer_suite/run_repeated_optimizer.py",
           "--method", "nsga2", "--noise-floor",
           "--output", "/kaggle/working/asc_optimizer_suite/outputs_nsga2_noise",
           "--archive", "/kaggle/working/asc_nsga2_noise_results.zip"]
if SMOKE: command.append("--smoke")
start = time.time(); subprocess.run(command, check=True)
print("Elapsed hours:", (time.time()-start)/3600)
'''), cell("code", '''from pathlib import Path
archive = Path("/kaggle/working/asc_nsga2_noise_results.zip")
print("DOWNLOAD THIS FILE:", archive, f"({archive.stat().st_size/1e6:.1f} MB)")
print(Path("/kaggle/working/asc_optimizer_suite/outputs_nsga2_noise/status.json").read_text())
''')])


def second():
    write("ASC_optimizer_02_NSGA3.ipynb", [
        cell("markdown", """# ASC optimizer study 2/3 — NSGA-III

This self-contained notebook runs **15 independent NSGA-III searches** under precisely the same
data split, search space, run-wise training seeds, three objectives, and 280-evaluation budget used
for NSGA-II. NSGA-III uses the 28 Das–Dennis reference directions for three objectives with H=6.

Runtime on a Kaggle T4 is approximately 6–7 hours. A rolling ZIP checkpoint is rewritten after
every completed repetition. Download `/kaggle/working/asc_nsga3_results.zip` when complete.
"""), cell("code", GPU), cell("code", packed_assets({})),
        cell("code", '''import subprocess, sys, time
command = [sys.executable, "/kaggle/working/asc_optimizer_suite/run_repeated_optimizer.py",
           "--method", "nsga3", "--output", "/kaggle/working/asc_optimizer_suite/outputs_nsga3",
           "--archive", "/kaggle/working/asc_nsga3_results.zip"]
if SMOKE: command.append("--smoke")
start = time.time(); subprocess.run(command, check=True)
print("Elapsed hours:", (time.time()-start)/3600)
'''), cell("code", '''from pathlib import Path
archive = Path("/kaggle/working/asc_nsga3_results.zip")
print("DOWNLOAD THIS FILE:", archive, f"({archive.stat().st_size/1e6:.1f} MB)")
print(Path("/kaggle/working/asc_optimizer_suite/outputs_nsga3/status.json").read_text())
''')])


def third():
    write("ASC_optimizer_03_random_analysis_finals.ipynb", [
        cell("markdown", """# ASC optimizer study 3/3 — random search, analysis, and final confirmation

Before running, add the two ZIPs produced by notebooks 1 and 2 as Kaggle inputs (upload them as a
private dataset or attach them directly): `asc_nsga2_noise_results.zip` and `asc_nsga3_results.zip`.
This notebook runs 15 matched-budget random searches, pools all 12,600 optimizer evaluations under
one global normalization, computes hypervolume and IGD curves, performs Kruskal–Wallis and
Holm-corrected pairwise tests with A12 effect sizes, re-screens five candidates per method, and
finally evaluates the manual and three selected configurations over ten matched seeds.

Runtime on a Kaggle T4 is approximately 7–8 hours. The complete publishable evidence archive is
`/kaggle/working/asc_optimizer_complete_results.zip`.
"""), cell("code", GPU), cell("code", packed_assets({"analyze_optimizer_study.py": ROOT / "analyze_optimizer_study.py"})),
        cell("code", '''from pathlib import Path
import zipfile, pandas as pd, shutil

def import_result(filename, expected_method, target):
    target.mkdir(parents=True, exist_ok=True)
    matches = list(Path("/kaggle/input").rglob(filename))
    if matches:
        print("Found archive", matches[0])
        with zipfile.ZipFile(matches[0]) as z: z.extractall(target)
        return
    # Kaggle sometimes automatically unpacks ZIP files uploaded as Datasets.
    for csv in Path("/kaggle/input").rglob("all_trials.csv"):
        try:
            frame = pd.read_csv(csv, nrows=1)
            column = "method" if "method" in frame else "algorithm"
            if len(frame) and str(frame[column].iloc[0]) == expected_method:
                print("Found Kaggle-unpacked result directory", csv.parent)
                shutil.copytree(csv.parent, target, dirs_exist_ok=True)
                return
        except Exception:
            pass
    raise FileNotFoundError(f"Attach {filename} (or its Kaggle-unpacked dataset) and rerun.")

imports = Path("/kaggle/working/asc_optimizer_suite/imported")
import_result("asc_nsga2_noise_results.zip", "NSGA-II", imports/"nsga2")
import_result("asc_nsga3_results.zip", "NSGA-III", imports/"nsga3")
print("Imported prerequisite results.")
'''), cell("code", '''import subprocess, sys, time
root = "/kaggle/working/asc_optimizer_suite"
random_command = [sys.executable, f"{root}/run_repeated_optimizer.py", "--method", "random",
                  "--output", f"{root}/outputs_random", "--archive", "/kaggle/working/asc_random_results.zip"]
if SMOKE: random_command.append("--smoke")
start = time.time(); subprocess.run(random_command, check=True)
analysis_command = [sys.executable, f"{root}/analyze_optimizer_study.py",
                    "--nsga2", f"{root}/imported/nsga2", "--nsga3", f"{root}/imported/nsga3",
                    "--random", f"{root}/outputs_random", "--output", f"{root}/complete_analysis",
                    "--archive", "/kaggle/working/asc_optimizer_complete_results.zip"]
if SMOKE: analysis_command.append("--smoke")
subprocess.run(analysis_command, check=True)
print("Total elapsed hours:", (time.time()-start)/3600)
'''), cell("code", '''from pathlib import Path
import pandas as pd
archive = Path("/kaggle/working/asc_optimizer_complete_results.zip")
print("DOWNLOAD THIS FILE:", archive, f"({archive.stat().st_size/1e6:.1f} MB)")
output = Path("/kaggle/working/asc_optimizer_suite/complete_analysis")
display(pd.read_csv(output/"hv_igd_convergence_summary.csv").groupby("method").tail(1))
display(pd.read_csv(output/"final_ten_seed_summary.csv"))
display(pd.read_csv(output/"optimizer_statistical_tests.csv"))
''')])


if __name__ == "__main__": first(); second(); third()
