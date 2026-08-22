"""Launch an archived stress, TCN, transfer-control, or statistics experiment."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("task", choices=["stress-ablation","transfer-controls","tcn","statistics","all"])
    a=p.parse_args()
    runner=REPO/"notebooks/archive/07_round2_revision/run_round2_experiments.py"
    inputs=runner.parent/"inputs"
    if not inputs.exists(): raise SystemExit(f"Stage licensed processed input CSVs first: {inputs}")
    raise SystemExit(subprocess.call([sys.executable,str(runner),a.task],cwd=runner.parent))

if __name__ == "__main__": main()

