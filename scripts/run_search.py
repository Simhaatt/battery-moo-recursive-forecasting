"""Launch one exact archived optimizer run block after explicit data staging."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["nsga2", "nsga3", "random"])
    p.add_argument("--archive", required=True, type=Path, help="Phase-1 search archive ZIP expected by the final runner")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--noise-floor", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if not a.archive.is_file(): raise SystemExit(f"Archive not found: {a.archive}")
    if a.output.exists() and any(a.output.iterdir()): raise SystemExit("Refusing to overwrite a nonempty output directory")
    runner = REPO / "notebooks/archive/10_asc_optimizer_study/run_repeated_optimizer.py"
    cmd = [sys.executable, str(runner), "--method", a.method, "--archive", str(a.archive), "--output", str(a.output)]
    if a.noise_floor: cmd.append("--noise-floor")
    if a.smoke: cmd.append("--smoke")
    raise SystemExit(subprocess.call(cmd, cwd=runner.parent))

if __name__ == "__main__": main()

