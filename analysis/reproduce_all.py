"""Run Level-1, artifact-only reproduction without retraining."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from battery_moo.reproduce import run

if __name__ == "__main__":
    raise SystemExit(run(REPO))

