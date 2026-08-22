from pathlib import Path
from battery_moo.reproduce import manuscript_check

ROOT=Path(__file__).resolve().parents[1]

def test_all_expected_manuscript_values_match():
    failures=[r for r in manuscript_check(ROOT) if r["status"]!="PASS"]
    assert not failures, failures

