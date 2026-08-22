from pathlib import Path
import pandas as pd
from battery_moo.integrity import validate_search_trials
from battery_moo.model import lstm_parameter_count

ROOT=Path(__file__).resolve().parents[1]

def test_complete_search_integrity():
    checks=validate_search_trials(pd.read_csv(ROOT/"results/optimizer/all_optimizer_trials.csv"))
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}

def test_final_evaluation_has_four_configs_and_ten_seeds():
    d=pd.read_csv(ROOT/"results/final/final_ten_seed_raw.csv")
    assert set(d.configuration)=={"Manual","NSGA-II","NSGA-III","Random"}
    assert d.groupby("configuration").size().eq(10).all()

def test_parameter_count_is_deterministic():
    assert lstm_parameter_count(11,64,1)==36418
    assert lstm_parameter_count(11,192,2)==486978

def test_expected_optimizer_and_training_seeds():
    d=pd.read_csv(ROOT/"results/optimizer/all_optimizer_trials.csv")
    assert set(d.run)==set(range(1,16))
    assert set(d.training_seed)==set(range(41001,41016))
