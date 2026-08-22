from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_final_and_superseded_transfer_are_explicitly_distinct():
    final=pd.read_csv(ROOT/"results/transfer/transfer_cell_level_tests.csv").iloc[0]
    old=pd.read_csv(ROOT/"results/audit/superseded_transfer_cell_level_tests.csv").iloc[0]
    col="mean_cell_MAPE_difference_a_minus_b_pp"
    assert abs(final[col]-6.987501122438184)<1e-12
    assert old[col] < 0

