from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
 print(pd.read_csv(ROOT/"results/transfer/transfer_summary.csv").to_string(index=False))
 print(pd.read_csv(ROOT/"results/transfer/transfer_cell_level_tests.csv").to_string(index=False))

