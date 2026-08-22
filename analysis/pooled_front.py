from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
 d=pd.read_csv(ROOT/"results/audit/pooled_pareto_tidy.csv")
 print(d.groupby("method").size().rename("pooled_points").to_string()); print("Total:",len(d))

