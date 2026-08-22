from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if __name__=="__main__":
 d=pd.read_csv(ROOT/"results/optimizer/hv_igd_by_run_and_evaluation.csv").query("evaluations==280")
 print(d.groupby("method")[["hypervolume","IGD"]].agg(["mean","std"]).to_string())

