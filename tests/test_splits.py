from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_fixed_split_cell_counts():
    d=pd.read_csv(ROOT/"data/splits/fixed_cell_split.csv")
    assert len(d)==228
    assert d.cell_id.nunique()==228
    assert d.split.value_counts().to_dict()=={"train":209,"test":10,"validation":9}

def test_grouped_folds_are_cell_disjoint_assignments():
    d=pd.read_csv(ROOT/"data/splits/grouped_cv_folds.csv")
    assert len(d)==228 and d.cell_id.nunique()==228 and d.fold.nunique()==5

