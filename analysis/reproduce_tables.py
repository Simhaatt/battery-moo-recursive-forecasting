from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from battery_moo.reproduce import make_tables
if __name__=="__main__": make_tables(ROOT)

