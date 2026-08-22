from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from battery_moo.reproduce import manuscript_check
if __name__=="__main__": print(json.dumps(manuscript_check(ROOT),indent=2))

