"""Write SHA-256 checksums for immutable manuscript-contract inputs."""
from pathlib import Path
import csv,hashlib

ROOT=Path(__file__).resolve().parents[1]
INCLUDE=(ROOT/"configs",ROOT/"data/splits",ROOT/"manuscript",ROOT/"results")
EXCLUDE={ROOT/"results/reproduction_check.json",ROOT/"results/REPRODUCTION_CHECK.md"}

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

rows=[]
for directory in INCLUDE:
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if path in EXCLUDE or "generated" in path.parts: continue
        rows.append((path.relative_to(ROOT).as_posix(),path.stat().st_size,digest(path)))
with (ROOT/"ARTIFACT_MANIFEST.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.writer(f); writer.writerow(("path","bytes","sha256")); writer.writerows(rows)
print(f"Wrote {len(rows)} checksums")

