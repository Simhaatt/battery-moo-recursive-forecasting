"""Fetch only the 40 Kirkaldy Performance Summary CSVs from Zenodo.

The five source ZIP archives total roughly 51 GB. Zenodo supports HTTP range
requests, so ``remotezip`` can read each central directory and download only
the small CSV members needed for trajectory-level external validation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

from remotezip import RemoteZip


RECORD_ID = "10637534"
ARCHIVES = {
    "expt1": "Expt 1 - Si-based Degradation.zip",
    "expt2_2": "Expt 2,2 - C-based Degradation 2.zip",
    "expt3": "Expt 3 - Cathode Degradation and Li-Plating.zip",
    "expt4": "Expt 4 - Drive Cycle Aging (Control).zip",
    "expt5": "Expt 5 - Standard Cycle Aging (Control).zip",
}
EXPECTED_COUNTS = {"expt1": 9, "expt2_2": 6, "expt3": 9, "expt4": 8, "expt5": 8}

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "kirkaldy_40_raw"


def archive_url(filename: str) -> str:
    encoded = quote(filename, safe="")
    return f"https://zenodo.org/api/records/{RECORD_ID}/files/{encoded}/content"


def safe_member_name(member: str) -> str:
    name = Path(member).name
    match = re.search(r"Expt ([^ ]+) - cell ([A-Z]+) \((\d+)degC\)", name)
    if not match:
        raise ValueError(f"Unexpected summary filename: {name}")
    expt, cell, temperature = match.groups()
    expt = expt.replace(",", "_")
    return f"expt{expt}_cell{cell}_{temperature}degC.csv"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for archive_key, filename in ARCHIVES.items():
        url = archive_url(filename)
        print(f"Opening {filename}")
        with RemoteZip(url) as remote_zip:
            members = sorted(
                (
                    info
                    for info in remote_zip.infolist()
                    if "Summary Data/Performance Summary/" in info.filename
                    and info.filename.endswith("Processed Data.csv")
                ),
                key=lambda info: info.filename,
            )
            expected = EXPECTED_COUNTS[archive_key]
            if len(members) != expected:
                raise RuntimeError(
                    f"{filename}: expected {expected} Performance Summary CSVs, found {len(members)}"
                )
            for info in members:
                payload = remote_zip.read(info.filename)
                output_path = OUTPUT / safe_member_name(info.filename)
                output_path.write_bytes(payload)
                manifest.append(
                    {
                        "record_id": RECORD_ID,
                        "archive": filename,
                        "archive_url": url,
                        "member": info.filename,
                        "compressed_bytes": info.compress_size,
                        "uncompressed_bytes": info.file_size,
                        "output_file": output_path.name,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                print(f"  {output_path.name}: {len(payload):,} bytes")
                time.sleep(0.15)

    if len(manifest) != 40:
        raise RuntimeError(f"Expected 40 cell summaries, downloaded {len(manifest)}")
    manifest_path = OUTPUT / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} summaries and {manifest_path}")


if __name__ == "__main__":
    main()
