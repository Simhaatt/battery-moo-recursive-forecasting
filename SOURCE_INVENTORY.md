# Source inventory summary

`SOURCE_INVENTORY.csv` records 587 files from the supplied scientific revision package, excluding unrelated `node_modules` content. Every row contains the original relative path, byte size, SHA-256 digest, and a role classification.

The public repository retains the authored/executed notebooks and Python exports under `notebooks/archive/`, while large model checkpoints, duplicated Kaggle ZIPs, raw licensed data, and submission-layout copies are represented by inventory hashes rather than duplicated again. Authoritative derived CSV/JSON results required for Level-1 reproduction are included under `results/`.

Authority order used during conversion:

1. supplied final LaTeX manuscript;
2. final ASC evidence and complete optimizer audit packs;
3. round-two matched experiment outputs;
4. historical executed notebooks and ZIPs;
5. exploratory notebooks.

This ordering matters for the conflicting transfer artifacts documented in `AUDIT_REPORT.md`.

