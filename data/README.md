# Data access and placement

Raw third-party battery datasets are intentionally excluded from this repository. Obtain them from their original providers under their original licenses, then place local copies under `data/raw/` (ignored by Git). Do not commit or redistribute them without permission.

The public repository contains only cell-level split assignments and derived experimental outputs. `splits/fixed_cell_split.csv` encodes the immutable 209/9/10-cell train/validation/test partition. `splits/grouped_cv_folds.csv` encodes the five cell-grouped folds. The fixed counts are validated automatically.

Historical source tables contain `Rct`/`Rct0`; public code and writing map this field to the pulse-resistance proxy `Rpulse`/`Rpulse0`. This renaming is semantic documentation only and does not alter stored numeric values.

For a Level-2 rerun, stage the exact processed CSV inputs expected by the archived runner under `notebooks/archive/07_round2_revision/inputs/`. Compare hashes and schemas with `SOURCE_INVENTORY.csv` and the archived manifests before launching expensive computation.

