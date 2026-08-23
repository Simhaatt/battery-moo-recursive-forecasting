"""Reproduce the corrected phase-two preprocessing and fixed cell split.

This standalone extraction preserves the preprocessing implemented and executed in
``pinn-battery (27).ipynb``. On the dataset used by that notebook, it produced
3,944 observations from 228 cells, split into 180 training cells (3,157 rows),
27 validation cells (392 rows), and 21 test cells (395 rows).

An earlier archived combined table contains 3,980 rows. Therefore, this script is
the preserved corrected preprocessing revision; it must not be represented as the
exact producer of that earlier 3,980-row archive.

Example:
    python scripts/preprocess_phase2_split.py --raw-base DATASET --output-dir OUTPUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
PROTOCOL_CYC_CONDITION = (2,)
PROTOCOL_CYC_CHARGED = (0,)
ALLOW_PROTOCOL_FALLBACK = True
MIN_ROWS_AFTER_FILTER = 2_000


def choose_column(
    columns: list[str],
    direct_candidates: list[str],
    required_tokens: list[str] | None = None,
    forbidden_tokens: list[str] | None = None,
) -> str | None:
    """Select a source column using the notebook's ordered matching rules."""
    column_set = set(columns)
    for candidate in direct_candidates:
        if candidate in column_set:
            return candidate
    required = [token.lower() for token in (required_tokens or [])]
    forbidden = [token.lower() for token in (forbidden_tokens or [])]
    for column in columns:
        lowered = column.lower()
        if required and not all(token in lowered for token in required):
            continue
        if any(token in lowered for token in forbidden):
            continue
        return column
    return None


def infer_eis_resistance_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Reproduce the EIS resistance-column resolution used by the notebook."""
    lowered = [column.lower() for column in columns]
    preferred_re = [
        "z_ref_now_mohm", "eis_re_ohm", "re_ohm", "r_e_ohm",
        "electrolyte_resistance_ohm", "rs_ohm", "r0_ohm", "r_s_ohm",
    ]
    preferred_rct = [
        "eis_rct_ohm", "rct_ohm", "r_ct_ohm",
        "charge_transfer_resistance_ohm", "r1_ohm", "r_ct",
    ]
    re_column = next(
        (columns[lowered.index(name)] for name in preferred_re if name in lowered), None
    )
    rct_column = next(
        (columns[lowered.index(name)] for name in preferred_rct if name in lowered), None
    )
    resistance_like = [
        columns[index]
        for index, name in enumerate(lowered)
        if "ohm" in name or name.startswith("r_") or name.startswith("r")
    ]
    if re_column is None:
        re_column = next(
            (column for column in resistance_like
             if not any(token in column.lower() for token in ("init", "rct", "ct"))),
            None,
        )
    if rct_column is None:
        rct_column = next(
            (column for column in resistance_like
             if "init" not in column.lower()
             and "re_ohm" not in column.lower()
             and any(token in column.lower() for token in ("rct", "ct", "r1"))),
            None,
        )
    # Preserve the notebook's last-resort behavior exactly. In the recorded run
    # this selected z_ref_now_mOhm as Re and z_ref_init_mOhm as the second target.
    if (re_column is None or rct_column is None) and len(resistance_like) >= 2:
        if re_column is None:
            re_column = resistance_like[0]
        if rct_column is None:
            rct_column = next(
                (column for column in resistance_like if column != re_column), None
            )
    return re_column, rct_column


def find_eis_files(raw_base: Path) -> list[Path]:
    """Locate CSV files in EIS-, impedance-, or resistance-named directories."""
    candidate_directories = {
        path for path in raw_base.rglob("*")
        if path.is_dir()
        and any(token in path.name.lower() for token in ("eis", "imp", "res"))
    }
    return sorted({
        csv_path for directory in candidate_directories
        for csv_path in directory.glob("*.csv")
    })


def match_eis_file(cell_id: str, eis_files: list[Path]) -> Path | None:
    exact = [
        path for path in eis_files
        if f"_{cell_id}.csv" in path.name or f"{cell_id}.csv" in path.name
    ]
    if exact:
        return exact[0]
    return next((path for path in eis_files if cell_id in path.name), None)


def load_and_preprocess(raw_base: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Construct the corrected phase-two table from raw EOC and EIS CSV files."""
    eoc_files = sorted((raw_base / "cell_eocv2").glob("cell_eocv2_*.csv"))
    if not eoc_files:
        raise FileNotFoundError(f"No cell_eocv2 CSV files found under {raw_base}")
    sample_columns = pd.read_csv(eoc_files[0], sep=";").columns.tolist()
    q_column = choose_column(sample_columns, ["cap_aged_est_Ah"])
    if q_column is None:
        raise ValueError("Cannot detect the capacity target cap_aged_est_Ah")

    eis_files = find_eis_files(raw_base)
    if not eis_files:
        raise FileNotFoundError(f"No EIS CSV files found under {raw_base}")
    eis_sample_columns = pd.read_csv(eis_files[0], sep=";").columns.tolist()
    eis_re_column, eis_rct_column = infer_eis_resistance_columns(eis_sample_columns)
    if eis_re_column is None or eis_rct_column is None:
        raise ValueError("Cannot resolve both EIS resistance columns")

    column_map = {
        q_column: "Q", "age_temp": "temperature",
        "age_chg_rate": "c_rate_chg", "age_dischg_rate": "c_rate_dischg",
        "age_soc": "soc_window", "num_cycles_op": "cycle",
    }
    base_columns = [
        "cell_id", "timestamp_s", "checkup_idx", "age_type", "soh_cap",
        "cyc_condition", "cyc_charged", "t_start_degC",
    ]
    rows: list[pd.DataFrame] = []
    for eoc_path in eoc_files:
        cell_id = eoc_path.stem.replace("cell_eocv2_", "")
        cell = pd.read_csv(eoc_path, sep=";")
        cell["cell_id"] = cell_id
        if "checkup_idx" not in cell.columns:
            cell = cell.sort_values("timestamp_s").reset_index(drop=True)
            cell["checkup_idx"] = np.arange(len(cell), dtype=int)
        kept = list(dict.fromkeys(
            [column for column in base_columns if column in cell.columns]
            + [column for column in column_map if column in cell.columns]
        ))
        cell = cell[kept].copy().rename(columns=column_map)
        cell = cell.loc[:, ~cell.columns.duplicated()].copy()
        cell["Re"] = np.nan
        cell["Rct"] = np.nan

        eis_path = match_eis_file(cell_id, eis_files)
        if eis_path is not None:
            eis = pd.read_csv(eis_path, sep=";")
            if "checkup_idx" not in eis.columns:
                if "timestamp_s" in eis.columns:
                    eis = eis.sort_values("timestamp_s").reset_index(drop=True)
                eis["checkup_idx"] = np.arange(len(eis), dtype=int)
            eis = eis[["checkup_idx", eis_re_column, eis_rct_column]].rename(
                columns={eis_re_column: "Re_eis", eis_rct_column: "Rct_eis"}
            )
            cell = pd.merge_asof(
                cell.sort_values("checkup_idx"), eis.sort_values("checkup_idx"),
                on="checkup_idx", direction="nearest",
            )
            cell["Re"] = cell["Re"].fillna(cell["Re_eis"])
            cell["Rct"] = cell["Rct"].fillna(cell["Rct_eis"])
            cell = cell.drop(columns=["Re_eis", "Rct_eis"])
        rows.append(cell)

    unfiltered = pd.concat(rows, ignore_index=True)
    if float(unfiltered["Re"].dropna().std()) < 1e-6:
        raise ValueError("Re has near-zero variance; check EIS column resolution")
    data = unfiltered.dropna(subset=["Q", "Re", "Rct"]).copy()
    if "cyc_condition" in data.columns:
        data = data[data["cyc_condition"].isin(PROTOCOL_CYC_CONDITION)]
    if "cyc_charged" in data.columns:
        data = data[data["cyc_charged"].isin(PROTOCOL_CYC_CHARGED)]

    fallback_used = False
    if len(data) < MIN_ROWS_AFTER_FILTER and ALLOW_PROTOCOL_FALLBACK:
        fallback_used = True
        data = unfiltered.dropna(subset=["Q", "Re", "Rct"]).copy()
        if "cyc_condition" in data.columns:
            data = data[data["cyc_condition"].isin([1, 2])]
        if "cyc_charged" in data.columns:
            data = data[data["cyc_charged"].isin([0, 1])]

    data = data[(data["Q"] > 0) & (data["Re"] > 0) & (data["Rct"] > 0)].copy()
    if "soh_cap" in data.columns:
        data = data[data["soh_cap"].between(0, 105)].copy()
    data = data.sort_values(["cell_id", "checkup_idx"]).reset_index(drop=True)
    data["Q"] = data.groupby("cell_id")["Q"].transform(lambda values: values.cummin())
    data["Re"] = data.groupby("cell_id")["Re"].transform(lambda values: values.cummax())
    data["Rct"] = data.groupby("cell_id")["Rct"].transform(lambda values: values.cummax())
    data["Q0"] = data.groupby("cell_id")["Q"].transform("first")
    data["Re0"] = data.groupby("cell_id")["Re"].transform("first")
    data["Rct0"] = data.groupby("cell_id")["Rct"].transform("first")
    if "cycle" in data.columns and data["cycle"].notna().mean() > 0.9:
        data["cycle"] = data["cycle"].astype(float)
        data["k_exp"] = data["cycle"] - data.groupby("cell_id")["cycle"].transform("first")
    else:
        data["k_exp"] = data["checkup_idx"].astype(float)
    data["k_exp"] = data["k_exp"].clip(lower=0)
    if "temperature" not in data.columns:
        for alternative in ("t_start_degC", "age_temp", "t_avg_degC"):
            if alternative in data.columns:
                data = data.rename(columns={alternative: "temperature"})
                break
    if "age_type" in data.columns and data["age_type"].dtype == object:
        data["age_type"] = pd.Categorical(data["age_type"]).codes.astype(np.float32)

    metadata = {
        "eoc_file_count": len(eoc_files), "eis_file_count": len(eis_files),
        "capacity_source": q_column, "re_source": eis_re_column,
        "rct_source": eis_rct_column, "protocol_fallback_used": fallback_used,
    }
    return data, metadata


def split_by_cell(
    data: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the notebook's seeded, cell-disjoint 180/27/21 split procedure."""
    cell_metadata = data.groupby("cell_id")[["temperature", "age_type"]].first().reset_index()
    cell_metadata["strata"] = (
        cell_metadata["temperature"].astype(str) + "_"
        + cell_metadata["age_type"].astype(str)
    )
    try:
        train_cells, temporary_cells = train_test_split(
            cell_metadata, test_size=0.21, stratify=cell_metadata["strata"],
            random_state=seed,
        )
        validation_cells, test_cells = train_test_split(
            temporary_cells, test_size=0.43, stratify=temporary_cells["strata"],
            random_state=seed,
        )
    except ValueError:
        train_cells, temporary_cells = train_test_split(
            cell_metadata, test_size=0.21, random_state=seed
        )
        validation_cells, test_cells = train_test_split(
            temporary_cells, test_size=0.43, random_state=seed
        )
    membership = pd.concat([
        train_cells.assign(split="train"),
        validation_cells.assign(split="validation"),
        test_cells.assign(split="test"),
    ], ignore_index=True)[
        ["cell_id", "split", "temperature", "age_type", "strata"]
    ].sort_values(["split", "cell_id"])
    training = data[data["cell_id"].isin(train_cells["cell_id"])].copy()
    validation = data[data["cell_id"].isin(validation_cells["cell_id"])].copy()
    test = data[data["cell_id"].isin(test_cells["cell_id"])].copy()
    return training, validation, test, membership


def write_outputs(raw_base: Path, output_dir: Path, seed: int = SEED) -> dict[str, object]:
    data, metadata = load_and_preprocess(raw_base)
    training, validation, test, membership = split_by_cell(data, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(output_dir / "phase2_train.csv", index=False)
    validation.to_csv(output_dir / "phase2_val.csv", index=False)
    test.to_csv(output_dir / "phase2_test.csv", index=False)
    membership.to_csv(output_dir / "phase2_cell_split.csv", index=False)
    manifest = {
        "provenance": "standalone extraction from pinn-battery (27).ipynb",
        "revision_note": (
            "Corrected 3,944-row revision; not asserted to be the exact producer "
            "of the earlier 3,980-row archive."
        ),
        "seed": seed,
        "rows": {
            "all": len(data), "train": len(training),
            "validation": len(validation), "test": len(test),
        },
        "cells": {
            "all": int(data["cell_id"].nunique()),
            "train": int(training["cell_id"].nunique()),
            "validation": int(validation["cell_id"].nunique()),
            "test": int(test["cell_id"].nunique()),
        },
        **metadata,
    }
    (output_dir / "phase2_preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-base", type=Path, required=True,
        help="Dataset directory containing cell_eocv2 and cell_eisv2",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for processed CSVs, split membership, and manifest",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    manifest = write_outputs(
        arguments.raw_base.resolve(), arguments.output_dir.resolve(), arguments.seed
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
