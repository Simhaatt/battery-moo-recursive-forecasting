"""Prepare the 40-cell Kirkaldy external-validation table from summary CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "kirkaldy_40_raw"
OUTPUT = ROOT / "kirkaldy_40_normalized_features.csv"
SUMMARY = ROOT / "kirkaldy_40_dataset_summary.json"
EPS = 1e-9

EXP_SOC_MAP = {
    1: (0.00, 0.30),
    2: (0.70, 0.85),
    3: (0.85, 1.00),
    4: (0.00, 1.00),
    5: (0.00, 1.00),
}


def normalize_name(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def find_col(columns, keywords, exclude=()):
    for keyword in keywords:
        for column in columns:
            if keyword in column and not any(token in column for token in exclude):
                return column
    return None


def parse_local_filename(path: Path) -> tuple[int, str, int]:
    match = re.fullmatch(r"expt(\d+)(?:_2)?_cell([A-Z]+)_(\d+)degC\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected file name: {path.name}")
    return int(match.group(1)), match.group(2), int(match.group(3))


def main() -> None:
    rows = []
    parse_report = []
    for path in sorted(RAW.glob("expt*_cell*.csv")):
        experiment, cell_id, nominal_temperature = parse_local_filename(path)
        frame = pd.read_csv(path)
        frame.columns = [normalize_name(column) for column in frame.columns]
        rpt_col = find_col(frame.columns, ["ageing_sets", "ageing_set", "rpt"])
        cycles_col = find_col(frame.columns, ["ageing_cycles", "ageing_cycle", "cycles"])
        capacity_col = find_col(frame.columns, ["c_10", "c10"], exclude=["c_2", "c2"])
        if capacity_col is None:
            capacity_col = find_col(frame.columns, ["c_2", "c2"])
        resistance_col = find_col(frame.columns, ["0_1s_resist", "0_1s_res", "resistance"])
        temperature_col = find_col(
            frame.columns,
            ["age_set_av_temperature", "temperature", "temp"],
            exclude=["min", "max"],
        )
        if cycles_col is None or capacity_col is None:
            raise RuntimeError(f"Missing required fields in {path.name}")

        soc_low, soc_high = EXP_SOC_MAP[experiment]
        cell_key = f"exp{experiment}_cell{cell_id}"
        accepted = 0
        for row_index, source_row in frame.iterrows():
            capacity_mah = pd.to_numeric(source_row.get(capacity_col), errors="coerce")
            cycles = pd.to_numeric(source_row.get(cycles_col), errors="coerce")
            if not np.isfinite(capacity_mah) or not np.isfinite(cycles):
                continue
            rpt = pd.to_numeric(source_row.get(rpt_col), errors="coerce") if rpt_col else row_index
            temperature = (
                pd.to_numeric(source_row.get(temperature_col), errors="coerce")
                if temperature_col
                else nominal_temperature
            )
            if not np.isfinite(temperature):
                temperature = nominal_temperature
            resistance = (
                pd.to_numeric(source_row.get(resistance_col), errors="coerce")
                if resistance_col
                else np.nan
            )
            rows.append(
                {
                    "cell_key": cell_key,
                    "exp": experiment,
                    "cell_id": cell_id,
                    "rpt_idx": float(rpt) if np.isfinite(rpt) else float(row_index),
                    "ageing_cycles": float(cycles),
                    "temperature": float(temperature),
                    "c_rate_chg": 0.3,
                    "c_rate_dischg": 1.0,
                    "soc_window": float(soc_high - soc_low),
                    "age_type": 3.0 if experiment == 4 else 2.0,
                    "Q": float(capacity_mah) / 1000.0,
                    "Re": float(resistance) * 1000.0 if np.isfinite(resistance) else np.nan,
                    "source_file": path.name,
                }
            )
            accepted += 1
        parse_report.append({"file": path.name, "cell_key": cell_key, "accepted_rows": accepted})

    output = pd.DataFrame(rows)
    output = (
        output.sort_values(["cell_key", "rpt_idx", "ageing_cycles"])
        .drop_duplicates(["cell_key", "rpt_idx"], keep="last")
        .query("Q > 0.1")
        .copy()
    )
    output["Re"] = output.groupby("cell_key")["Re"].transform(
        lambda series: series.interpolate(limit_direction="both").fillna(series.median())
    )
    output["Re"] = output["Re"].fillna(output["Re"].median())
    output["k_exp_raw"] = output.groupby("cell_key").cumcount()
    output["k_exp"] = output.groupby("cell_key")["k_exp_raw"].transform(
        lambda series: series / max(float(series.max()), 1.0)
    )
    output["Q0"] = output.groupby("cell_key")["Q"].transform("first")
    output["Re0"] = output.groupby("cell_key")["Re"].transform("first")
    output["SOH"] = output["Q"] / output["Q0"]
    output["Re_norm"] = output["Re"] / output["Re0"]
    temperature_kelvin = output["temperature"] + 273.15
    output["stress"] = (
        output["c_rate_chg"].abs()
        * (output["soc_window"].abs() + EPS)
        * np.exp((temperature_kelvin - 298.15) / 50.0)
    )
    output = output.sort_values(["cell_key", "k_exp"]).reset_index(drop=True)

    if output["cell_key"].nunique() != 40:
        raise RuntimeError(f"Expected 40 cells, found {output['cell_key'].nunique()}")
    if not np.isfinite(output[["Q", "Re", "SOH", "Re_norm"]].to_numpy()).all():
        raise RuntimeError("Prepared table contains non-finite targets")
    output.to_csv(OUTPUT, index=False)

    per_experiment = (
        output.groupby("exp")
        .agg(cells=("cell_key", "nunique"), records=("cell_key", "size"), min_temperature=("temperature", "min"), max_temperature=("temperature", "max"))
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "cells": int(output["cell_key"].nunique()),
        "records": int(len(output)),
        "external_points_after_bol": int(len(output) - output["cell_key"].nunique()),
        "experiments": per_experiment,
        "temperature_range_c": [float(output["temperature"].min()), float(output["temperature"].max())],
        "ageing_cycle_range": [float(output["ageing_cycles"].min()), float(output["ageing_cycles"].max())],
        "q_range_ah": [float(output["Q"].min()), float(output["Q"].max())],
        "re_range_milliohm": [float(output["Re"].min()), float(output["Re"].max())],
        "parse_report": parse_report,
        "output": OUTPUT.name,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "parse_report"}, indent=2))


if __name__ == "__main__":
    main()
