"""Metrics used by the archived experiment analysis."""
from __future__ import annotations

import numpy as np


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined for zero-valued targets")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def macro_mape(q_true, q_pred, re_true, re_pred) -> float:
    return (mape(q_true, q_pred) + mape(re_true, re_pred)) / 2.0


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    """Return minimization nondominance mask; equal points do not dominate."""
    values = np.asarray(values, dtype=float)
    return np.array([
        not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
        for row in values
    ])

