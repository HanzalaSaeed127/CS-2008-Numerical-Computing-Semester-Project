"""Consistent display formatting for numerical results."""
from __future__ import annotations

import numpy as np
import pandas as pd


def format_number(value, precision=8, scientific=False):
    if isinstance(value, (float, np.floating, int, np.integer)):
        value = float(value)
        if not np.isfinite(value):
            return str(value)
        if scientific:
            return f"{value:.{precision}e}"
        if value != 0 and (abs(value) >= 10**precision or abs(value) < 10**-4):
            return f"{value:.{precision}e}"
        return f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return str(value)


def format_dataframe(frame, precision=8, scientific=False):
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].map(lambda value: format_number(value, precision, scientific))
    return result


def matrix_latex(matrix, precision=8):
    rows = [
        " & ".join(format_number(value, precision) for value in row)
        for row in np.asarray(matrix)
    ]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"
