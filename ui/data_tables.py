from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def points_editor(key="points", default=None):
    default = default or pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.0, 5.0, 10.0]})
    value = st.data_editor(
        default, num_rows="dynamic", width="stretch", key=key,
        column_config={"x": st.column_config.NumberColumn("x", required=True),
                       "y": st.column_config.NumberColumn("y", required=True)},
    )
    if value.isna().any().any():
        raise ValueError("Complete every x and y cell before calculating.")
    if len(value) < 2:
        raise ValueError("Enter at least two data points.")
    return value[["x", "y"]].to_numpy(dtype=float).T


def matrix_editor(key="matrix", default=None):
    default = default or pd.DataFrame([[4.0, 1.0], [2.0, 3.0]], columns=["c1", "c2"])
    value = st.data_editor(default, num_rows="dynamic", width="stretch", key=key)
    if value.empty or value.isna().any().any():
        raise ValueError("Complete every matrix cell before calculating.")
    try:
        return value.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Matrix cells must be numeric.") from exc


def vector_editor(key="vector", default=None):
    default = default or pd.DataFrame({"b": [1.0, 2.0]})
    value = st.data_editor(default, num_rows="dynamic", width="stretch", key=key)
    if value.empty or value.isna().any().any():
        raise ValueError("Complete every vector cell before calculating.")
    return value.iloc[:, 0].to_numpy(dtype=float)
