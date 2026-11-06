from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


def _validate(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must be one-dimensional arrays with at least two matching points.")
    if len(np.unique(x)) != len(x): raise ValueError("Interpolation nodes must be distinct.")
    return x, y


def lagrange(x, y, points):
    x, y = _validate(x, y); points = np.asarray(points, dtype=float)
    values = np.zeros_like(points, dtype=float)
    for i in range(len(x)):
        basis = np.ones_like(points, dtype=float)
        for j in range(len(x)):
            if i != j: basis *= (points - x[j]) / (x[i] - x[j])
        values += y[i] * basis
    return values


def newton_divided_difference(x, y, points):
    x, y = _validate(x, y); coeff = y.copy()
    for j in range(1, len(x)):
        coeff[j:] = (coeff[j:] - coeff[j - 1:-1]) / (x[j:] - x[:len(x) - j])
    points = np.asarray(points, dtype=float)
    values = np.full_like(points, coeff[-1], dtype=float)
    for j in range(len(coeff) - 2, -1, -1):
        values = coeff[j] + (points - x[j]) * values
    return values, coeff


def cubic_spline(x, y, points):
    x, y = _validate(x, y)
    return CubicSpline(x, y)(points)


def polynomial_fit(x, y, degree, points=None):
    x, y = _validate(x, y)
    if degree < 0 or degree >= len(x): raise ValueError("Degree must be non-negative and less than number of points.")
    coeff = np.polyfit(x, y, degree)
    return coeff, np.polyval(coeff, x if points is None else points)

