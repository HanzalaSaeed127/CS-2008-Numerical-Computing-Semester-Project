from __future__ import annotations
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import newton_cotes as scipy_newton_cotes


def _grid(a, b, n):
    if b <= a or n < 1: raise ValueError("Require b > a and a positive number of subintervals.")
    return np.linspace(a, b, n + 1)


def midpoint(fn, a, b, n=1):
    edges = _grid(a, b, n); mids = (edges[:-1] + edges[1:]) / 2
    return float(np.sum(fn(mids) * np.diff(edges))), mids


def trapezoidal(fn, a, b, n=1):
    x = _grid(a, b, n)
    return float(np.trapezoid(fn(x), x) if hasattr(np, "trapezoid") else np.trapz(fn(x), x)), x


def simpson(fn, a, b, n=2):
    if n % 2: raise ValueError("Simpson's rule requires an even number of subintervals.")
    x = _grid(a, b, n); h = (b - a) / n; values = fn(x)
    return float(h / 3 * (values[0] + values[-1] + 4 * np.sum(values[1:-1:2]) + 2 * np.sum(values[2:-2:2]))), x


def simpson_three_eighths(fn, a, b, n=3):
    if n < 3 or n % 3:
        raise ValueError("Simpson's 3/8 rule requires a positive number of subintervals divisible by 3.")
    x = _grid(a, b, n); h = (b - a) / n; values = fn(x)
    interior = values[1:-1]
    weights = np.where(np.arange(1, n) % 3 == 0, 2, 3)
    return float(3 * h / 8 * (values[0] + values[-1] + np.dot(weights, interior))), x


def gaussian(fn, a, b, points=5):
    if points < 1: raise ValueError("Use at least one quadrature point.")
    nodes, weights = leggauss(points); mapped = (a + b) / 2 + (b - a) / 2 * nodes
    return float((b - a) / 2 * np.dot(weights, fn(mapped))), mapped


def newton_cotes(fn, a, b, n=4):
    """Closed Newton-Cotes rule over n equal panels."""
    if n < 1 or b <= a:
        raise ValueError("Require b > a and a positive panel count.")
    nodes = np.linspace(a, b, n + 1)
    weights, error = scipy_newton_cotes(n, equal=1)
    return float((b - a) / n * np.dot(weights, fn(nodes))), nodes, float(error)
