from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np


@dataclass
class RootResult:
    root: float
    value: float
    iterations: list[dict]
    converged: bool
    message: str


def _result(root, fn, history, converged, message):
    return RootResult(float(root), float(fn(root)), history, converged, message)


def bisection(fn: Callable, a: float, b: float, tol=1e-8, max_iter=100) -> RootResult:
    if tol <= 0 or max_iter < 1 or a >= b:
        raise ValueError("Require a < b, positive tolerance, and a positive iteration limit.")
    fa, fb = fn(a), fn(b)
    if fa == 0: return _result(a, fn, [], True, "Exact endpoint root.")
    if fb == 0: return _result(b, fn, [], True, "Exact endpoint root.")
    if fa * fb > 0: raise ValueError("The interval must bracket a sign change.")
    history = []
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = fn(c)
        history.append({"iteration": i, "a": a, "b": b, "x": c, "f(x)": fc, "error": abs(b - a) / 2})
        if abs(fc) <= tol or abs(b - a) / 2 <= tol:
            return _result(c, fn, history, True, "Converged.")
        if fa * fc < 0: b, fb = c, fc
        else: a, fa = c, fc
    return _result(c, fn, history, False, "Maximum iterations reached.")


def newton(fn, x0, tol=1e-8, max_iter=100, derivative_fn=None) -> RootResult:
    if tol <= 0 or max_iter < 1: raise ValueError("Tolerance and iteration limit must be positive.")
    derivative_fn = derivative_fn or (lambda x: (fn(x + 1e-6) - fn(x - 1e-6)) / 2e-6)
    history = []
    x = float(x0)
    for i in range(1, max_iter + 1):
        fx, dfx = fn(x), derivative_fn(x)
        if abs(dfx) < 1e-14: raise ValueError("Derivative is too close to zero.")
        next_x = x - fx / dfx
        history.append({"iteration": i, "x": x, "f(x)": fx, "derivative": dfx, "error": abs(next_x - x)})
        if abs(next_x - x) <= tol or abs(fn(next_x)) <= tol:
            return _result(next_x, fn, history, True, "Converged.")
        x = next_x
        if not np.isfinite(x): break
    return _result(x, fn, history, False, "Maximum iterations reached or non-finite iterate.")


def secant(fn, x0, x1, tol=1e-8, max_iter=100) -> RootResult:
    if tol <= 0 or max_iter < 1: raise ValueError("Tolerance and iteration limit must be positive.")
    history, f0, f1 = [], fn(x0), fn(x1)
    for i in range(1, max_iter + 1):
        if abs(f1 - f0) < 1e-14: raise ValueError("Secant denominator is too close to zero.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append({"iteration": i, "x0": x0, "x1": x1, "x": x2, "f(x)": fn(x2), "error": abs(x2 - x1)})
        if abs(x2 - x1) <= tol or abs(fn(x2)) <= tol:
            return _result(x2, fn, history, True, "Converged.")
        x0, f0, x1, f1 = x1, f1, x2, fn(x2)
    return _result(x1, fn, history, False, "Maximum iterations reached.")


def fixed_point(g, x0, tol=1e-8, max_iter=100, fn=None) -> RootResult:
    if tol <= 0 or max_iter < 1: raise ValueError("Tolerance and iteration limit must be positive.")
    fn = fn or (lambda x: g(x) - x)
    history, x = [], float(x0)
    for i in range(1, max_iter + 1):
        nxt = g(x)
        history.append({"iteration": i, "x": x, "g(x)": nxt, "error": abs(nxt - x)})
        if not np.isfinite(nxt): break
        if abs(nxt - x) <= tol:
            return _result(nxt, fn, history, True, "Converged.")
        x = nxt
    return _result(x, fn, history, False, "Maximum iterations reached or iteration diverged.")

