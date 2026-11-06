from __future__ import annotations
import numpy as np


def power_iteration(A, x0=None, tol=1e-8, max_iter=500):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]: raise ValueError("A must be square.")
    x = np.ones(A.shape[0]) if x0 is None else np.asarray(x0, dtype=float)
    x /= np.linalg.norm(x); history = []
    for i in range(1, max_iter + 1):
        y = A @ x; next_x = y / np.linalg.norm(y); eigenvalue = float(next_x @ A @ next_x)
        error = np.linalg.norm(next_x - x); history.append({"iteration": i, "eigenvalue": eigenvalue, "error": error})
        if error <= tol: return eigenvalue, next_x, history, True
        x = next_x
    return eigenvalue, x, history, False


def eig(A): return np.linalg.eig(np.asarray(A, dtype=float))


def inverse_power_iteration(A, x0=None, shift=0.0, tol=1e-8, max_iter=500):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    shifted = A - shift * np.eye(A.shape[0])
    x = np.ones(A.shape[0]) if x0 is None else np.asarray(x0, dtype=float)
    x /= np.linalg.norm(x)
    history = []
    for i in range(1, max_iter + 1):
        try:
            y = np.linalg.solve(shifted, x)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Shifted matrix is singular.") from exc
        next_x = y / np.linalg.norm(y)
        eigenvalue = float(next_x @ A @ next_x)
        error = np.linalg.norm(next_x - x)
        history.append({"iteration": i, "eigenvalue": eigenvalue, "error": error})
        if error <= tol:
            return eigenvalue, next_x, history, True
        x = next_x
    return eigenvalue, x, history, False
