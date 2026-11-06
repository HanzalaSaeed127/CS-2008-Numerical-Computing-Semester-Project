from __future__ import annotations
import numpy as np


def validate_system(A, b):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],): raise ValueError("A must be square and b must match its rows.")
    return A, b


def condition_number(A, norm=None):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    return float(np.linalg.cond(A, p=norm))


def gaussian_elimination(A, b, pivot=True):
    A, b = validate_system(A, b); M = np.column_stack((A.copy(), b.copy())); n = len(b)
    for k in range(n - 1):
        if pivot:
            row = k + np.argmax(abs(M[k:, k])); M[[k, row]] = M[[row, k]]
        if abs(M[k, k]) < 1e-14: raise ValueError("Matrix is singular or nearly singular.")
        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]; M[i, k:] -= factor * M[k, k:]
    if abs(M[-1, -2]) < 1e-14: raise ValueError("Matrix is singular or nearly singular.")
    x = np.zeros(n)
    for i in range(n - 1, -1, -1): x[i] = (M[i, -1] - M[i, i + 1:-1] @ x[i + 1:]) / M[i, i]
    return x, M


def lu_factor(A):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]: raise ValueError("A must be square.")
    return np.linalg.lu_factor(A) if hasattr(np.linalg, "lu_factor") else _lu(A)


def _lu(A):
    n = len(A); L, U = np.eye(n), A.copy()
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-14: raise ValueError("Zero pivot in LU factorization.")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]; U[i, k:] -= L[i, k] * U[k, k:]
    return L, U


def cholesky(A):
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T): raise ValueError("Cholesky requires a symmetric matrix.")
    try: return np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc: raise ValueError("Matrix is not positive definite.") from exc


def iterative_solve(A, b, method="Jacobi", tol=1e-8, max_iter=500):
    A, b = validate_system(A, b)
    if np.any(abs(np.diag(A)) < 1e-14): raise ValueError("Diagonal entries must be nonzero.")
    x = np.zeros_like(b); history = []
    for i in range(1, max_iter + 1):
        if method == "Jacobi": new = (b - (A - np.diag(np.diag(A))) @ x) / np.diag(A)
        elif method == "Gauss-Seidel":
            new = x.copy()
            for j in range(len(b)): new[j] = (b[j] - A[j, :j] @ new[:j] - A[j, j+1:] @ x[j+1:]) / A[j, j]
        else: raise ValueError("Unknown iterative method.")
        residual = np.linalg.norm(A @ new - b); history.append({"iteration": i, "residual": residual})
        if residual <= tol: return new, history, True
        x = new
    return x, history, False
