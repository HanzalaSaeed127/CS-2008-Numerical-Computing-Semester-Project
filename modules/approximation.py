import numpy as np


def least_squares(A, b):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if A.ndim != 2 or b.shape != (A.shape[0],): raise ValueError("A and b dimensions do not match.")
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    return x, residuals, rank, singular


def gram_schmidt(A):
    A = np.asarray(A, dtype=float); Q = np.zeros_like(A); R = np.zeros((A.shape[1], A.shape[1]))
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j): R[i, j] = Q[:, i] @ v; v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] < 1e-14: raise ValueError("Columns are linearly dependent.")
        Q[:, j] = v / R[j, j]
    return Q, R


def qr(A): return np.linalg.qr(np.asarray(A, dtype=float), mode="reduced")


def svd(A): return np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)

