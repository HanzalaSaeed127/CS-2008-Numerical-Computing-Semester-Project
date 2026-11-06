from scipy.optimize import minimize, minimize_scalar
import numpy as np


def bounded_minimize(fn, a, b, tol=1e-8):
    if b <= a: raise ValueError("Upper bound must exceed lower bound.")
    return minimize_scalar(fn, bounds=(a, b), method="bounded", options={"xatol": tol})


def multivariable_minimize(fn, initial, method="BFGS", tol=1e-8):
    return minimize(fn, initial, method=method, tol=tol)


def make_multivariable_function(expression):
    from .expression_parser import make_function
    return make_function(expression, ("x", "y"))
