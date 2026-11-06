import numpy as np


def finite_difference(fn, x, h, scheme="central"):
    if h <= 0: raise ValueError("Step size must be positive.")
    if scheme == "forward": return (fn(x + h) - fn(x)) / h
    if scheme == "backward": return (fn(x) - fn(x - h)) / h
    if scheme == "second": return (fn(x + h) - 2 * fn(x) + fn(x - h)) / h**2
    if scheme == "central": return (fn(x + h) - fn(x - h)) / (2 * h)
    raise ValueError("Unknown finite-difference scheme.")

