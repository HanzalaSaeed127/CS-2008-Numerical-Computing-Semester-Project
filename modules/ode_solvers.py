from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def solve_ode(fn, x0, y0, x_end, steps, method="RK4"):
    if steps < 1 or x_end == x0: raise ValueError("Use a nonzero interval and positive step count.")
    h = (x_end - x0) / steps; x = np.linspace(x0, x_end, steps + 1); y = np.empty(steps + 1); y[0] = y0
    for i in range(steps):
        xi, yi = x[i], y[i]
        if method == "Euler": y[i + 1] = yi + h * fn(xi, yi)
        elif method == "Heun":
            k1 = fn(xi, yi); k2 = fn(xi + h, yi + h * k1); y[i + 1] = yi + h * (k1 + k2) / 2
        elif method == "RK2":
            k1 = fn(xi, yi); k2 = fn(xi + h / 2, yi + h * k1 / 2); y[i + 1] = yi + h * k2
        elif method == "RK4":
            k1 = fn(xi, yi); k2 = fn(xi + h / 2, yi + h * k1 / 2); k3 = fn(xi + h / 2, yi + h * k2 / 2); k4 = fn(xi + h, yi + h * k3)
            y[i + 1] = yi + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        else: raise ValueError("Unknown ODE method.")
    return x, y


def solve_ode_scipy(fn, x0, y0, x_end, steps, method="RK45"):
    if steps < 1 or x_end == x0:
        raise ValueError("Use a nonzero interval and positive step count.")
    x_eval = np.linspace(x0, x_end, steps + 1)
    result = solve_ivp(lambda x, y: [fn(x, y[0])], (x0, x_end), [y0], t_eval=x_eval, method=method)
    if not result.success:
        raise ValueError(result.message)
    return result.t, result.y[0]
