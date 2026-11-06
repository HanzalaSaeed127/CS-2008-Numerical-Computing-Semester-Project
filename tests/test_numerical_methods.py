import numpy as np
import pytest

from modules.expression_parser import make_function
from modules.roots import bisection, newton, secant, fixed_point
from modules.interpolation import lagrange, newton_divided_difference, cubic_spline
from modules.integration import midpoint, trapezoidal, simpson, gaussian
from modules.differentiation import finite_difference
from modules.ode_solvers import solve_ode
from modules.linear_algebra import gaussian_elimination, _lu, cholesky, iterative_solve
from modules.approximation import gram_schmidt, qr, svd
from modules.eigenvalues import power_iteration
from modules.formatting import format_number


def test_roots():
    f = make_function("x**3-x-2")
    expected = 1.5213797
    assert bisection(f, 1, 2).root == pytest.approx(expected, abs=1e-6)
    assert newton(f, 1.5).root == pytest.approx(expected, abs=1e-6)
    assert secant(f, 1, 2).root == pytest.approx(expected, abs=1e-6)
    assert fixed_point(make_function("(x+2)**(1/3)"), 1.5, max_iter=500).root == pytest.approx(expected, abs=1e-5)


def test_interpolation_and_integration():
    x, y = np.array([0, 1, 2]), np.array([0, 1, 4])
    assert lagrange(x, y, [1.5])[0] == pytest.approx(2.25)
    assert newton_divided_difference(x, y, [1.5])[0][0] == pytest.approx(2.25)
    assert cubic_spline(x, y, [1])[0] == pytest.approx(1)
    f = make_function("x**2")
    assert midpoint(f, 0, 1, 100)[0] == pytest.approx(1/3, rel=1e-4)
    assert trapezoidal(f, 0, 1, 100)[0] == pytest.approx(1/3, rel=1e-4)
    assert simpson(f, 0, 1, 100)[0] == pytest.approx(1/3, rel=1e-10)
    assert gaussian(f, 0, 1, 5)[0] == pytest.approx(1/3)


def test_differentiation_and_ode():
    f = make_function("x**2")
    assert finite_difference(f, 2, 1e-5) == pytest.approx(4, rel=1e-5)
    x, y = solve_ode(make_function("y", ("x", "y")), 0, 1, 1, 20, "RK4")
    assert y[-1] == pytest.approx(np.e, rel=1e-5)


def test_linear_algebra_and_factorizations():
    A, b = np.array([[3., 2], [1, 2]]), np.array([5., 5])
    x, _ = gaussian_elimination(A, b)
    assert A @ x == pytest.approx(b)
    L, U = _lu(A)
    assert L @ U == pytest.approx(A)
    assert cholesky(np.array([[4., 1], [1, 3]])) @ cholesky(np.array([[4., 1], [1, 3]])).T == pytest.approx(np.array([[4., 1], [1, 3]]))
    x, _, converged = iterative_solve(np.array([[4., 1], [2, 3]]), np.array([1., 2]), "Jacobi")
    assert converged


def test_approximation_and_eigen():
    A = np.array([[1., 0], [0, 1], [1, 1]])
    q, r = qr(A); assert q.T @ q == pytest.approx(np.eye(2)); assert q @ r == pytest.approx(A)
    q, r = gram_schmidt(A); assert q.T @ q == pytest.approx(np.eye(2))
    u, s, vh = svd(A); assert u @ np.diag(s) @ vh == pytest.approx(A)
    value, vector, _, converged = power_iteration(np.array([[2., 0], [0, 1]]))
    assert converged and value == pytest.approx(2)


def test_validation():
    with pytest.raises(ValueError): simpson(make_function("x"), 0, 1, 3)
    with pytest.raises(ValueError): lagrange([0, 0], [1, 2], [0.5])


def test_natural_equation_notation():
    cases = {
        "x^3 - x - 2": (2, 4),
        "2x + 3": (2, 7),
        "3(x + 1)": (2, 9),
        "(x + 1)(x - 1)": (2, 3),
        "sin(x) + cos(x)": (0, 1),
        "ln(x)": (1, 0),
        "sqrt(x + 4)": (5, 3),
        "(x^2 + 1)/(x - 2)": (3, 10),
    }
    for expression, (point, expected) in cases.items():
        assert make_function(expression)(point) == pytest.approx(expected)
    with pytest.raises(ValueError):
        make_function("unknown(x)")
    with pytest.raises(ValueError):
        make_function("sqrt(x)")( -1)


def test_display_formatting():
    assert format_number(1.5213797068, 8) == "1.52137971"
    assert format_number(0.3333333333, 8) == "0.33333333"
    assert "e" in format_number(1.2e-8, 8)
