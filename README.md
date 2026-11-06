# Numerica

Numerica is a Streamlit calculator for numerical methods. It uses Python, NumPy, SciPy, and SymPy.

## Features

- Root finding: bisection, Newton-Raphson, secant, and fixed-point iteration.
- Interpolation and polynomial fitting: Lagrange, Newton divided differences, cubic splines, and least-squares polynomial fits.
- Integration: midpoint, trapezoidal, Simpson 1/3, and Gauss-Legendre quadrature.
- Finite differences and Euler, Heun, RK2, and RK4 ODE solvers.
- Linear systems, LU/Cholesky factorization, Jacobi and Gauss-Seidel iteration.
- Least squares, QR, Gram-Schmidt, SVD, bounded optimization, and power iteration.
- Error metrics and IEEE-754 single/double precision inspection.
- Natural equation entry with implicit multiplication (`2x`, `3(x+1)`), `^` powers,
  `ln`, live SymPy validation, and rendered LaTeX previews.
- Editable point, matrix, and vector tables with validation and configurable
  numerical display precision.
- Iteration tables, CSV downloads, convergence plots, and input validation.

## Setup on Windows

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the app with:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

You can also double-click `run_app.bat`.

Open http://localhost:8501. To run tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Expression examples

Use natural mathematical notation with `x`, or `x` and `y` for ODEs:
`x^3 - x - 2`, `2x + 3`, `3(x + 1)`, `sin(x) + exp(-x)`, and
`y - x^2 + 1`. Supported functions include trigonometric functions, `exp`,
`log`/`ln`, `sqrt`, `abs`, and constants `pi` and `e`. Inputs are parsed by a
restricted SymPy allowlist; Python code, imports, attributes, and arbitrary
function execution are rejected.

Matrices are entered one row per line with comma-separated values, for example:

```text
4,1
2,3
```

## Project structure

Numerical methods are in `modules/`, the Streamlit interface is in `app.py`,
and reusable UI components are in `ui/`.
