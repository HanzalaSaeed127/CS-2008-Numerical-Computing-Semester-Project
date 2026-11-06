# Numerica

Numerica is a local Streamlit numerical-computing calculator for a university numerical methods course. It combines explicit teaching implementations with NumPy/SciPy routines for comparison and supports safe SymPy expressions rather than unrestricted `eval`.

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

## Windows setup

The project uses a local `.venv`, so its packages are kept with the project
and are not affected by closing or reopening VS Code. Open the project folder
(`E:\Study\NC\project`) in VS Code and press `F5`, then select **Run Numerica
(Streamlit)**. VS Code is configured to select `.venv` automatically.

For a first-time setup, run:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To run it manually after setup:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

You can also double-click `run_app.bat` in the project folder. Do not use
`python app.py`; that runs Streamlit outside its normal launcher and causes
`missing ScriptRunContext` warnings.

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

## Structure and limitations

Numerical algorithms are in `modules/` and the presentation layer is in `app.py`. The UI is intentionally local and does not use external APIs. Multivariable optimization and SciPy ODE comparison are represented by reusable modules but the compact UI focuses on the most teachable scalar workflows. Complex eigenvalue display is available through NumPy's eigensolver; power iteration is intended for real matrices with a dominant eigenvalue.
