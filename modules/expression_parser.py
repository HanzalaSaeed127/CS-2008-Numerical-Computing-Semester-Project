"""Safe SymPy-backed expression parsing utilities."""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
import sympy as sp
import re
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr,
    standard_transformations,
)


class ExpressionError(ValueError):
    """Raised when a user expression cannot be evaluated safely."""


_ALLOWED = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin,
    "acos": sp.acos, "atan": sp.atan, "exp": sp.exp, "log": sp.log,
    "sqrt": sp.sqrt, "abs": sp.Abs, "ln": sp.log, "pi": sp.pi, "e": sp.E, "E": sp.E,
}
_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)
_SAFE_GLOBALS = {
    "__builtins__": {},
    "Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float,
    "Rational": sp.Rational, "Add": sp.Add, "Mul": sp.Mul, "Pow": sp.Pow,
}


def parse_expression(expression: str, variables: tuple[str, ...] = ("x",)) -> sp.Expr:
    if not expression or not expression.strip():
        raise ExpressionError("Enter a mathematical expression.")
    symbols = {name: sp.Symbol(name, real=True) for name in variables}
    source = expression.strip()
    if "__" in source or not re.fullmatch(r"[A-Za-z0-9_+\-*/^().,\s]+", source):
        raise ExpressionError("Expression contains unsupported characters or syntax.")
    names = set(re.findall(r"[A-Za-z_]\w*", source))
    allowed_names = set(symbols) | set(_ALLOWED)
    if names - allowed_names:
        raise ExpressionError(f"Unsupported name(s): {', '.join(sorted(names - allowed_names))}.")
    try:
        parsed = parse_expr(
            source,
            local_dict={**_ALLOWED, **symbols},
            global_dict=_SAFE_GLOBALS,
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        raise ExpressionError(f"Invalid expression: {exc}") from exc
    if parsed.free_symbols - set(symbols.values()):
        raise ExpressionError(f"Only these variables are allowed: {', '.join(variables)}.")
    return parsed


def make_function(expression: str, variables: tuple[str, ...] = ("x",)) -> Callable:
    parsed = parse_expression(expression, variables)
    symbols = [sp.Symbol(name, real=True) for name in variables]
    fn = sp.lambdify(symbols, parsed, modules=["numpy"])

    def evaluate(*values):
        try:
            with np.errstate(all="raise"):
                result = np.asarray(fn(*values), dtype=float)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
            raise ExpressionError(f"Expression could not be evaluated: {exc}") from exc
        if not np.all(np.isfinite(result)):
            raise ExpressionError("The expression produced a non-finite value.")
        return float(result) if result.ndim == 0 else result

    return evaluate


def derivative(expression: str) -> Callable:
    x = sp.Symbol("x", real=True)
    parsed = parse_expression(expression)
    return make_function(str(sp.diff(parsed, x)))
