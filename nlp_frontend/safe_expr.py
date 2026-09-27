"""
Restricted, safe linear-expression parser for untrusted LLM output.

PR #9 review item 2 (round 2): sympify() over arbitrary LLM-supplied
text is not a security boundary -- sympy's general parsing pathway
accepts far more than we need and was never designed to safely
evaluate untrusted input.

This module parses a *small, fixed grammar* using Python's `ast` module
and walks the resulting tree by hand -- there is no eval()/exec() and
no sympy anywhere here. It accepts only:
  - names that are in the caller-supplied set of declared variables
  - numeric literals (int/float)
  - +, -, unary +/-
  - * and / where at least one operand is a pure numeric constant
  - parentheses (handled naturally by ast's tree structure)

Anything else -- function calls, attribute access, subscripting,
comparisons, boolean/lambda expressions, exponentiation, string
literals, variable*variable products, or references to any name not in
the declared set -- is rejected before any evaluation happens.
"""

import ast
import math


class ExpressionError(ValueError):
    pass


class _LinearExpr:
    """coeffs: dict[str, float] (nonzero entries only); constant: float."""

    __slots__ = ("coeffs", "constant")

    def __init__(self, coeffs=None, constant=0.0):
        self.coeffs = dict(coeffs) if coeffs else {}
        self.constant = float(constant)

    def add(self, other, sign=1.0):
        coeffs = dict(self.coeffs)
        for name, c in other.coeffs.items():
            coeffs[name] = coeffs.get(name, 0.0) + sign * c
            if coeffs[name] == 0.0:
                del coeffs[name]
        return _LinearExpr(coeffs, self.constant + sign * other.constant)

    def scale(self, factor):
        return _LinearExpr({n: c * factor for n, c in self.coeffs.items()}, self.constant * factor)

    def is_pure_constant(self):
        return len(self.coeffs) == 0


def _check_finite(value, where):
    if not math.isfinite(value):
        raise ExpressionError(f"{where}: value is not finite ({value})")


def _eval(node, var_names, where):
    if isinstance(node, ast.Expression):
        return _eval(node.body, var_names, where)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"{where}: literal must be a plain number, got {node.value!r}")
        _check_finite(float(node.value), where)
        return _LinearExpr({}, float(node.value))

    if isinstance(node, ast.Name):
        if node.id not in var_names:
            raise ExpressionError(f"{where} references undeclared variable(s): {{{node.id}}}")
        return _LinearExpr({node.id: 1.0}, 0.0)

    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, var_names, where)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return operand.scale(-1.0)
        raise ExpressionError(f"{where}: unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval(node.left, var_names, where)
        right = _eval(node.right, var_names, where)

        if isinstance(node.op, ast.Add):
            return left.add(right, sign=1.0)
        if isinstance(node.op, ast.Sub):
            return left.add(right, sign=-1.0)

        if isinstance(node.op, ast.Mult):
            if left.is_pure_constant():
                return right.scale(left.constant)
            if right.is_pure_constant():
                return left.scale(right.constant)
            raise ExpressionError(f"{where} is nonlinear: cannot multiply two variable terms together")

        if isinstance(node.op, ast.Div):
            if not right.is_pure_constant():
                raise ExpressionError(f"{where} is nonlinear: cannot divide by a variable expression")
            if right.constant == 0.0:
                raise ExpressionError(f"{where}: division by zero")
            return left.scale(1.0 / right.constant)

        raise ExpressionError(f"{where}: unsupported operator {type(node.op).__name__}")

    raise ExpressionError(f"{where}: unsupported syntax ({type(node).__name__})")


def linear_terms(expr_str: str, var_names, where: str):
    """
    Parses `expr_str` under the restricted grammar above.

    var_names: iterable of declared, allowed variable names.
    where: human-readable label used in error messages
           (e.g. "objective", "constraint 'c1'").

    Returns (coeffs: dict[str, float], constant: float).
    Raises ExpressionError on anything outside the grammar, references
    to undeclared variables, nonlinear products/divisions, or
    non-finite literals.
    """
    var_names = set(var_names)
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"Could not parse expression in {where}: '{expr_str}' ({e})")

    result = _eval(tree, var_names, where)

    for name, c in result.coeffs.items():
        _check_finite(c, f"{where} coefficient for '{name}'")
    _check_finite(result.constant, f"{where} constant term")

    return result.coeffs, result.constant