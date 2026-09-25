"""Tests for LaTeX parsing and AST translation."""

import pytest
import sympy
from math_engine.latex_parser import parse_latex_to_sympy, parse_latex_to_ast, clean_latex
from math_engine.ast import EquationNode, AddNode, MulNode, DivNode, PowNode, NumberNode, SymbolNode


def test_clean_latex():
    assert clean_latex("$2x + 5 = 15$") == "2x + 5 = 15"
    assert clean_latex(r"\[ \frac{1}{2} \]") == r"\frac{1}{2}"


def test_parse_linear_equation():
    eq = parse_latex_to_sympy("2x + 5 = 15")
    assert isinstance(eq, sympy.Eq)
    assert eq.lhs == 2 * sympy.Symbol("x") + 5
    assert eq.rhs == 15


def test_parse_quadratic_equation():
    eq = parse_latex_to_sympy("x^2 - 5x + 6 = 0")
    assert isinstance(eq, sympy.Eq)
    x = sympy.Symbol("x")
    assert sympy.simplify(eq.lhs - (x**2 - 5*x + 6)) == 0


def test_parse_fraction():
    expr = parse_latex_to_sympy(r"\frac{1}{x - 2} = 3")
    assert isinstance(expr, sympy.Eq)
    x = sympy.Symbol("x")
    assert expr.lhs == 1 / (x - 2)


def test_parse_sqrt():
    expr = parse_latex_to_sympy(r"\sqrt{x + 1} = 4")
    assert isinstance(expr, sympy.Eq)
    x = sympy.Symbol("x")
    assert expr.lhs == sympy.sqrt(x + 1)


def test_ast_generation():
    ast = parse_latex_to_ast("2(x + 3) = 14")
    assert isinstance(ast, EquationNode)
    assert ast.lhs is not None
    assert ast.rhs is not None

    tree_str = ast.to_ascii_tree()
    assert "Equation" in tree_str
    assert "14" in tree_str

    mermaid = ast.to_mermaid()
    assert "graph TD" in mermaid
