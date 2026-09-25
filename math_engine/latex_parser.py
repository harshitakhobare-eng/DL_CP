"""LaTeX Parser and Mathematical Translation Engine.

Converts LaTeX expressions into:
1. Internal Mathematical AST (preserving 2D structures: fractions, roots, powers)
2. Deterministic SymPy symbolic expressions

Converts SymPy expressions back into AST representations with full preservation of operations.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from math_engine.ast import (
    ASTNode,
    NumberNode,
    SymbolNode,
    AddNode,
    SubNode,
    MulNode,
    DivNode,
    PowNode,
    SqrtNode,
    EquationNode,
    InequalityNode,
    FuncNode,
)


def clean_latex(latex_str: str) -> str:
    """Clean and normalize raw LaTeX strings."""
    if not latex_str:
        return ""
    s = latex_str.strip()
    # Remove math mode delimiters if present
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    # Remove \left and \right formatting
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\quad", "")
    s = s.strip()
    return s


def latex_to_sympy_string(latex_str: str) -> str:
    """Convert LaTeX math syntax to standard SymPy-compatible string syntax."""
    s = clean_latex(latex_str)

    # Replace multiplication symbols
    s = s.replace(r"\cdot", " * ")
    s = s.replace(r"\times", " * ")

    # Expand \frac{num}{den} recursively
    frac_pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_pattern.search(s):
        s = frac_pattern.sub(r"((\1)/(\2))", s)

    # Expand \sqrt[n]{x} and \sqrt{x}
    s = re.sub(r"\\sqrt\[([^{}]+)\]\{([^{}]+)\}", r"(\2)**(1/(\1))", s)
    sqrt_pattern = re.compile(r"\\sqrt\{([^{}]+)\}")
    while sqrt_pattern.search(s):
        s = sqrt_pattern.sub(r"sqrt(\1)", s)

    # Replace power notation: x^{2} -> x**(2) and x^2 -> x**(2)
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([a-zA-Z0-9])", r"**(\1)", s)

    # Common functions
    s = s.replace(r"\sin", "sin")
    s = s.replace(r"\cos", "cos")
    s = s.replace(r"\tan", "tan")
    s = s.replace(r"\log", "log")
    s = s.replace(r"\ln", "log")
    s = s.replace(r"\exp", "exp")

    # Inequalities
    s = s.replace(r"\leq", "<=").replace(r"\le", "<=")
    s = s.replace(r"\geq", ">=").replace(r"\ge", ">=")
    s = s.replace(r"\neq", "!=")

    # Greek letters
    s = s.replace(r"\pi", "pi")
    s = s.replace(r"\alpha", "alpha")
    s = s.replace(r"\beta", "beta")
    s = s.replace(r"\theta", "theta")

    # Clean remaining curly braces
    s = s.replace("{", "(").replace("}", ")")

    return s.strip()


def parse_latex_to_sympy(latex_str: str) -> Union[sympy.Eq, sympy.Rel, sympy.Expr]:
    """Parse a mathematical LaTeX string into a SymPy expression or equation.
    
    Handles:
    - Equations: LHS = RHS -> sympy.Eq
    - Inequalities: LHS <, <=, >, >= RHS -> sympy.Rel
    - Expressions: Polynomials, rationals, functions -> sympy.Expr
    """
    clean = clean_latex(latex_str)
    if not clean:
        raise ValueError("Cannot parse empty mathematical expression.")

    # Check for equations
    if "=" in clean and not ("<=" in clean or ">=" in clean or "!=" in clean):
        parts = clean.split("=")
        if len(parts) == 2:
            lhs_str = latex_to_sympy_string(parts[0])
            rhs_str = latex_to_sympy_string(parts[1])
            transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
            lhs_expr = parse_expr(lhs_str, transformations=transformations)
            rhs_expr = parse_expr(rhs_str, transformations=transformations)
            return sympy.Eq(lhs_expr, rhs_expr)

    # Check for inequalities
    for op_sym, sympy_rel in [
        (r"\leq", "<="), (r"\geq", ">="), (r"\le", "<="), (r"\ge", ">="),
        ("<=", "<="), (">=", ">="), ("<", "<"), (">", ">"),
    ]:
        if op_sym in clean:
            parts = clean.split(op_sym, 1)
            lhs_str = latex_to_sympy_string(parts[0])
            rhs_str = latex_to_sympy_string(parts[1])
            transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
            lhs_expr = parse_expr(lhs_str, transformations=transformations)
            rhs_expr = parse_expr(rhs_str, transformations=transformations)
            if sympy_rel == "<=":
                return sympy.Le(lhs_expr, rhs_expr)
            elif sympy_rel == ">=":
                return sympy.Ge(lhs_expr, rhs_expr)
            elif sympy_rel == "<":
                return sympy.Lt(lhs_expr, rhs_expr)
            elif sympy_rel == ">":
                return sympy.Gt(lhs_expr, rhs_expr)

    # Standard expression
    sympy_str = latex_to_sympy_string(clean)
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(sympy_str, transformations=transformations)


def sympy_to_ast(expr: Union[sympy.Eq, sympy.Rel, sympy.Expr, Any]) -> ASTNode:
    """Recursively convert a SymPy expression or equation into an ASTNode hierarchy."""
    # Equation: Eq(lhs, rhs)
    if isinstance(expr, sympy.Eq):
        return EquationNode(
            lhs=sympy_to_ast(expr.lhs),
            rhs=sympy_to_ast(expr.rhs),
        )

    # Inequalities
    if isinstance(expr, (sympy.core.relational.Relational, sympy.Rel)):
        rel_op = expr.rel_op
        return InequalityNode(
            lhs=sympy_to_ast(expr.lhs),
            op=rel_op,
            rhs=sympy_to_ast(expr.rhs),
        )

    # Numbers
    if isinstance(expr, (sympy.Integer, sympy.core.numbers.Integer)):
        return NumberNode(int(expr))
    if isinstance(expr, (sympy.Float, sympy.core.numbers.Float)):
        return NumberNode(float(expr))
    if isinstance(expr, (sympy.Rational, sympy.core.numbers.Rational)):
        if expr.q == 1:
            return NumberNode(int(expr.p))
        return DivNode(NumberNode(int(expr.p)), NumberNode(int(expr.q)))

    # Symbols
    if isinstance(expr, (sympy.Symbol, sympy.core.symbol.Symbol)):
        return SymbolNode(expr.name)

    # Square Root special case: Pow(x, 1/2)
    if isinstance(expr, sympy.Pow):
        if expr.exp == sympy.Rational(1, 2):
            return SqrtNode(sympy_to_ast(expr.base))
        return PowNode(sympy_to_ast(expr.base), sympy_to_ast(expr.exp))

    # Additions
    if isinstance(expr, sympy.Add):
        # Separate positive and negative terms for natural subtraction representation
        operands = [sympy_to_ast(arg) for arg in expr.args]
        return AddNode(operands)

    # Multiplications
    if isinstance(expr, sympy.Mul):
        # Check if there is a division (negative power or rational fraction)
        numerators = []
        denominators = []
        for factor in expr.args:
            if isinstance(factor, sympy.Pow) and factor.exp.is_negative:
                pos_exp = -factor.exp
                if pos_exp == 1:
                    denominators.append(sympy_to_ast(factor.base))
                else:
                    denominators.append(PowNode(sympy_to_ast(factor.base), sympy_to_ast(pos_exp)))
            elif isinstance(factor, sympy.Rational) and factor.q != 1:
                if factor.p != 1:
                    numerators.append(NumberNode(factor.p))
                denominators.append(NumberNode(factor.q))
            else:
                numerators.append(sympy_to_ast(factor))

        if len(denominators) > 0:
            num_node = numerators[0] if len(numerators) == 1 else MulNode(numerators) if len(numerators) > 1 else NumberNode(1)
            den_node = denominators[0] if len(denominators) == 1 else MulNode(denominators)
            return DivNode(num_node, den_node)

        return MulNode([sympy_to_ast(arg) for arg in expr.args])

    # Functions
    if isinstance(expr, sympy.Function):
        func_name = expr.func.__name__
        arg = sympy_to_ast(expr.args[0]) if len(expr.args) == 1 else ASTNode()
        return FuncNode(func_name, arg)

    # Fallback to string-based symbol node
    return SymbolNode(str(expr))


def parse_latex_to_ast(latex_str: str) -> ASTNode:
    """End-to-end conversion from LaTeX string to structured AST."""
    sympy_obj = parse_latex_to_sympy(latex_str)
    return sympy_to_ast(sympy_obj)
