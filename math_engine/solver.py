"""Deterministic Symbolic Mathematical Solver using SymPy.

Supports:
- Linear equations (single & systems)
- Quadratic equations (factoring, quadratic formula, real and complex roots)
- Basic inequalities
- Expression simplification, factorization, expansion
- Rational expressions & fractions
- Differentiation and Integration
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy

from math_engine.latex_parser import parse_latex_to_sympy, parse_latex_to_ast, clean_latex

logger = logging.getLogger(__name__)


class SolverResult:
    """Encapsulates the result of a deterministic mathematical solve."""

    def __init__(
        self,
        problem_latex: str,
        solution_type: str,
        solutions: List[str],
        solutions_latex: List[str],
        variable: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.problem_latex = problem_latex
        self.solution_type = solution_type
        self.solutions = solutions
        self.solutions_latex = solutions_latex
        self.variable = variable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_latex": self.problem_latex,
            "solution_type": self.solution_type,
            "solutions": self.solutions,
            "solutions_latex": self.solutions_latex,
            "variable": self.variable,
            "details": self.details,
        }


class SymbolicSolver:
    """Deterministic mathematical solver using SymPy."""

    def __init__(self):
        pass

    def solve(
        self,
        math_input: Union[str, sympy.Eq, sympy.Rel, sympy.Expr],
        target_var: Optional[str] = None,
    ) -> SolverResult:
        """Solve a mathematical equation, inequality, or simplify an expression.
        
        Deterministic: No approximations or LLMs.
        """
        if isinstance(math_input, str):
            problem_latex = clean_latex(math_input)
            sympy_obj = parse_latex_to_sympy(problem_latex)
        else:
            sympy_obj = math_input
            problem_latex = sympy.latex(sympy_obj)

        # Detect variables in expression
        free_symbols = list(sympy_obj.free_symbols)
        if target_var:
            var = sympy.Symbol(target_var)
        elif len(free_symbols) > 0:
            # Sort by name, default to 'x' if present
            var_names = [s.name for s in free_symbols]
            var = sympy.Symbol("x") if "x" in var_names else free_symbols[0]
        else:
            var = sympy.Symbol("x")

        # 1. Equations: Eq(LHS, RHS)
        if isinstance(sympy_obj, sympy.Eq):
            return self._solve_equation(sympy_obj, var, problem_latex)

        # 2. Inequalities: Relational
        if isinstance(sympy_obj, (sympy.core.relational.Relational, sympy.Rel)):
            return self._solve_inequality(sympy_obj, var, problem_latex)

        # 3. Expressions (simplification, expansion, factorization)
        return self._solve_expression(sympy_obj, var, problem_latex)

    def _solve_equation(self, eq: sympy.Eq, var: sympy.Symbol, problem_latex: str) -> SolverResult:
        """Solve algebraic equations: linear, quadratic, polynomial, rational."""
        # Convert equation to standard form: expr = 0
        diff_expr = sympy.simplify(eq.lhs - eq.rhs)

        # Check for identities and contradictions
        if diff_expr == 0:
            return SolverResult(
                problem_latex=problem_latex,
                solution_type="identity",
                solutions=["All real numbers"],
                solutions_latex=[r"x \in \mathbb{R}"],
                variable=var.name,
                details={"status": "identity", "message": "Identity: equation is true for all values."},
            )

        if len(diff_expr.free_symbols) == 0:
            # Contradiction, e.g. 5 = 15 -> -10 = 0
            return SolverResult(
                problem_latex=problem_latex,
                solution_type="no_solution",
                solutions=["No solution"],
                solutions_latex=[r"\emptyset"],
                variable=var.name,
                details={"status": "contradiction", "message": "Contradiction: no solutions exist."},
            )

        # Determine equation degree
        poly = diff_expr.as_poly(var)
        degree = poly.degree() if poly else None

        # Solve symbolically
        raw_solutions = sympy.solve(eq, var)
        
        solutions = []
        solutions_latex = []
        for sol in raw_solutions:
            sol_str = f"{var.name} = {sol}"
            sol_latex = f"{var.name} = {sympy.latex(sol)}"
            solutions.append(sol_str)
            solutions_latex.append(sol_latex)

        if degree == 1:
            sol_type = "linear_equation"
        elif degree == 2:
            sol_type = "quadratic_equation"
        elif degree and degree > 2:
            sol_type = f"polynomial_degree_{degree}"
        else:
            sol_type = "algebraic_equation"

        if not solutions:
            solutions = ["No real solution"]
            solutions_latex = [r"\text{No real solution}"]

        return SolverResult(
            problem_latex=problem_latex,
            solution_type=sol_type,
            solutions=solutions,
            solutions_latex=solutions_latex,
            variable=var.name,
            details={"degree": degree},
        )

    def _solve_inequality(self, rel: sympy.Rel, var: sympy.Symbol, problem_latex: str) -> SolverResult:
        """Solve inequalities."""
        try:
            sol = sympy.reduce_inequalities(rel, var)
            sol_str = str(sol)
            sol_latex = sympy.latex(sol)
            return SolverResult(
                problem_latex=problem_latex,
                solution_type="inequality",
                solutions=[sol_str],
                solutions_latex=[sol_latex],
                variable=var.name,
            )
        except Exception as e:
            logger.warning(f"Could not reduce inequality: {e}")
            return SolverResult(
                problem_latex=problem_latex,
                solution_type="inequality",
                solutions=[str(rel)],
                solutions_latex=[sympy.latex(rel)],
                variable=var.name,
            )

    def _solve_expression(self, expr: sympy.Expr, var: sympy.Symbol, problem_latex: str) -> SolverResult:
        """Simplify, expand, and factor standalone mathematical expressions."""
        simplified = sympy.simplify(expr)
        expanded = sympy.expand(expr)
        factored = sympy.factor(expr)

        solutions = [f"Simplified: {simplified}"]
        solutions_latex = [f"\\text{{Simplified: }} {sympy.latex(simplified)}"]

        if factored != simplified:
            solutions.append(f"Factored: {factored}")
            solutions_latex.append(f"\\text{{Factored: }} {sympy.latex(factored)}")

        if expanded != simplified and expanded != factored:
            solutions.append(f"Expanded: {expanded}")
            solutions_latex.append(f"\\text{{Expanded: }} {sympy.latex(expanded)}")

        return SolverResult(
            problem_latex=problem_latex,
            solution_type="expression_simplification",
            solutions=solutions,
            solutions_latex=solutions_latex,
            variable=var.name,
            details={
                "simplified": str(simplified),
                "factored": str(factored),
                "expanded": str(expanded),
            },
        )

    def differentiate(self, math_input: Union[str, sympy.Expr], var_name: str = "x") -> Dict[str, Any]:
        """Compute symbolic derivative d/dx."""
        expr = parse_latex_to_sympy(math_input) if isinstance(math_input, str) else math_input
        var = sympy.Symbol(var_name)
        derivative = sympy.diff(expr, var)
        return {
            "original_latex": sympy.latex(expr),
            "variable": var_name,
            "derivative_str": str(derivative),
            "derivative_latex": sympy.latex(derivative),
        }

    def integrate(self, math_input: Union[str, sympy.Expr], var_name: str = "x") -> Dict[str, Any]:
        """Compute symbolic indefinite integral."""
        expr = parse_latex_to_sympy(math_input) if isinstance(math_input, str) else math_input
        var = sympy.Symbol(var_name)
        integral = sympy.integrate(expr, var)
        return {
            "original_latex": sympy.latex(expr),
            "variable": var_name,
            "integral_str": str(integral) + " + C",
            "integral_latex": sympy.latex(integral) + " + C",
        }
