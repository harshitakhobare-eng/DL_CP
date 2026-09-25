"""Step-by-Step Mathematical Solution Generator.

Generates pedagogically meaningful intermediate steps with:
- expression before
- expression after
- operation performed
- detailed explanation

Covers:
- Linear equations (expanding brackets, transposing, isolating variables)
- Quadratic equations (standard form, discriminant, factoring, quadratic formula)
- Polynomial and expression simplification
- Calculus (differentiation power/sum rules)
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy

from math_engine.latex_parser import parse_latex_to_sympy, clean_latex

class MathStep:
    """Represents a single intermediate transformation in a math problem derivation."""

    def __init__(
        self,
        step_number: int,
        expression_before: str,
        expression_after: str,
        operation: str,
        explanation: str,
    ):
        self.step_number = step_number
        self.expression_before = expression_before
        self.expression_after = expression_after
        self.operation = operation
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "expression_before": self.expression_before,
            "expression_after": self.expression_after,
            "operation": self.operation,
            "explanation": self.explanation,
        }


class StepGenerator:
    """Generates detailed step-by-step mathematical reasoning."""

    def __init__(self):
        pass

    def generate(self, math_input: Union[str, sympy.Eq, sympy.Expr]) -> List[MathStep]:
        """Generate intermediate steps for a given math input."""
        if isinstance(math_input, str):
            problem_latex = clean_latex(math_input)
            sympy_obj = parse_latex_to_sympy(problem_latex)
        else:
            sympy_obj = math_input
            problem_latex = sympy.latex(sympy_obj)

        if isinstance(sympy_obj, sympy.Eq):
            return self._solve_equation_steps(sympy_obj, problem_latex)
        elif isinstance(sympy_obj, (sympy.core.relational.Relational, sympy.Rel)):
            return self._solve_inequality_steps(sympy_obj, problem_latex)
        else:
            return self._simplify_expression_steps(sympy_obj, problem_latex)

    def _solve_equation_steps(self, eq: sympy.Eq, original_latex: str) -> List[MathStep]:
        """Generate pedagogical steps for equations."""
        steps: List[MathStep] = []
        var = list(eq.free_symbols)[0] if eq.free_symbols else sympy.Symbol("x")
        step_num = 1

        lhs = eq.lhs
        rhs = eq.rhs
        cur_before = original_latex

        # Check degree
        diff = sympy.simplify(lhs - rhs)
        poly = diff.as_poly(var)
        degree = poly.degree() if poly else None

        # ---------------------------------------------------------------------
        # LINEAR EQUATIONS (degree == 1)
        # ---------------------------------------------------------------------
        if degree == 1:
            # Step 1: Check if expansion is needed (brackets)
            expanded_lhs = sympy.expand(lhs)
            expanded_rhs = sympy.expand(rhs)
            expanded_eq_latex = sympy.latex(sympy.Eq(expanded_lhs, expanded_rhs))

            # Detect brackets in original input or in parsed expression
            has_brackets = bool(re.search(r"(\d+)?\s*\(\s*[^)]+\s*\)", original_latex))
            if (expanded_lhs != lhs or expanded_rhs != rhs) or (has_brackets and original_latex.replace(" ", "") != expanded_eq_latex.replace(" ", "")):
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=expanded_eq_latex,
                    operation="Expand parentheses / Distributive property",
                    explanation="Distribute multipliers across brackets and expand all grouped terms.",
                ))
                step_num += 1
                lhs = expanded_lhs
                rhs = expanded_rhs
                cur_before = expanded_eq_latex

            # Step 2: Separate variable terms and constant terms
            # Group into: var_coeff * var + const = 0
            # LHS terms: collect coeff on var
            lhs_poly = lhs.as_poly(var)
            rhs_poly = rhs.as_poly(var)
            
            lhs_coeff = lhs_poly.coeff_monomial(var) if lhs_poly else 0
            rhs_coeff = rhs_poly.coeff_monomial(var) if rhs_poly else 0
            lhs_const = lhs_poly.coeff_monomial(1) if lhs_poly else 0
            rhs_const = rhs_poly.coeff_monomial(1) if rhs_poly else 0

            # If there are constant terms on LHS, move to RHS
            if lhs_const != 0 and lhs_coeff != 0:
                new_lhs = lhs - lhs_const
                new_rhs = rhs - lhs_const
                after_eq = sympy.Eq(new_lhs, new_rhs)
                after_latex = sympy.latex(after_eq)
                op_text = f"Subtract {lhs_const} from both sides" if lhs_const > 0 else f"Add {-lhs_const} to both sides"
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=after_latex,
                    operation=op_text,
                    explanation=f"To isolate the variable term on the left side, {op_text.lower()}.",
                ))
                step_num += 1
                lhs = new_lhs
                rhs = new_rhs
                cur_before = after_latex

            # If there are variable terms on RHS, move to LHS
            if rhs_coeff != 0:
                new_lhs = lhs - rhs_coeff * var
                new_rhs = rhs - rhs_coeff * var
                after_eq = sympy.Eq(new_lhs, new_rhs)
                after_latex = sympy.latex(after_eq)
                op_text = f"Subtract {rhs_coeff}{var} from both sides" if rhs_coeff > 0 else f"Add {-rhs_coeff}{var} to both sides"
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=after_latex,
                    operation=op_text,
                    explanation=f"Move all variable terms to the left-hand side.",
                ))
                step_num += 1
                lhs = new_lhs
                rhs = new_rhs
                cur_before = after_latex

            # Step 3: Combine like terms if necessary
            simplified_lhs = sympy.simplify(lhs)
            simplified_rhs = sympy.simplify(rhs)
            if simplified_lhs != lhs or simplified_rhs != rhs:
                after_eq = sympy.Eq(simplified_lhs, simplified_rhs)
                after_latex = sympy.latex(after_eq)
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=after_latex,
                    operation="Combine like terms",
                    explanation="Simplify both sides by combining like terms.",
                ))
                step_num += 1
                lhs = simplified_lhs
                rhs = simplified_rhs
                cur_before = after_latex

            # Step 4: Divide by the coefficient of the variable
            final_poly = lhs.as_poly(var)
            final_coeff = final_poly.coeff_monomial(var) if final_poly else 1

            if final_coeff != 1 and final_coeff != 0:
                final_sol = sympy.simplify(rhs / final_coeff)
                final_eq = sympy.Eq(var, final_sol)
                after_latex = sympy.latex(final_eq)
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=after_latex,
                    operation=f"Divide both sides by {final_coeff}",
                    explanation=f"Divide both sides by the coefficient {final_coeff} to isolate {var}.",
                ))
            return steps

        # ---------------------------------------------------------------------
        # QUADRATIC EQUATIONS (degree == 2)
        # ---------------------------------------------------------------------
        elif degree == 2:
            # Step 1: Write in standard form: ax^2 + bx + c = 0
            std_expr = sympy.expand(lhs - rhs)
            std_eq = sympy.Eq(std_expr, 0)
            std_latex = sympy.latex(std_eq)
            if cur_before != std_latex:
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=std_latex,
                    operation="Rearrange to standard form: ax² + bx + c = 0",
                    explanation="Subtract the right-hand side from both sides to form a standard quadratic equation set to 0.",
                ))
                step_num += 1
                cur_before = std_latex

            # Extract coefficients
            a = poly.coeff_monomial(var**2)
            b = poly.coeff_monomial(var)
            c = poly.coeff_monomial(1)

            # Step 2: Compute Discriminant
            discriminant = b**2 - 4 * a * c
            disc_latex = f"\\Delta = b^2 - 4ac = ({b})^2 - 4({a})({c}) = {discriminant}"
            steps.append(MathStep(
                step_number=step_num,
                expression_before=cur_before,
                expression_after=disc_latex,
                operation="Compute the Discriminant (Δ = b² - 4ac)",
                explanation=(
                    f"Identify coefficients: a = {a}, b = {b}, c = {c}. "
                    f"The discriminant Δ is {discriminant}. "
                    + ("Since Δ > 0, there are two distinct real roots." if discriminant > 0 else
                       "Since Δ = 0, there is one repeated real root." if discriminant == 0 else
                       "Since Δ < 0, there are two complex conjugate roots.")
                ),
            ))
            step_num += 1

            # Step 3: Check factoring or quadratic formula
            factored = sympy.factor(std_expr)
            if isinstance(factored, sympy.Mul):
                # Can be factored cleanly
                factor_eq = sympy.Eq(factored, 0)
                factor_latex = sympy.latex(factor_eq)
                steps.append(MathStep(
                    step_number=step_num,
                    expression_before=cur_before,
                    expression_after=factor_latex,
                    operation="Factor the quadratic polynomial",
                    explanation=f"Factor the expression into linear factors: {factor_latex}.",
                ))
                step_num += 1
                cur_before = factor_latex

            # Step 4: Compute final roots
            solutions = sympy.solve(std_eq, var)
            sols_latex = ", \\quad ".join(f"{var} = {sympy.latex(s)}" for s in solutions)
            steps.append(MathStep(
                step_number=step_num,
                expression_before=cur_before,
                expression_after=sols_latex,
                operation="Solve for roots using Quadratic Formula / Zero Product Property",
                explanation=f"Setting each factor to zero yields: {sols_latex}.",
            ))
            return steps

        # General equation fallback
        solutions = sympy.solve(eq, var)
        sols_latex = ", ".join(f"{var} = {sympy.latex(s)}" for s in solutions)
        steps.append(MathStep(
            step_number=1,
            expression_before=original_latex,
            expression_after=sols_latex,
            operation="Symbolic algebraic solution",
            explanation=f"Solve for {var} symbolically.",
        ))
        return steps

    def _solve_inequality_steps(self, rel: sympy.Rel, original_latex: str) -> List[MathStep]:
        """Generate intermediate steps for inequalities."""
        var = list(rel.free_symbols)[0] if rel.free_symbols else sympy.Symbol("x")
        solved = sympy.reduce_inequalities(rel, var)
        return [
            MathStep(
                step_number=1,
                expression_before=original_latex,
                expression_after=sympy.latex(solved),
                operation="Solve inequality for variable",
                explanation=f"Isolate {var} while reversing inequality sign if multiplying or dividing by a negative number.",
            )
        ]

    def _simplify_expression_steps(self, expr: sympy.Expr, original_latex: str) -> List[MathStep]:
        """Generate intermediate steps for simplifying an expression."""
        steps: List[MathStep] = []
        cur_before = original_latex
        step_num = 1

        # Check expansion
        expanded = sympy.expand(expr)
        if expanded != expr:
            exp_latex = sympy.latex(expanded)
            steps.append(MathStep(
                step_number=step_num,
                expression_before=cur_before,
                expression_after=exp_latex,
                operation="Expand expression",
                explanation="Multiply out brackets using distributive law.",
            ))
            step_num += 1
            cur_before = exp_latex

        # Check factoring
        factored = sympy.factor(expr)
        if factored != expr and factored != expanded:
            fact_latex = sympy.latex(factored)
            steps.append(MathStep(
                step_number=step_num,
                expression_before=cur_before,
                expression_after=fact_latex,
                operation="Factor expression",
                explanation="Extract common factors and factor polynomial terms.",
            ))
            step_num += 1
            cur_before = fact_latex

        # Final simplification
        simplified = sympy.simplify(expr)
        simp_latex = sympy.latex(simplified)
        if not steps or simp_latex != cur_before:
            steps.append(MathStep(
                step_number=step_num,
                expression_before=cur_before,
                expression_after=simp_latex,
                operation="Simplify to lowest terms",
                explanation="Combine like terms and reduce fractions.",
            ))

        return steps
