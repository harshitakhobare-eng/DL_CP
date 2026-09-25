"""Pedagogical Mistake Explanation Engine.

Delivers the 6 required components for every detected mistake:
1. What the student wrote
2. What was expected
3. Why the step is incorrect
4. Mathematical rule involved
5. The corrected step
6. A short tip to avoid repeating the mistake
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from math_engine.error_classifier import MathErrorClassification


class MistakeExplainer:
    """Generates structured pedagogical explanations for mathematical errors."""

    def __init__(self):
        pass

    def explain(self, classification: MathErrorClassification) -> Dict[str, Any]:
        """Produce the 6-point pedagogical explanation."""
        error_type = classification.error_type
        student_expr = classification.student_expression
        expected_expr = classification.expected_expression
        why_incorrect = classification.explanation
        rule = classification.rule_name or self._default_rule(error_type)
        corrected_step = classification.suggested_correction or expected_expr
        tip = classification.tip or self._default_tip(error_type)

        return {
            "error_type": error_type,
            "step_number": classification.step_number,
            "1_what_student_wrote": student_expr,
            "2_what_was_expected": expected_expr,
            "3_why_incorrect": why_incorrect,
            "4_mathematical_rule": rule,
            "5_corrected_step": corrected_step,
            "6_avoidance_tip": tip,
            "confidence": classification.confidence,
            "is_recognition_suspect": classification.is_recognition_suspect,
        }

    def _default_rule(self, error_type: str) -> str:
        rules = {
            "Distribution Error": "Distributive Property: a(b + c) = a · b + a · c",
            "Sign Error": "Sign Inversion Rule: Moving a term across '=' flips its sign (+ ↔ -)",
            "Arithmetic Error": "Basic Laws of Arithmetic",
            "Fraction Error": "Fraction Equality & Cancellation Rules",
            "Exponent Error": "Product and Power of Powers Rules: (xᵃ)ᵇ = xᵃᵇ, xᵃ · xᵇ = xᵃ⁺ᵇ",
            "Incorrect Transposition": "Multiplicative and Additive Inverse Operations",
            "Incorrect Cancellation": "Cancellation is only valid over common factors of the entire numerator and denominator",
            "Missing Term": "Law of Conservation of Algebraic Terms",
            "Extra Term": "Equivalence Transformation Rules",
            "Recognition Error": "Visual Character Disambiguation",
            "Algebraic Manipulation Error": "Properties of Equality: Whatever is done to one side must be identically done to the other",
        }
        return rules.get(error_type, "Standard Mathematical Deduction")

    def _default_tip(self, error_type: str) -> str:
        tips = {
            "Distribution Error": "When expanding parentheses, multiply the outside term by EVERY term inside the bracket.",
            "Sign Error": "Pay close attention to negative signs. Use parentheses around substituted negative numbers.",
            "Arithmetic Error": "Double check mental arithmetic before moving on to the next step.",
            "Fraction Error": "Find the least common denominator before adding fractions, and never cancel terms in an addition.",
            "Exponent Error": "Remember: (a + b)² = a² + 2ab + b². Don't forget the middle cross-term!",
            "Incorrect Transposition": "To isolate a multiplied variable, divide both sides. Do not subtract.",
            "Incorrect Cancellation": "You can only cancel terms that are multiplied, never terms connected by plus or minus.",
            "Missing Term": "Count your terms before and after each algebraic step.",
            "Recognition Error": "If the handwriting is ambiguous, you can click on the step to correct it.",
        }
        return tips.get(error_type, "Check each step by substituting test values back into the equation.")
