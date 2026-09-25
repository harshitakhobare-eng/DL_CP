"""Mathematical Error Taxonomy and Classification Engine.

Implements structured error taxonomy:
- Sign Error
- Arithmetic Error
- Distribution Error
- Fraction Error
- Exponent Error
- Incorrect Transposition
- Incorrect Cancellation
- Missing Term
- Extra Term
- Incorrect Substitution
- Algebraic Manipulation Error
- Recognition Error
- Unknown/Other

Distinguishes between handwriting recognition ambiguity and genuine mathematical flaws.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy

from math_engine.latex_parser import parse_latex_to_sympy, clean_latex

logger = logging.getLogger(__name__)


# Common OCR visual ambiguities
AMBIGUOUS_CHAR_PAIRS = [
    ("x", "y"),
    ("x", "\\times"),
    ("+", "t"),
    ("1", "l"),
    ("0", "O"),
    ("5", "s"),
    ("2", "z"),
    ("8", "B"),
    ("9", "g"),
    ("-", "="),
    (".", ","),
]


class MathErrorClassification:
    """Encapsulates a classified student error in a derivation step."""

    def __init__(
        self,
        error_type: str,
        step_number: int,
        student_expression: str,
        expected_expression: str,
        explanation: str,
        confidence: float,
        is_recognition_suspect: bool = False,
        suggested_correction: Optional[str] = None,
        tip: Optional[str] = None,
        rule_name: Optional[str] = None,
    ):
        self.error_type = error_type
        self.step_number = step_number
        self.student_expression = student_expression
        self.expected_expression = expected_expression
        self.explanation = explanation
        self.confidence = confidence
        self.is_recognition_suspect = is_recognition_suspect
        self.suggested_correction = suggested_correction or expected_expression
        self.tip = tip or ""
        self.rule_name = rule_name or ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "step_number": self.step_number,
            "student_expression": self.student_expression,
            "expected_expression": self.expected_expression,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 4),
            "is_recognition_suspect": self.is_recognition_suspect,
            "suggested_correction": self.suggested_correction,
            "tip": self.tip,
            "rule_name": self.rule_name,
        }


class ErrorClassifier:
    """Classifies mathematical errors between consecutive derivation steps."""

    def __init__(self):
        pass

    def classify(
        self,
        prev_step_str: str,
        curr_step_str: str,
        step_number: int,
        step_confidences: Optional[List[float]] = None,
        low_confidence_tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> MathErrorClassification:
        """Analyze consecutive steps and determine exact error type."""
        prev_clean = clean_latex(prev_step_str)
        curr_clean = clean_latex(curr_step_str)

        # 1. Check for Recognition Error first
        rec_error = self._check_recognition_error(
            prev_clean, curr_clean, step_number, step_confidences, low_confidence_tokens
        )
        if rec_error:
            return rec_error

        # Parse into SymPy
        try:
            prev_sympy = parse_latex_to_sympy(prev_clean)
            curr_sympy = parse_latex_to_sympy(curr_clean)
        except Exception as e:
            return MathErrorClassification(
                error_type="Unknown/Other",
                step_number=step_number,
                student_expression=curr_clean,
                expected_expression=prev_clean,
                explanation=f"Malformed mathematical syntax: unable to parse expression.",
                confidence=0.5,
                rule_name="Syntax Rules",
                tip="Check for missing brackets or unbalanced operators.",
            )

        # Check for Distribution Error
        dist_error = self._check_distribution_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if dist_error:
            return dist_error

        # Check for Sign Error
        sign_error = self._check_sign_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if sign_error:
            return sign_error

        # Check for Transposition Error
        trans_error = self._check_transposition_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if trans_error:
            return trans_error

        # Check for Arithmetic Error
        arith_error = self._check_arithmetic_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if arith_error:
            return arith_error

        # Check for Exponent Error
        exp_error = self._check_exponent_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if exp_error:
            return exp_error

        # Check for Fraction Error / Incorrect Cancellation
        frac_error = self._check_fraction_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if frac_error:
            return frac_error

        # Check for Missing or Extra Term
        term_error = self._check_term_count_error(prev_sympy, curr_sympy, prev_clean, curr_clean, step_number)
        if term_error:
            return term_error

        # Fallback: Algebraic Manipulation Error
        correct_step = self._generate_correct_transition(prev_sympy)
        return MathErrorClassification(
            error_type="Algebraic Manipulation Error",
            step_number=step_number,
            student_expression=curr_clean,
            expected_expression=correct_step,
            explanation="The mathematical transformation between these consecutive steps is invalid.",
            confidence=0.88,
            rule_name="Fundamental Properties of Equality",
            tip="Ensure that the same valid mathematical operation is applied to both sides of the equation.",
        )

    def _check_recognition_error(
        self,
        prev_str: str,
        curr_str: str,
        step_number: int,
        confidences: Optional[List[float]],
        low_confidence_tokens: Optional[List[Dict[str, Any]]],
    ) -> Optional[MathErrorClassification]:
        """Detect if an apparent error is actually due to OCR misrecognition."""
        # Condition A: Explicit low confidence tokens reported
        if low_confidence_tokens and len(low_confidence_tokens) > 0:
            tokens_str = ", ".join(t.get("token", "?") for t in low_confidence_tokens)
            return MathErrorClassification(
                error_type="Recognition Error",
                step_number=step_number,
                student_expression=curr_str,
                expected_expression=curr_str,
                explanation=(
                    f"Low visual recognition confidence detected on symbol(s): '{tokens_str}'. "
                    "The system is uncertain whether handwriting was recognized correctly."
                ),
                confidence=0.92,
                is_recognition_suspect=True,
                rule_name="Visual Symbol Ambiguity",
                tip="Please verify or edit the recognized expression above.",
            )

        # Condition B: Single character swap between ambiguous pairs creates valid math
        for char1, char2 in AMBIGUOUS_CHAR_PAIRS:
            if char1 in curr_str:
                candidate_str = curr_str.replace(char1, char2, 1)
                try:
                    p_sym = parse_latex_to_sympy(prev_str)
                    c_sym = parse_latex_to_sympy(candidate_str)
                    if self._is_step_valid(p_sym, c_sym):
                        return MathErrorClassification(
                            error_type="Recognition Error",
                            step_number=step_number,
                            student_expression=curr_str,
                            expected_expression=candidate_str,
                            explanation=(
                                f"Possible recognition ambiguity: '{char1}' may have been misrecognized for '{char2}'. "
                                f"With '{char2}', the step is mathematically valid."
                            ),
                            confidence=0.95,
                            is_recognition_suspect=True,
                            rule_name="Character Ambiguity Detection",
                            tip=f"Check if you wrote '{char2}' instead of '{char1}'.",
                        )
                except Exception:
                    pass

        return None

    def _check_distribution_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect: a(x + b) expanded as ax + b instead of ax + ab."""
        # Regex search for distribution pattern in previous string: a(x + b) or a(x - b)
        match = re.search(r"(\d+)\s*\(\s*([a-zA-Z0-9]+)\s*([+-])\s*(\d+)\s*\)", prev_str)
        if match:
            mult = int(match.group(1))
            var_part = match.group(2)
            op = match.group(3)
            const_part = int(match.group(4))

            # Flawed distribution: mult * var + const (multiplier was NOT distributed to const)
            flawed_expansion = f"{mult}{var_part} {op} {const_part}"
            flawed_clean = flawed_expansion.replace(" ", "")

            curr_normalized = curr_str.replace(" ", "")
            if flawed_clean in curr_normalized:
                correct_const = mult * const_part
                correct_expansion = f"{mult}{var_part} {op} {correct_const}"
                return MathErrorClassification(
                    error_type="Distribution Error",
                    step_number=step_number,
                    student_expression=curr_str,
                    expected_expression=curr_str.replace(flawed_expansion, correct_expansion),
                    explanation=(
                        f"The factor {mult} must multiply EVERY term inside the parentheses: "
                        f"{mult} × {var_part} {op} {mult} × {const_part} = {correct_expansion}. "
                        f"Only the first term was multiplied."
                    ),
                    confidence=0.96,
                    rule_name="Distributive Property: a(b + c) = ab + ac",
                    tip="When expanding brackets, multiply the outside number by all terms inside.",
                )

        return None

    def _check_sign_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect sign errors during transposition or subtraction."""
        if isinstance(prev, sympy.Eq) and isinstance(curr, sympy.Eq):
            # Check if student moved a term across '=' but did NOT invert the sign
            # E.g. 2x + 5 = 15 -> 2x = 15 + 5 instead of 15 - 5
            prev_diff = sympy.simplify(prev.lhs - prev.rhs)
            curr_diff = sympy.simplify(curr.lhs - curr.rhs)

            var = list(prev.free_symbols)[0] if prev.free_symbols else sympy.Symbol("x")
            poly_p = prev_diff.as_poly(var)
            poly_c = curr_diff.as_poly(var)

            if poly_p and poly_c and poly_p.degree() == 1 and poly_c.degree() == 1:
                coeff_p = poly_p.coeff_monomial(var)
                coeff_c = poly_c.coeff_monomial(var)
                const_p = poly_p.coeff_monomial(1)
                const_c = poly_c.coeff_monomial(1)

                # If variable coefficients match, but constant has opposite sign or 2*const shift
                if coeff_p == coeff_c and const_c != const_p:
                    # Check if flipping sign of constant matches
                    if const_c == -const_p or const_c == 2 * const_p:
                        expected = sympy.latex(sympy.Eq(coeff_p * var, -const_p))
                        return MathErrorClassification(
                            error_type="Sign Error",
                            step_number=step_number,
                            student_expression=curr_str,
                            expected_expression=expected,
                            explanation=(
                                "A sign error occurred during transposition. "
                                "When moving a term to the opposite side of an equation, its sign must invert."
                            ),
                            confidence=0.94,
                            rule_name="Additive Inverse / Transposition Rule",
                            tip="Remember: adding a term to one side is equivalent to subtracting it from the other.",
                        )
        return None

    def _check_arithmetic_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect basic arithmetic miscalculation in constants."""
        if isinstance(prev, sympy.Eq) and isinstance(curr, sympy.Eq):
            var = list(prev.free_symbols)[0] if prev.free_symbols else sympy.Symbol("x")
            # If the algebraic structure is preserved (same variable coefficient), but RHS constant is wrong
            poly_curr_lhs = curr.lhs.as_poly(var)
            if poly_curr_lhs and poly_curr_lhs.degree() == 1:
                # Expected RHS:
                expected_sol = sympy.solve(prev, curr.lhs)
                if expected_sol:
                    exp_rhs = expected_sol[0]
                    if exp_rhs.is_number and curr.rhs.is_number and exp_rhs != curr.rhs:
                        return MathErrorClassification(
                            error_type="Arithmetic Error",
                            step_number=step_number,
                            student_expression=curr_str,
                            expected_expression=sympy.latex(sympy.Eq(curr.lhs, exp_rhs)),
                            explanation=(
                                f"An arithmetic calculation error occurred on the right-hand side. "
                                f"Evaluating the operation yields {exp_rhs}, not {curr.rhs}."
                            ),
                            confidence=0.95,
                            rule_name="Arithmetic Evaluation",
                            tip="Double-check basic addition, subtraction, multiplication, or division steps.",
                        )
        return None

    def _check_exponent_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect exponent law errors: (x + y)^2 = x^2 + y^2 or (x^a)^b = x^{a+b}."""
        # Freshman's dream: (a + b)^2 -> a^2 + b^2 (missing 2ab cross term)
        match = re.search(r"\(\s*([a-zA-Z0-9]+)\s*\+\s*([a-zA-Z0-9]+)\s*\)\^2", prev_str)
        if match:
            a_part, b_part = match.group(1), match.group(2)
            flawed = f"{a_part}^2 + {b_part}^2".replace(" ", "")
            if flawed in curr_str.replace(" ", ""):
                expected = f"{a_part}^2 + 2{a_part}{b_part} + {b_part}^2"
                return MathErrorClassification(
                    error_type="Exponent Error",
                    step_number=step_number,
                    student_expression=curr_str,
                    expected_expression=curr_str.replace(flawed, expected),
                    explanation=(
                        f"Expanding ({a_part} + {b_part})² requires the cross term 2·{a_part}·{b_part}. "
                        f"Expected: {expected}."
                    ),
                    confidence=0.97,
                    rule_name="Square of a Binomial: (a + b)² = a² + 2ab + b²",
                    tip="Never distribute an exponent across an addition or subtraction.",
                )
        return None

    def _check_fraction_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect invalid fraction cancellation or denominator handling."""
        if r"\frac" in prev_str and r"\frac" not in curr_str:
            # Check if term was cancelled incorrectly across addition: e.g. (2x + 4)/2 -> x + 4
            return MathErrorClassification(
                error_type="Fraction Error",
                step_number=step_number,
                student_expression=curr_str,
                expected_expression=self._generate_correct_transition(prev),
                explanation=(
                    "An error occurred while simplifying or clearing fractions. "
                    "Terms in the numerator cannot be cancelled individually across addition without dividing all terms."
                ),
                confidence=0.91,
                rule_name="Fraction Division & Common Denominators",
                tip="To cancel a common factor, factor it out from the entire numerator first.",
            )
        return None

    def _check_transposition_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect moving terms across equality incorrectly (e.g. subtracting instead of dividing)."""
        if isinstance(prev, sympy.Eq) and isinstance(curr, sympy.Eq):
            var = list(prev.free_symbols)[0] if prev.free_symbols else sympy.Symbol("x")
            # If prev was 2x = 10 and student wrote x = 10 - 2 = 8
            coeff = prev.lhs.as_poly(var).coeff_monomial(var) if prev.lhs.as_poly(var) else 1
            if coeff > 1 and prev.lhs == coeff * var:
                if curr.lhs == var and curr.rhs == prev.rhs - coeff:
                    return MathErrorClassification(
                        error_type="Incorrect Transposition",
                        step_number=step_number,
                        student_expression=curr_str,
                        expected_expression=sympy.latex(sympy.Eq(var, sympy.Rational(prev.rhs, coeff))),
                        explanation=(
                            f"The coefficient {coeff} is multiplied by {var}. "
                            f"To isolate {var}, you must divide both sides by {coeff}, not subtract."
                        ),
                        confidence=0.96,
                        rule_name="Multiplicative Inverse Property",
                        tip=f"Undo multiplication by dividing both sides by {coeff}.",
                    )
        return None

    def _check_term_count_error(
        self,
        prev: Union[sympy.Eq, sympy.Expr],
        curr: Union[sympy.Eq, sympy.Expr],
        prev_str: str,
        curr_str: str,
        step_number: int,
    ) -> Optional[MathErrorClassification]:
        """Detect dropped (missing) or extraneous (extra) terms."""
        prev_symbols = prev.free_symbols
        curr_symbols = curr.free_symbols

        missing = prev_symbols - curr_symbols
        if missing:
            missing_names = ", ".join(s.name for s in missing)
            return MathErrorClassification(
                error_type="Missing Term",
                step_number=step_number,
                student_expression=curr_str,
                expected_expression=self._generate_correct_transition(prev),
                explanation=f"Variable term(s) '{missing_names}' disappeared without being eliminated or solved.",
                confidence=0.92,
                rule_name="Conservation of Algebraic Terms",
                tip="Check each term as you copy expressions from one line to the next.",
            )

        extra = curr_symbols - prev_symbols
        if extra:
            extra_names = ", ".join(s.name for s in extra)
            return MathErrorClassification(
                error_type="Extra Term",
                step_number=step_number,
                student_expression=curr_str,
                expected_expression=self._generate_correct_transition(prev),
                explanation=f"New unexpected variable(s) '{extra_names}' introduced.",
                confidence=0.92,
                rule_name="Conservation of Algebraic Terms",
                tip="Make sure not to introduce extraneous variables.",
            )

        return None

    def _is_step_valid(self, prev: Any, curr: Any) -> bool:
        """Check if transition from prev to curr is mathematically valid."""
        try:
            if isinstance(prev, sympy.Eq) and isinstance(curr, sympy.Eq):
                diff_prev = sympy.simplify(prev.lhs - prev.rhs)
                diff_curr = sympy.simplify(curr.lhs - curr.rhs)
                # Check proportional or identical
                if diff_prev == 0 and diff_curr == 0:
                    return True
                ratio = sympy.simplify(diff_curr / diff_prev)
                return ratio.is_number and ratio != 0
            else:
                return sympy.simplify(prev - curr) == 0
        except Exception:
            return False

    def _generate_correct_transition(self, prev: Union[sympy.Eq, sympy.Expr]) -> str:
        """Generate the expected correct transformation for a previous expression."""
        if isinstance(prev, sympy.Eq):
            var = list(prev.free_symbols)[0] if prev.free_symbols else sympy.Symbol("x")
            diff = sympy.simplify(prev.lhs - prev.rhs)
            poly = diff.as_poly(var)
            if poly and poly.degree() == 1:
                coeff = poly.coeff_monomial(var)
                const = poly.coeff_monomial(1)
                return sympy.latex(sympy.Eq(coeff * var, -const))
            sols = sympy.solve(prev, var)
            if sols:
                return sympy.latex(sympy.Eq(var, sols[0]))
        return sympy.latex(sympy.simplify(prev))
