"""Multi-Step Handwritten Mathematical Solution Verification Engine.

Performs:
1. Multi-line step extraction from handwritten solution images
2. Step recognition and LaTeX conversion
3. Deterministic mathematical equivalence checking between consecutive steps
4. Identification of the first incorrect step
5. Deep error classification via ErrorClassifier
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import sympy

from math_engine.latex_parser import parse_latex_to_sympy, clean_latex
from math_engine.error_classifier import ErrorClassifier, MathErrorClassification
from ml.preprocessing import segment_solution_steps

logger = logging.getLogger(__name__)


class VerificationResult:
    """Encapsulates the end-to-end verification outcome of a student solution."""

    def __init__(
        self,
        is_correct: bool,
        total_steps: int,
        step_statuses: List[Dict[str, Any]],
        first_incorrect_step: Optional[int] = None,
        error_classification: Optional[MathErrorClassification] = None,
        problem: Optional[str] = None,
        correct_solution: Optional[str] = None,
    ):
        self.is_correct = is_correct
        self.total_steps = total_steps
        self.step_statuses = step_statuses
        self.first_incorrect_step = first_incorrect_step
        self.error_classification = error_classification
        self.problem = problem
        self.correct_solution = correct_solution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_correct": self.is_correct,
            "total_steps": self.total_steps,
            "first_incorrect_step": self.first_incorrect_step,
            "step_statuses": self.step_statuses,
            "error_classification": self.error_classification.to_dict() if self.error_classification else None,
            "problem": self.problem,
            "correct_solution": self.correct_solution,
        }


class SolutionVerifier:
    """Verifies student mathematical solutions step-by-step."""

    def __init__(self):
        self.error_classifier = ErrorClassifier()

    def verify_steps(
        self,
        steps: List[str],
        confidences: Optional[List[float]] = None,
        low_confidence_tokens: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> VerificationResult:
        """Verify an ordered list of LaTeX mathematical steps.
        
        Args:
            steps: List of mathematical step LaTeX strings
            confidences: Optional confidence per step
            low_confidence_tokens: Optional list of uncertain tokens per step
        """
        if not steps:
            return VerificationResult(
                is_correct=False,
                total_steps=0,
                step_statuses=[],
                first_incorrect_step=None,
            )

        step_statuses: List[Dict[str, Any]] = []
        first_error: Optional[MathErrorClassification] = None
        first_error_step: Optional[int] = None

        # Clean all steps
        clean_steps = [clean_latex(s) for s in steps if clean_latex(s)]
        if not clean_steps:
            return VerificationResult(is_correct=False, total_steps=0, step_statuses=[])

        # Step 1 is always the problem / premise
        c0 = confidences[0] if confidences and len(confidences) > 0 else 1.0
        step_statuses.append({
            "step_number": 1,
            "latex": clean_steps[0],
            "is_valid": True,
            "confidence": round(c0, 4),
            "status_message": "Problem statement / initial premise",
        })

        # Verify each transition from Step k-1 to Step k
        for k in range(1, len(clean_steps)):
            prev_step = clean_steps[k - 1]
            curr_step = clean_steps[k]
            step_conf = confidences[k] if confidences and k < len(confidences) else 1.0
            low_tokens = low_confidence_tokens[k] if low_confidence_tokens and k < len(low_confidence_tokens) else None

            is_valid = self._check_step_equivalence(prev_step, curr_step)

            if is_valid:
                step_statuses.append({
                    "step_number": k + 1,
                    "latex": curr_step,
                    "is_valid": True,
                    "confidence": round(step_conf, 4),
                    "status_message": "Mathematically valid deduction",
                })
            else:
                # Step is invalid!
                step_statuses.append({
                    "step_number": k + 1,
                    "latex": curr_step,
                    "is_valid": False,
                    "confidence": round(step_conf, 4),
                    "status_message": "Mathematical error detected at this step",
                })

                # If this is the FIRST incorrect step, classify the error
                if first_error is None:
                    first_error_step = k + 1
                    first_error = self.error_classifier.classify(
                        prev_step_str=prev_step,
                        curr_step_str=curr_step,
                        step_number=k + 1,
                        step_confidences=[step_conf],
                        low_confidence_tokens=low_tokens,
                    )

        is_overall_correct = (first_error is None)

        return VerificationResult(
            is_correct=is_overall_correct,
            total_steps=len(clean_steps),
            step_statuses=step_statuses,
            first_incorrect_step=first_error_step,
            error_classification=first_error,
            problem=clean_steps[0],
            correct_solution=clean_steps[-1] if is_overall_correct else (first_error.expected_expression if first_error else None),
        )

    def verify_handwritten_image(
        self,
        image_input: Union[Image.Image, str, bytes],
        recognizer: Any,
    ) -> VerificationResult:
        """End-to-end verification of a multi-line handwritten solution image."""
        # 1. Segment image into line/step images
        step_crops = segment_solution_steps(image_input)
        logger.info(f"Segmented handwritten solution into {len(step_crops)} step lines.")

        recognized_steps: List[str] = []
        confidences: List[float] = []
        low_confidence_tokens_list: List[List[Dict[str, Any]]] = []

        # 2. Recognize each step image
        for crop in step_crops:
            rec_result = recognizer.recognize(crop)
            latex = rec_result.get("latex", "")
            if latex.strip():
                recognized_steps.append(latex)
                confidences.append(rec_result.get("confidence", 1.0))
                low_confidence_tokens_list.append(rec_result.get("low_confidence_tokens", []))

        # 3. Verify step sequence
        return self.verify_steps(
            steps=recognized_steps,
            confidences=confidences,
            low_confidence_tokens=low_confidence_tokens_list,
        )

    def _check_step_equivalence(self, prev_str: str, curr_str: str) -> bool:
        """Check symbolic equivalence between consecutive steps."""
        try:
            prev_obj = parse_latex_to_sympy(prev_str)
            curr_obj = parse_latex_to_sympy(curr_str)

            # Both equations: Eq(lhs1, rhs1) and Eq(lhs2, rhs2)
            if isinstance(prev_obj, sympy.Eq) and isinstance(curr_obj, sympy.Eq):
                # Standard form difference: diff = lhs - rhs
                diff_prev = sympy.simplify(prev_obj.lhs - prev_obj.rhs)
                diff_curr = sympy.simplify(curr_obj.lhs - curr_obj.rhs)

                if diff_prev == 0 and diff_curr == 0:
                    return True

                # Check if proportional (multiplying or dividing both sides by constant)
                try:
                    ratio = sympy.simplify(diff_curr / diff_prev)
                    if ratio.is_number and ratio != 0:
                        return True
                except Exception:
                    pass

                # Check if solutions match for equations in single variable
                var_p = list(prev_obj.free_symbols)
                var_c = list(curr_obj.free_symbols)
                if len(var_p) == 1 and len(var_c) == 1 and var_p[0] == var_c[0]:
                    sols_p = set(sympy.solve(prev_obj, var_p[0]))
                    sols_c = set(sympy.solve(curr_obj, var_c[0]))
                    if sols_p and sols_p == sols_c:
                        return True

                return False

            # Both expressions: simplify(prev - curr) == 0
            if not isinstance(prev_obj, sympy.Eq) and not isinstance(curr_obj, sympy.Eq):
                diff = sympy.simplify(prev_obj - curr_obj)
                return diff == 0

            return False
        except Exception as e:
            logger.warning(f"Error checking step equivalence: {e}")
            return False
