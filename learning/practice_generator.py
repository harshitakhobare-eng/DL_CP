"""Personalized Practice Generator targeting student weaknesses.

Analyzes mistake history to detect weak areas, generates targeted problems,
and computes verified deterministic step-by-step solutions for every problem.
"""

from __future__ import annotations
import logging
import random
from typing import Any, Dict, List, Optional
import sympy

from math_engine.solver import SymbolicSolver
from math_engine.step_generator import StepGenerator
from learning.mistake_history import MistakeHistoryDB

logger = logging.getLogger(__name__)


class PracticeProblem:
    """Represents a generated practice problem with verified solutions."""

    def __init__(
        self,
        problem_id: str,
        problem_latex: str,
        topic: str,
        targeted_mistake_type: str,
        difficulty: str,
        solutions_latex: List[str],
        final_answer: str,
        steps: List[Dict[str, Any]],
    ):
        self.problem_id = problem_id
        self.problem_latex = problem_latex
        self.topic = topic
        self.targeted_mistake_type = targeted_mistake_type
        self.difficulty = difficulty
        self.solutions_latex = solutions_latex
        self.final_answer = final_answer
        self.steps = steps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "problem_latex": self.problem_latex,
            "topic": self.topic,
            "targeted_mistake_type": self.targeted_mistake_type,
            "difficulty": self.difficulty,
            "solutions_latex": self.solutions_latex,
            "final_answer": self.final_answer,
            "steps": self.steps,
        }


class PracticeGenerator:
    """Synthesizes targeted mathematical practice problems."""

    def __init__(self, db: Optional[MistakeHistoryDB] = None):
        self.db = db
        self.solver = SymbolicSolver()
        self.step_generator = StepGenerator()

    def generate_practice_set(
        self,
        mistake_type: Optional[str] = None,
        topic: Optional[str] = None,
        difficulty: str = "medium",
        num_questions: int = 3,
    ) -> Dict[str, Any]:
        """Generate personalized practice set targeting identified weakness."""
        # 1. Identify targeted weakness
        target_weakness = mistake_type
        if not target_weakness and self.db:
            counts = self.db.get_mistake_counts_by_type()
            if counts:
                target_weakness = max(counts, key=counts.get)

        if not target_weakness:
            target_weakness = "Distribution Error"

        logger.info(f"Generating {num_questions} practice problems for weakness '{target_weakness}' ({difficulty}).")

        problems: List[PracticeProblem] = []
        for i in range(num_questions):
            prob = self._generate_single_problem(target_weakness, difficulty, f"q_{i+1}")
            problems.append(prob)

        return {
            "targeted_weakness": target_weakness,
            "difficulty": difficulty,
            "num_questions": len(problems),
            "problems": [p.to_dict() for p in problems],
        }

    def _generate_single_problem(
        self,
        mistake_type: str,
        difficulty: str,
        qid: str,
    ) -> PracticeProblem:
        """Synthesize a single problem with deterministic verified steps."""
        x = sympy.Symbol("x")

        # -------------------------------------------------------------
        # DISTRIBUTION ERROR GENERATOR
        # -------------------------------------------------------------
        if mistake_type == "Distribution Error":
            topic = "Linear Equations with Brackets"
            if difficulty == "easy":
                a = random.choice([2, 3, 4, 5])
                b = random.choice([1, 2, 3, 4, 5])
                sol_x = random.choice([2, 3, 4, 5])
                c = a * (sol_x + b)
                latex_expr = f"{a}(x + {b}) = {c}"
            elif difficulty == "hard":
                a = random.choice([-2, -3, -4, -5])
                b = random.choice([2, 3, 4, 6])
                sol_x = random.choice([-3, -2, 2, 3, 4])
                c = a * (sol_x + b)
                latex_expr = f"{a}(x + {b}) = {c}"
            else:  # medium
                a = random.choice([2, 3, 4, 5])
                b = random.choice([2, 3, 4, 5])
                sol_x = random.choice([3, 4, 5, 6])
                c = a * (sol_x - b)
                latex_expr = f"{a}(x - {b}) = {c}"

        # -------------------------------------------------------------
        # SIGN ERROR GENERATOR
        # -------------------------------------------------------------
        elif mistake_type == "Sign Error":
            topic = "Linear Equations with Negative Terms"
            if difficulty == "easy":
                b = random.choice([3, 4, 5, 7, 9])
                sol_x = random.choice([2, 3, 4, 5])
                c = 2 * sol_x - b
                latex_expr = f"2x - {b} = {c}"
            else:
                a1 = random.choice([3, 4, 5])
                a2 = random.choice([1, 2])
                b = random.choice([4, 6, 8])
                sol_x = random.choice([3, 4, 5])
                d = (a1 - a2) * sol_x - b
                latex_expr = f"{a1}x - {b} = {a2}x + {d}"

        # -------------------------------------------------------------
        # FRACTION ERROR GENERATOR
        # -------------------------------------------------------------
        elif mistake_type == "Fraction Error":
            topic = "Fractions and Rational Equations"
            den = random.choice([2, 3, 4, 5])
            sol_x = random.choice([2, 3, 4, 5])
            offset = random.choice([1, 3, 5])
            num_val = sol_x + offset
            c = num_val * den  # ensure clean
            latex_expr = f"\\frac{{x + {offset}}}{{{den}}} = {sol_x}"

        # -------------------------------------------------------------
        # EXPONENT / QUADRATICS GENERATOR
        # -------------------------------------------------------------
        elif mistake_type in ("Exponent Error", "Quadratics"):
            topic = "Quadratic Equations"
            r1 = random.choice([1, 2, 3, 4])
            r2 = random.choice([2, 3, 5])
            b_val = -(r1 + r2)
            c_val = r1 * r2
            sign_b = "+" if b_val >= 0 else "-"
            abs_b = abs(b_val)
            latex_expr = f"x^2 {sign_b} {abs_b}x + {c_val} = 0"

        # -------------------------------------------------------------
        # DEFAULT / ARITHMETIC / TRANSPOSITION GENERATOR
        # -------------------------------------------------------------
        else:
            topic = "Linear Equations"
            a = random.choice([2, 3, 4, 6])
            sol_x = random.choice([2, 3, 4, 5])
            b = random.choice([3, 5, 7, 11])
            c = a * sol_x + b
            latex_expr = f"{a}x + {b} = {c}"

        # Deterministically compute solution and steps
        solver_res = self.solver.solve(latex_expr)
        steps_objs = self.step_generator.generate(latex_expr)
        steps_dicts = [s.to_dict() for s in steps_objs]

        # Extract final answer
        final_ans = solver_res.solutions[0] if solver_res.solutions else "N/A"

        return PracticeProblem(
            problem_id=qid,
            problem_latex=latex_expr,
            topic=topic,
            targeted_mistake_type=mistake_type,
            difficulty=difficulty,
            solutions_latex=solver_res.solutions_latex,
            final_answer=final_ans,
            steps=steps_dicts,
        )
