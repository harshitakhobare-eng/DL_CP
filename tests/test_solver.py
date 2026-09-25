"""Tests for Symbolic Solver and Step Generator."""

import pytest
from math_engine.solver import SymbolicSolver
from math_engine.step_generator import StepGenerator


def test_solver_linear():
    solver = SymbolicSolver()
    res = solver.solve("2x + 5 = 15")
    assert res.solution_type == "linear_equation"
    assert "x = 5" in res.solutions


def test_solver_quadratic():
    solver = SymbolicSolver()
    res = solver.solve("x^2 - 5x + 6 = 0")
    assert res.solution_type == "quadratic_equation"
    # Roots 2 and 3
    sol_str = " ".join(res.solutions)
    assert "2" in sol_str
    assert "3" in sol_str


def test_solver_simplification():
    solver = SymbolicSolver()
    res = solver.solve("2(x + 3)")
    assert res.solution_type == "expression_simplification"
    details = res.details
    assert "2*x + 6" in details.get("expanded", "") or "2*x + 6" in details.get("simplified", "")


def test_step_generator_linear():
    generator = StepGenerator()
    steps = generator.generate("2x + 5 = 15")
    assert len(steps) >= 2
    # Check that each step has all required components
    for step in steps:
        assert step.step_number > 0
        assert step.expression_before != ""
        assert step.expression_after != ""
        assert step.operation != ""
        assert step.explanation != ""


def test_step_generator_distribution():
    generator = StepGenerator()
    steps = generator.generate("3(x + 4) = 21")
    assert len(steps) >= 2
    # At least one step should mention Distributive or Expand
    ops = [s.operation for s in steps]
    assert any("Expand" in op or "Distribut" in op for op in ops)


def test_step_generator_quadratic():
    generator = StepGenerator()
    steps = generator.generate("x^2 - 5x + 6 = 0")
    assert len(steps) >= 2
    ops = [s.operation for s in steps]
    assert any("Discriminant" in op or "Factor" in op or "standard" in op.lower() for op in ops)
