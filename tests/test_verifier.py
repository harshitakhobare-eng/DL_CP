"""Tests for Solution Verifier and Error Classification."""

import pytest
from math_engine.verifier import SolutionVerifier
from math_engine.error_classifier import ErrorClassifier


def test_verify_valid_solution():
    verifier = SolutionVerifier()
    steps = [
        "2x + 5 = 15",
        "2x = 10",
        "x = 5"
    ]
    res = verifier.verify_steps(steps)
    assert res.is_correct is True
    assert res.first_incorrect_step is None
    assert len(res.step_statuses) == 3
    assert all(s["is_valid"] for s in res.step_statuses)


def test_verify_distribution_error():
    verifier = SolutionVerifier()
    # Flawed solution: 3(x + 4) = 21 -> 3x + 4 = 21 (only multiplied x, not 4)
    steps = [
        "3(x + 4) = 21",
        "3x + 4 = 21",
        "3x = 17"
    ]
    res = verifier.verify_steps(steps)
    assert res.is_correct is False
    assert res.first_incorrect_step == 2
    assert res.error_classification is not None
    assert res.error_classification.error_type == "Distribution Error"
    assert "3x + 12" in res.error_classification.expected_expression or "12" in res.error_classification.explanation


def test_verify_arithmetic_error():
    verifier = SolutionVerifier()
    # Flawed solution: 2x + 5 = 15 -> 2x = 11 (15 - 5 miscalculated as 11)
    steps = [
        "2x + 5 = 15",
        "2x = 11",
        "x = 5.5"
    ]
    res = verifier.verify_steps(steps)
    assert res.is_correct is False
    assert res.first_incorrect_step == 2
    assert res.error_classification.error_type == "Arithmetic Error"


def test_verify_transposition_error():
    classifier = ErrorClassifier()
    # 2x = 10 -> x = 10 - 2 = 8 instead of dividing by 2
    err = classifier.classify("2x = 10", "x = 8", step_number=2)
    assert err.error_type in ("Incorrect Transposition", "Algebraic Manipulation Error")


def test_recognition_error_flagging():
    classifier = ErrorClassifier()
    # Uncertain symbol reported in low_confidence_tokens
    err = classifier.classify(
        "2x + 5 = 15",
        "2y + 5 = 15",
        step_number=2,
        low_confidence_tokens=[{"token": "y", "confidence": 0.45}]
    )
    assert err.error_type == "Recognition Error"
    assert err.is_recognition_suspect is True
