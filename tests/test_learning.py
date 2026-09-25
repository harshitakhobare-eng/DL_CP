"""Tests for Mistake History, Explanation, Practice Generator, and Progress."""

import os
import pytest
from learning.mistake_history import MistakeHistoryDB
from learning.explain_mistake import MistakeExplainer
from learning.practice_generator import PracticeGenerator
from learning.progress import ProgressTracker
from math_engine.error_classifier import MathErrorClassification


@pytest.fixture
def test_db(tmp_path):
    db_file = os.path.join(tmp_path, "test_learning.db")
    return MistakeHistoryDB(db_path=db_file)


def test_record_and_retrieve_mistakes(test_db):
    row_id = test_db.record_attempt(
        problem="3(x+4)=21",
        topic="Linear Equations",
        student_solution=["3(x+4)=21", "3x+4=21"],
        final_correct=False,
        mistake_type="Distribution Error",
        step_number=2,
        student_step="3x+4=21",
        correct_step="3x+12=21",
        explanation="Forgot to distribute 3 to 4",
    )
    assert row_id > 0

    mistakes = test_db.get_mistakes()
    assert len(mistakes) == 1
    assert mistakes[0]["mistake_type"] == "Distribution Error"
    assert mistakes[0]["student_step"] == "3x+4=21"


def test_mistake_counts_and_topic_stats(test_db):
    test_db.record_attempt("3(x+4)=21", "Linear Equations", [], False, mistake_type="Distribution Error")
    test_db.record_attempt("2(x+5)=20", "Linear Equations", [], False, mistake_type="Distribution Error")
    test_db.record_attempt("2x-5=15", "Linear Equations", [], False, mistake_type="Sign Error")
    test_db.record_attempt("x+2=5", "Linear Equations", [], True)

    counts = test_db.get_mistake_counts_by_type()
    assert counts["Distribution Error"] == 2
    assert counts["Sign Error"] == 1

    stats = test_db.get_topic_statistics()
    assert len(stats) == 1
    assert stats[0]["total_attempts"] == 4
    assert stats[0]["total_correct"] == 1


def test_explain_my_mistake():
    explainer = MistakeExplainer()
    classification = MathErrorClassification(
        error_type="Distribution Error",
        step_number=2,
        student_expression="3x + 4 = 21",
        expected_expression="3x + 12 = 21",
        explanation="The 3 must multiply both terms inside the parentheses.",
        confidence=0.95,
        rule_name="Distributive Property",
        tip="Distribute multiplier to every term inside bracket.",
    )
    exp = explainer.explain(classification)
    assert "1_what_student_wrote" in exp
    assert "2_what_was_expected" in exp
    assert "3_why_incorrect" in exp
    assert "4_mathematical_rule" in exp
    assert "5_corrected_step" in exp
    assert "6_avoidance_tip" in exp
    assert exp["1_what_student_wrote"] == "3x + 4 = 21"


def test_practice_generator_with_verified_solutions(test_db):
    test_db.record_attempt("3(x+4)=21", "Linear Equations", [], False, mistake_type="Distribution Error")
    generator = PracticeGenerator(db=test_db)
    
    practice_set = generator.generate_practice_set(num_questions=3)
    assert practice_set["targeted_weakness"] == "Distribution Error"
    assert len(practice_set["problems"]) == 3

    # Check that each generated problem has a verified solution
    for p in practice_set["problems"]:
        assert len(p["solutions_latex"]) > 0
        assert len(p["steps"]) > 0
        assert p["final_answer"] != ""


def test_progress_tracker(test_db):
    test_db.record_attempt("3(x+4)=21", "Linear Equations", [], False, mistake_type="Distribution Error")
    test_db.record_attempt("2x=10", "Linear Equations", [], True)

    tracker = ProgressTracker(test_db)
    summary = tracker.get_dashboard_summary()
    assert summary["total_attempts"] == 2
    assert summary["total_correct"] == 1
    assert summary["overall_accuracy"] == 50.0
    assert "Distribution Error" in summary["mistakes_by_category"]
