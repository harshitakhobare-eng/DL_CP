"""Integration tests for FastAPI REST API endpoints."""

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.server import app
from ml.dataset import generate_synthetic_math_image

client = TestClient(app)


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_api_solve_linear():
    res = client.post("/api/solve", json={"latex": "2x + 5 = 15"})
    assert res.status_code == 200
    data = res.json()
    assert data["solution_type"] == "linear_equation"
    assert "x = 5" in data["solutions"]
    assert len(data["steps"]) >= 2
    assert "ast" in data
    assert "ast_mermaid" in data


def test_api_solve_quadratic():
    res = client.post("/api/solve", json={"latex": "x^2 - 5x + 6 = 0"})
    assert res.status_code == 200
    data = res.json()
    assert data["solution_type"] == "quadratic_equation"
    assert any("2" in s for s in data["solutions"])
    assert any("3" in s for s in data["solutions"])


def test_api_verify_valid_solution():
    steps = ["2x + 5 = 15", "2x = 10", "x = 5"]
    res = client.post(
        "/api/verify",
        data={"steps_json": json.dumps(steps), "topic": "Linear Equations"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_correct"] is True
    assert data["first_incorrect_step"] is None
    assert len(data["step_statuses"]) == 3


def test_api_verify_distribution_flaw():
    steps = ["3(x + 4) = 21", "3x + 4 = 21"]
    res = client.post(
        "/api/verify",
        data={"steps_json": json.dumps(steps), "topic": "Linear Equations"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_correct"] is False
    assert data["first_incorrect_step"] == 2
    assert data["error_classification"] is not None
    assert data["error_classification"]["error_type"] == "Distribution Error"
    assert "explanation" in data
    assert "1_what_student_wrote" in data["explanation"]
    assert "6_avoidance_tip" in data["explanation"]


def test_api_explain():
    req_body = {
        "error_type": "Sign Error",
        "step_number": 2,
        "student_expression": "2x = 20",
        "expected_expression": "2x = 10",
        "explanation": "Sign error during subtraction",
        "rule_name": "Sign Inversion",
        "tip": "Carefully negate transposed terms",
        "confidence": 0.95,
    }
    res = client.post("/api/explain", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert data["error_type"] == "Sign Error"
    assert data["1_what_student_wrote"] == "2x = 20"
    assert data["2_what_was_expected"] == "2x = 10"


def test_api_mistakes_and_dashboard():
    # Record mistake via verify
    client.post(
        "/api/verify",
        data={"steps_json": json.dumps(["2x + 5 = 15", "2x = 11"]), "topic": "Linear Equations"}
    )

    # Get mistakes
    m_res = client.get("/api/mistakes")
    assert m_res.status_code == 200
    mistakes = m_res.json()
    assert len(mistakes) >= 1

    # Get dashboard
    d_res = client.get("/api/dashboard")
    assert d_res.status_code == 200
    dash = d_res.json()
    assert dash["total_attempts"] >= 1
    assert "mistakes_by_category" in dash
    assert "streak_days" in dash


def test_api_practice_generate_and_submit():
    # Generate
    gen_res = client.post(
        "/api/practice/generate",
        json={"mistake_type": "Distribution Error", "difficulty": "easy", "num_questions": 2}
    )
    assert gen_res.status_code == 200
    pdata = gen_res.json()
    assert pdata["targeted_weakness"] == "Distribution Error"
    assert len(pdata["problems"]) == 2

    prob = pdata["problems"][0]
    # Submit answer
    sub_res = client.post(
        "/api/practice/submit",
        json={
            "topic": prob["topic"],
            "problem_latex": prob["problem_latex"],
            "solution_latex": prob["solutions_latex"][0],
            "student_answer": prob["final_answer"],
            "is_correct": True,
            "mistake_type": prob["targeted_mistake_type"]
        }
    )
    assert sub_res.status_code == 200
    assert sub_res.json()["status"] == "recorded"


def test_api_recognize_image():
    # Create test synthetic image
    img = generate_synthetic_math_image("2x + 5 = 15")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    res = client.post(
        "/api/recognize",
        files={"file": ("math_test.png", buffer, "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "latex" in data
    assert "confidence" in data
    assert "preprocessed_image" in data
