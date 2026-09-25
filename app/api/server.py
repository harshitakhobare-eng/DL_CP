"""FastAPI REST API Server for Math Understanding & Personalized Learning System.

Exposes endpoints for:
- Recognition & confidence estimation (/api/recognize)
- Symbolic solving & step generation (/api/solve)
- Handwritten solution verification & error taxonomy (/api/verify)
- Pedagogical mistake explanation (/api/explain)
- Persistent mistake history (/api/mistakes)
- Real-time progress analytics (/api/dashboard)
- Personalized practice generation (/api/practice/generate, /api/practice/submit)
"""

from __future__ import annotations
import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import yaml

from ml.inference import MathRecognizer
from math_engine.latex_parser import parse_latex_to_ast, parse_latex_to_sympy, clean_latex
from math_engine.solver import SymbolicSolver
from math_engine.step_generator import StepGenerator
from math_engine.verifier import SolutionVerifier
from math_engine.error_classifier import MathErrorClassification
from learning.mistake_history import MistakeHistoryDB
from learning.explain_mistake import MistakeExplainer
from learning.practice_generator import PracticeGenerator
from learning.progress import ProgressTracker

logger = logging.getLogger(__name__)

# Load config
config = {}
if os.path.exists("config.yaml"):
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
db_path = config.get("storage", {}).get("db_path", "data/learning_system.db")

app = FastAPI(
    title="AI Math Understanding & Personalized Learning API",
    version="1.0.0",
    description="College-level handwritten math recognition, symbolic solver, step verifier, and personalized tutor.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
recognizer = MathRecognizer(checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None)
solver = SymbolicSolver()
step_generator = StepGenerator()
verifier = SolutionVerifier()
mistake_db = MistakeHistoryDB(db_path=db_path)
explainer = MistakeExplainer()
practice_gen = PracticeGenerator(db=mistake_db)
progress_tracker = ProgressTracker(db=mistake_db)


# Request Models
class SolveRequest(BaseModel):
    latex: str
    target_var: Optional[str] = None


class VerifyStepsRequest(BaseModel):
    steps: List[str]
    problem: Optional[str] = None
    topic: Optional[str] = "Linear Equations"


class ExplainRequest(BaseModel):
    error_type: str
    step_number: int
    student_expression: str
    expected_expression: str
    explanation: str
    rule_name: Optional[str] = None
    tip: Optional[str] = None
    confidence: Optional[float] = 0.9


class PracticeGenerateRequest(BaseModel):
    mistake_type: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = "medium"
    num_questions: Optional[int] = 3


class PracticeSubmitRequest(BaseModel):
    topic: str
    problem_latex: str
    solution_latex: str
    student_answer: str
    is_correct: bool
    mistake_type: Optional[str] = None


# Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": request.url.path},
    )


# Endpoints
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(recognizer.device),
        "checkpoint_loaded": os.path.exists(checkpoint_path),
        "db_path": db_path,
    }


@app.post("/api/recognize")
async def recognize_image(file: UploadFile = File(...)):
    """Upload an image of a handwritten mathematical expression for recognition."""
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        image = Image.open(io.BytesIO(contents))
        rec_result = recognizer.recognize(image)
        latex = rec_result["latex"]

        # Parse AST if valid LaTeX
        ast_dict = None
        ast_mermaid = None
        if latex.strip():
            try:
                ast_node = parse_latex_to_ast(latex)
                ast_dict = ast_node.to_dict()
                ast_mermaid = ast_node.to_mermaid()
            except Exception as e:
                logger.warning(f"Could not construct AST for recognized LaTeX '{latex}': {e}")

        return {
            "latex": latex,
            "confidence": rec_result["confidence"],
            "tokens": rec_result["tokens"],
            "token_confidences": rec_result["token_confidences"],
            "low_confidence_tokens": rec_result["low_confidence_tokens"],
            "preprocessed_image": rec_result["preprocessed_image_base64"],
            "ast": ast_dict,
            "ast_mermaid": ast_mermaid,
        }
    except Exception as e:
        logger.error(f"Error during recognition: {e}")
        raise HTTPException(status_code=400, detail=f"Image recognition failed: {str(e)}")


@app.post("/api/solve")
def solve_expression(req: SolveRequest):
    """Solve mathematical equation symbolically and generate intermediate steps."""
    clean = clean_latex(req.latex)
    if not clean:
        raise HTTPException(status_code=400, detail="Expression cannot be empty.")

    try:
        solver_res = solver.solve(clean, target_var=req.target_var)
        steps_objs = step_generator.generate(clean)
        steps_list = [s.to_dict() for s in steps_objs]

        # AST
        try:
            ast_node = parse_latex_to_ast(clean)
            ast_dict = ast_node.to_dict()
            ast_mermaid = ast_node.to_mermaid()
        except Exception:
            ast_dict = None
            ast_mermaid = None

        return {
            "problem_latex": solver_res.problem_latex,
            "solution_type": solver_res.solution_type,
            "solutions": solver_res.solutions,
            "solutions_latex": solver_res.solutions_latex,
            "variable": solver_res.variable,
            "steps": steps_list,
            "ast": ast_dict,
            "ast_mermaid": ast_mermaid,
        }
    except Exception as e:
        logger.error(f"Error solving expression '{req.latex}': {e}")
        raise HTTPException(status_code=400, detail=f"Mathematical solution failed: {str(e)}")


@app.post("/api/verify")
async def verify_solution(
    steps_json: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    topic: Optional[str] = Form("Linear Equations"),
):
    """Verify a student's handwritten multi-step solution."""
    try:
        if file is not None:
            # Multi-line handwritten image uploaded
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded solution file is empty.")
            img = Image.open(io.BytesIO(contents))
            res = verifier.verify_handwritten_image(img, recognizer)
        elif steps_json:
            import json
            steps_list = json.loads(steps_json)
            res = verifier.verify_steps(steps_list)
        else:
            raise HTTPException(status_code=400, detail="Provide either an image or a list of steps.")

        # If an error occurred, produce full pedagogical explanation and log to database
        explanation_dict = None
        if not res.is_correct and res.error_classification:
            explanation_dict = explainer.explain(res.error_classification)

            # Record in mistake history
            mistake_db.record_attempt(
                problem=res.problem or (res.step_statuses[0]["latex"] if res.step_statuses else ""),
                topic=topic or "Algebra",
                student_solution=[s["latex"] for s in res.step_statuses],
                final_correct=False,
                mistake_type=res.error_classification.error_type,
                step_number=res.first_incorrect_step,
                student_step=res.error_classification.student_expression,
                correct_step=res.error_classification.expected_expression,
                explanation=res.error_classification.explanation,
                recognition_confidence=res.error_classification.confidence,
            )
        elif res.is_correct and res.step_statuses:
            # Record successful attempt
            mistake_db.record_attempt(
                problem=res.problem or res.step_statuses[0]["latex"],
                topic=topic or "Algebra",
                student_solution=[s["latex"] for s in res.step_statuses],
                final_correct=True,
                mistake_type="Correct",
            )

        return {
            **res.to_dict(),
            "explanation": explanation_dict,
        }
    except Exception as e:
        logger.error(f"Error during solution verification: {e}")
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(e)}")


@app.post("/api/explain")
def explain_mistake(req: ExplainRequest):
    """Generate the 6-point pedagogical breakdown for a mistake."""
    classification = MathErrorClassification(
        error_type=req.error_type,
        step_number=req.step_number,
        student_expression=req.student_expression,
        expected_expression=req.expected_expression,
        explanation=req.explanation,
        confidence=req.confidence or 0.9,
        rule_name=req.rule_name,
        tip=req.tip,
    )
    return explainer.explain(classification)


@app.get("/api/mistakes")
def get_mistakes(limit: int = 50, topic: Optional[str] = None, mistake_type: Optional[str] = None):
    """Retrieve recorded student mistakes."""
    return mistake_db.get_mistakes(limit=limit, topic=topic, mistake_type=mistake_type)


@app.get("/api/dashboard")
def get_dashboard():
    """Retrieve progress metrics and dashboard visualizations."""
    return progress_tracker.get_dashboard_summary()


@app.post("/api/practice/generate")
def generate_practice(req: PracticeGenerateRequest):
    """Generate personalized practice problems targeting weak areas."""
    return practice_gen.generate_practice_set(
        mistake_type=req.mistake_type,
        topic=req.topic,
        difficulty=req.difficulty or "medium",
        num_questions=req.num_questions or 3,
    )


@app.post("/api/practice/submit")
def submit_practice(req: PracticeSubmitRequest):
    """Record student response to a practice problem."""
    row_id = mistake_db.record_practice_result(
        topic=req.topic,
        problem_latex=req.problem_latex,
        solution_latex=req.solution_latex,
        student_answer=req.student_answer,
        is_correct=req.is_correct,
        mistake_type=req.mistake_type,
    )
    streak = mistake_db.get_learning_streak()
    return {"status": "recorded", "id": row_id, "current_streak": streak}


# Mount Static Frontend Files
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
if os.path.exists(ui_dir):
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(ui_dir, "index.html"))
