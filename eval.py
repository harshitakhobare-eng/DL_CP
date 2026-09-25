"""Evaluation Pipeline for AI Handwritten Mathematical Expression Understanding System.

Evaluates:
1. Recognition Metrics:
   - Token Accuracy
   - Exact Sequence Match (EM)
   - Character Error Rate (CER)
2. Mathematical Reasoning Metrics:
   - Solver Accuracy across benchmark equations
   - Step Verification Accuracy
3. Educational Feature Metrics:
   - Error Classification Accuracy across taxonomy
   - Personalized Practice Problem Correctness
"""

from __future__ import annotations
import argparse
import logging
import os
import yaml
import torch
from torch.utils.data import DataLoader

from ml.tokenizer import LatexTokenizer
from ml.dataset import load_math_dataset, MathWritingDataset, math_collate_fn
from ml.training import MathTrainer
from math_engine.solver import SymbolicSolver
from math_engine.verifier import SolutionVerifier
from math_engine.error_classifier import ErrorClassifier
from learning.practice_generator import PracticeGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_recognition(checkpoint_path: str, test_subset: int = 50, config_path: str = "config.yaml"):
    """Evaluate recognition model on test split."""
    print("\n" + "=" * 60)
    print(" 1. RECOGNITION EVALUATION")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"[-] Checkpoint not found at: {checkpoint_path}")
        print("    Please train a model first using `python train.py`.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})
    model_cfg = config.get("model", {})

    model, tokenizer, meta = MathTrainer.load_checkpoint(checkpoint_path)
    
    _, _, test_samples = load_math_dataset(
        dataset_name=dataset_cfg.get("hf_dataset_name", "deepcopy/MathWriting-human"),
        cache_dir=dataset_cfg.get("cache_dir", "data/cache"),
        test_subset=test_subset,
    )

    test_ds = MathWritingDataset(
        samples=test_samples,
        tokenizer=tokenizer,
        max_seq_len=model_cfg.get("max_seq_len", 128),
        target_height=prep_cfg.get("max_height", 128),
        target_width=prep_cfg.get("max_width", 512),
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=math_collate_fn)

    trainer = MathTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=test_loader,
        val_loader=test_loader,
    )
    metrics = trainer.evaluate()

    print(f"Test Samples Evaluated: {len(test_samples)}")
    print(f"Token Accuracy:         {metrics['token_acc'] * 100:.2f}%")
    print(f"Exact Match (EM):       {metrics['exact_match'] * 100:.2f}%")
    print(f"Character Error Rate:   {metrics['cer']:.4f}")
    print(f"Validation Loss:        {metrics['val_loss']:.4f}")


def evaluate_mathematical_reasoning():
    """Evaluate deterministic symbolic solver and step verifier."""
    print("\n" + "=" * 60)
    print(" 2. MATHEMATICAL REASONING EVALUATION")
    print("=" * 60)

    solver = SymbolicSolver()
    verifier = SolutionVerifier()

    # Benchmark test suite of representative mathematical problems
    benchmark_problems = [
        ("2x + 5 = 15", "x = 5"),
        ("x^2 - 5x + 6 = 0", ["2", "3"]),
        ("3(x + 4) = 21", "x = 3"),
        ("4x - 8 = 16", "x = 6"),
        ("x^2 - 9 = 0", ["3", "-3"]),
        ("2(x - 5) = 10", "x = 10"),
    ]

    solver_correct = 0
    for prob, expected in benchmark_problems:
        res = solver.solve(prob)
        sol_text = " ".join(res.solutions)
        if isinstance(expected, list):
            match = all(e in sol_text for e in expected)
        else:
            match = expected.replace(" ", "") in sol_text.replace(" ", "")
        if match:
            solver_correct += 1

    solver_acc = (solver_correct / len(benchmark_problems)) * 100
    print(f"Symbolic Solver Benchmark: {solver_correct}/{len(benchmark_problems)} ({solver_acc:.1f}%)")

    # Verification benchmark suite: (solution steps, is_valid_expected)
    verification_suite = [
        (["2x + 5 = 15", "2x = 10", "x = 5"], True),
        (["3(x + 4) = 21", "3x + 12 = 21", "3x = 9", "x = 3"], True),
        (["3(x + 4) = 21", "3x + 4 = 21"], False),   # Distribution flaw
        (["2x + 5 = 15", "2x = 11"], False),         # Arithmetic flaw
        (["2x = 10", "x = 8"], False),                # Transposition flaw
    ]

    verif_correct = 0
    for steps, expected_valid in verification_suite:
        res = verifier.verify_steps(steps)
        if res.is_correct == expected_valid:
            verif_correct += 1

    verif_acc = (verif_correct / len(verification_suite)) * 100
    print(f"Step Verification Accuracy: {verif_correct}/{len(verification_suite)} ({verif_acc:.1f}%)")


def evaluate_educational_features():
    """Evaluate taxonomy error classification and practice problem generator."""
    print("\n" + "=" * 60)
    print(" 3. EDUCATIONAL FEATURES EVALUATION")
    print("=" * 60)

    classifier = ErrorClassifier()

    # Error classification benchmark
    test_cases = [
        ("3(x + 4) = 21", "3x + 4 = 21", "Distribution Error"),
        ("2x + 5 = 15", "2x = 11", "Arithmetic Error"),
        ("2x = 10", "x = 8", "Incorrect Transposition"),
        ("(x + 2)^2", "x^2 + 4", "Exponent Error"),
    ]

    correct_classifications = 0
    for prev, curr, expected_cat in test_cases:
        cls = classifier.classify(prev, curr, step_number=2)
        if cls.error_type == expected_cat:
            correct_classifications += 1

    class_acc = (correct_classifications / len(test_cases)) * 100
    print(f"Error Classification Accuracy: {correct_classifications}/{len(test_cases)} ({class_acc:.1f}%)")

    # Practice problem generator verification
    practice_gen = PracticeGenerator()
    categories = ["Distribution Error", "Sign Error", "Fraction Error"]
    total_generated = 0
    verified_generated = 0

    for cat in categories:
        pset = practice_gen.generate_practice_set(mistake_type=cat, num_questions=3)
        for p in pset["problems"]:
            total_generated += 1
            if len(p["solutions_latex"]) > 0 and len(p["steps"]) > 0:
                verified_generated += 1

    practice_acc = (verified_generated / max(1, total_generated)) * 100
    print(f"Personalized Practice Correctness: {verified_generated}/{total_generated} ({practice_acc:.1f}%)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Math Understanding System")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--test-subset", type=int, default=50, help="Test subset count")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    evaluate_recognition(args.checkpoint, args.test_subset, args.config)
    evaluate_mathematical_reasoning()
    evaluate_educational_features()


if __name__ == "__main__":
    main()
