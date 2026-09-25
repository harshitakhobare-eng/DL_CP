# AI-Based Handwritten Mathematical Expression Understanding & Personalized Learning System

An intelligent, college-level handwritten mathematics tutor and reasoning system. The system recognizes handwritten mathematical expressions and multi-step derivations, preserves 2D mathematical structure via an Abstract Syntax Tree (AST), solves problems deterministically using symbolic algebra, verifies student solutions step-by-step, classifies errors against a structured taxonomy, and generates personalized practice problems based on recurring mistakes.

---

## Key Features

1. **Vision-to-LaTeX Recognition Pipeline**:
   - CNN Visual Encoder + 2D Positional Embeddings + Transformer Decoder.
   - Grayscale conversion, stroke ink detection, whitespace cropping, and aspect-ratio-preserving padding.
   - Confidence estimation and low-confidence token flagging (distinguishing visual ambiguity from math mistakes).
2. **2D Mathematical Structure Preservation**:
   - Converts recognized LaTeX into a rich Abstract Syntax Tree (AST) preserving fractions, exponents, radicals, parentheses, and operations.
   - Interactive Mermaid tree diagram visualization and hierarchical serialization.
3. **Deterministic Symbolic Mathematics Engine**:
   - Powered by SymPy (no LLM approximations or fake outputs).
   - Solves linear, quadratic, and rational equations, inequalities, polynomial expansions, factoring, differentiation, and integration.
4. **Step-by-Step Pedagogical Derivations**:
   - Produces transparent intermediate transformations with operation descriptions, before/after states, and explanations.
5. **Multi-Step Handwritten Solution Verification**:
   - Segments multi-line handwritten student work into individual step lines.
   - Compares consecutive steps for symbolic mathematical equivalence.
   - Pinpoints the exact first incorrect step in the student's solution.
6. **13-Class Mathematical Error Taxonomy**:
   - Classifies errors: *Sign Error, Arithmetic Error, Distribution Error, Fraction Error, Exponent Error, Incorrect Transposition, Incorrect Cancellation, Missing Term, Extra Term, Incorrect Substitution, Algebraic Manipulation Error, Recognition Error, Unknown/Other*.
7. **Personalized Learning Intelligence Layer**:
   - **Mistake History**: Persistent SQLite database storing full step histories, confidences, and categories.
   - **Explain My Mistake**: 6-part pedagogical breakdown (What was written, What was expected, Why incorrect, Rule involved, Corrected step, Avoidance tip).
   - **Personalized Practice Generator**: Analyzes recurring weakness patterns to synthesize targeted problems—each accompanied by a deterministically verified solution.
   - **Progress Dashboard**: Real-time stats on accuracy, recent accuracy, streak, topic mastery, and mistake category charts.
8. **Modern Educational Web Interface**:
   - Clean, responsive 6-tab interface with KaTeX math rendering, Mermaid AST visualizer, and Chart.js analytics.

---

## Project Structure

```
MathSolverr/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── server.py              # FastAPI REST endpoints
│   └── ui/
│       ├── index.html             # Responsive educational interface
│       ├── style.css              # Custom styling
│       └── app.js                 # Frontend application logic
│
├── ml/
│   ├── __init__.py
│   ├── dataset.py                 # Hugging Face MathWriting-human loader & synthetic generator
│   ├── preprocessing.py           # Whitespace crop, aspect-ratio padding, line segmentation
│   ├── tokenizer.py               # Mathematical LaTeX tokenizer
│   ├── model.py                   # CNN Encoder + Transformer Decoder architecture
│   ├── training.py                # PyTorch training & validation loop
│   └── inference.py               # Lightweight inference recognizer
│
├── math_engine/
│   ├── __init__.py
│   ├── ast.py                     # Mathematical AST (nodes, ASCII tree, Mermaid)
│   ├── latex_parser.py            # LaTeX to SymPy & AST converter
│   ├── solver.py                  # Deterministic symbolic solver
│   ├── step_generator.py          # Step-by-step intermediate derivation engine
│   ├── verifier.py                # Multi-step solution verification engine
│   └── error_classifier.py        # 13-category error taxonomy classifier
│
├── learning/
│   ├── __init__.py
│   ├── mistake_history.py         # SQLite persistence layer for attempts & mistakes
│   ├── explain_mistake.py         # 6-point pedagogical mistake explainer
│   ├── practice_generator.py      # Targeted practice problem generator
│   └── progress.py                # Progress analytics & dashboard aggregator
│
├── data/                          # SQLite databases and cached datasets
├── checkpoints/                   # Saved model weights (.pt)
├── tests/                         # Automated test suite (pytest)
│   ├── test_preprocessing.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_latex_parser.py
│   ├── test_ast.py
│   ├── test_solver.py
│   ├── test_step_generator.py
│   ├── test_verifier.py
│   ├── test_learning.py
│   └── test_api.py
│
├── config.yaml                    # Central configuration
├── requirements.txt               # Dependencies
├── train.py                       # CLI training entry point
├── eval.py                        # Research evaluation pipeline
├── run.py                         # Application runner
├── ARCHITECTURE.md                # System architecture documentation
└── README.md                      # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Optional: CUDA GPU (automatically falls back to CPU if unavailable)

### Setup
```bash
# Clone or navigate to the project directory
cd d:/MathSolverr

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

The project uses the Hugging Face dataset:
```
deepcopy/MathWriting-human
```
Dataset loading is automatic using Hugging Face's `datasets` library. No manual downloads or Google Drive are required.

To customize subsets or cache directory, adjust `config.yaml`:
```yaml
dataset:
  hf_dataset_name: "deepcopy/MathWriting-human"
  cache_dir: "data/cache"
  train_subset: 5000       # Set integer for dev; null for full dataset
  val_subset: 500
  test_subset: 500
```
*Note: If network access to Hugging Face is restricted, the system automatically falls back to an offline synthetic mathematical benchmark generator to maintain complete offline reproducibility.*

---

## Model Training

To train the Vision-to-LaTeX recognition model:
```bash
python train.py --epochs 5 --batch-size 16 --train-subset 200
```

Arguments:
- `--epochs`: Number of training epochs (default: from config)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.0005)
- `--train-subset`: Number of training samples to load
- `--val-subset`: Number of validation samples to load
- `--checkpoint-dir`: Directory where checkpoints are saved (default: `checkpoints/`)

---

## Research Evaluation

To run the comprehensive research evaluation benchmark across Recognition, Mathematical Reasoning, and Educational Features:
```bash
python eval.py --checkpoint checkpoints/best_model.pt
```

Measures:
- **Recognition**: Token Accuracy, Exact Sequence Match (EM), Character Error Rate (CER).
- **Mathematical Reasoning**: Solver accuracy on benchmark equations, step verification accuracy.
- **Educational Features**: Error classification accuracy across taxonomy, personalized practice correctness.

---

## Application Launch

To start the web application and REST API server:
```bash
python run.py
```
Or directly via Uvicorn:
```bash
uvicorn app.api.server:app --host 127.0.0.1 --port 8000 --reload
```

Then open your browser to:
```
http://127.0.0.1:8000
```

---

## Running the Automated Test Suite

Run the full automated test suite:
```bash
pytest tests/ -v
```

All 46+ tests across tokenization, preprocessing, model architecture, AST generation, symbolic solving, step derivation, solution verification, error taxonomy, database persistence, and API endpoints run automatically.

---

## System Limitations & Future Work

- **Handwritten Line Segmentation**: Line segmentation currently uses horizontal projection profile analysis. Extremely skewed or overlapping handwritten multi-line work may benefit from 2D stroke grouping or connected-component graph analysis.
- **Complex Multi-Variable Systems**: Single and two-variable linear and quadratic equations are supported natively; higher-order multi-variable non-linear systems require additional algebraic step heuristics.
- **Recognition Model Size**: The default CNN-Transformer architecture is optimized for fast local training and CPU inference; scaling to ViT/Swin encoders on full multi-GPU clusters is recommended for production deployments on huge multi-million sample datasets.
#   D L _ C P  
 