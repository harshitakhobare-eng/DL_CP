<<<<<<< HEAD

# 🧮 MathTutor AI: Handwritten Math Solver & Personalized Learning System

An end-to-end AI/ML system that recognizes handwritten math expressions, solves them step-by-step deterministically, audits student solutions, and generates personalized practice problems based on diagnosed weaknesses.
=======
# 🧮 MathTutor AI: Handwritten Mathematical Expression Understanding & Personalized Learning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![SymPy](https://img.shields.io/badge/SymPy-Symbolic%20Engine-brightgreen.svg)](https://www.sympy.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-46%20passed-success.svg)](./tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A research-oriented AI/ML system that transforms handwritten mathematical solutions into deep pedagogical insights.

Unlike conventional solvers that produce black-box final answers or hallucinate derivations via generic LLMs, **MathTutor AI** combines a **deep neural vision-to-LaTeX sequence model** with a **100% deterministic symbolic algebra engine (SymPy)**. It parses expressions into Abstract Syntax Trees (AST), verifies multi-step student proofs, classifies algebraic errors using a 13-category pedagogical taxonomy, and dynamically generates targeted practice problems based on individual student mistake history.
>>>>>>> 19aa8a5 (Fix handwriting recognition training)

---

## 🌟 Key Features

<<<<<<< HEAD
* **Vision-to-LaTeX Recognition**: CNN Visual Encoder + 2D Positional Encoding + Transformer Decoder trained on the real-world `deepcopy/MathWriting-human` dataset.
* **Deterministic Symbolic Solver**: Uses SymPy to solve linear, quadratic, polynomial, rational equations, calculus (derivatives/integrals), and inequalities — with **zero hallucinations**.
* **Visual Abstract Syntax Tree (AST)**: Real-time interactive syntax tree visualization powered by Mermaid.js.
* **Multi-Step Solution Verifier**: Audits a student's handwritten working step-by-step to flag exactly where an algebraic error occurs.
* **13-Category Error Taxonomy**: Automatically detects and classifies errors (Sign Errors, Distribution Errors, Transposition Flaws, Arithmetic Mistakes, etc.).
* **Pedagogical Explanations**: 6-point breakdown explaining what was written, what was expected, why it was wrong, and the mathematical rule to remember.
* **Targeted Practice Generator**: Synthesizes custom practice questions targeted directly at your most frequent errors.
* **Student Dashboard**: Tracks accuracy trends, daily learning streaks, and mistake distribution over time.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Handwritten Image] --> B[Image Preprocessing]
    B --> C[CNN Visual Encoder]
    C --> D[2D Positional Encoding]
    D --> E[Transformer Decoder]
    E --> F[Recognized LaTeX]

    F --> G[LaTeX Parser & AST]
    G --> H[SymPy Symbolic Solver]
    H --> I[Step-by-Step Derivation]

    J[Student Solution Steps] --> K[Step Verifier]
    K --> L{Mathematically Sound?}
    L -- Yes --> M[Verified Correct]
    L -- No --> N[13-Category Error Classifier]
    N --> O[Pedagogical Explainer]
    O --> P[Targeted Practice Generator]

    P --> Q[(SQLite History DB)]
    Q --> R[Student Progress Dashboard]
=======
1. **Neural Vision-to-LaTeX Recognition**
   - Custom 4-stage convolutional feature extractor (64 to 256 channels).
   - 2D Sinusoidal positional embeddings encoding spatial 2D layout (fractions, powers, subscripts).
   - Multi-head autoregressive Transformer decoder with teacher forcing.
   - Trained on the real-world `deepcopy/MathWriting-human` Hugging Face dataset.

2. **Deterministic Symbolic Math Engine (No LLM Hallucinations)**
   - Full bi-directional LaTeX ↔ SymPy parsing pipeline.
   - Solves linear, quadratic, polynomial, rational equations, inequalities, limits, derivatives, and integrals.
   - Live generation of mathematical **Abstract Syntax Trees (AST)** rendered in real-time via Mermaid.js diagrams.

3. **Symbolic Multi-Step Solution Verifier**
   - Audits student handwritten steps line-by-line.
   - Symbolically checks mathematical equivalence between consecutive steps ($\text{Step}_k \iff \text{Step}_{k+1}$).
   - Immediately identifies the exact transition where an algebraic fallacy occurs.

4. **13-Category Pedagogical Error Taxonomy**
   - Automated rule-based error classification (Sign Errors, Distribution Errors, Transposition Flaws, etc.).
   - Prioritized analysis prevents cascading misclassifications.

5. **Personalized Learning & Weakness Remediation**
   - 6-point pedagogical explanations: *What was written*, *Expected step*, *Why it is invalid*, *Mathematical law*, *Corrected step*, and *Avoidance tip*.
   - Weakness-targeted synthetic practice problem generator.
   - Local SQLite persistence tracking accuracy trends, learning streaks, and topic mastery.

---

## 🏗 System Architecture

```mermaid
graph TD
    A["Handwritten Image (PNG/JPG)"] --> B["CV Preprocessing (Crop, Pad, Normalize)"]
    B --> C["Visual CNN Encoder (4-Stage: 8x32x256)"]
    C --> D["2D Sinusoidal Positional Encoding"]
    D --> E["Transformer Decoder"]
    E --> F["Recognized LaTeX String"]

    F --> G["LaTeX Parser & AST Builder"]
    G --> H["SymPy Symbolic Engine"]
    H --> I["Deterministic Step Derivation"]

    J["Student Solution Steps"] --> K["Step-by-Step Verifier"]
    K --> L{"Valid Transition?"}
    L -- Valid --> M["Success Verification Audit"]
    L -- Invalid --> N["13-Category Error Classifier"]
    N --> O["6-Point Pedagogical Explainer"]
    O --> P["Targeted Practice Generator"]

    P --> Q[("SQLite Learning DB")]
    Q --> R["Progress Dashboard"]
```

---

## 🔍 Error Classification Taxonomy

The system systematically categorizes student mistakes into 13 formal categories:

| Priority | Category | Typical Example | Diagnostic Mechanism |
| :---: | :--- | :--- | :--- |
| **1** | **Recognition Error** | `+` misidentified as `t` | Low token confidence or OCR anomaly |
| **2** | **Distribution Error** | $3(x + 4) = 21 \implies 3x + 4 = 21$ | Parentheses removed without distributing factor |
| **3** | **Sign Error** | $2x - 5 = 15 \implies 2x = 10$ | Sign flipped incorrectly during simplification |
| **4** | **Incorrect Transposition** | $2x = 10 \implies x = 10 - 2 = 8$ | Subtracted coefficient instead of dividing |
| **5** | **Arithmetic Error** | $2x = 15 - 5 \implies 2x = 11$ | Pure numerical computation mistake |
| **6** | **Exponent Error** | $x^2 \cdot x^3 = x^6$ | Multiplied powers instead of adding exponents |
| **7** | **Fraction Error** | $\frac{a+b}{c} = \frac{a}{c} + b$ | Denominator distribution / cancellation mistake |
| **8** | **Term Count Error** | Dropping terms across transitions | Mismatched additive terms count |
| **9** | **Factoring Error** | $x^2 - 5x + 6 = (x - 1)(x - 6)$ | Incorrect roots in factored binomials |
| **10** | **Division by Zero** | Undefined denominator substitution | Evaluation on singular points |
| **11** | **Domain Error** | $\sqrt{-4}$ in real domain | Domain restriction violations |
| **12** | **Inequality Direction Error** | $-2x < 6 \implies x < -3$ | Failure to invert relation upon negative division |
| **13** | **Algebraic Manipulation** | Unsound transformations | Generic symbolic non-equivalence fallback |

---

## 📁 Project Structure

```
MathSolverr/
├── app/
│   ├── api/
│   │   └── server.py             # FastAPI backend (9 REST endpoints)
│   └── ui/
│       ├── index.html            # Warm academic 6-tab frontend
│       ├── style.css             # Typography & design system
│       └── app.js                # State management, KaTeX, Mermaid & Chart.js
├── checkpoints/
│   └── best_model.pt             # Trained PyTorch weights
├── data/
│   └── learning_system.db        # SQLite database (auto-initialized)
├── learning/
│   ├── explain_mistake.py        # 6-point pedagogical error explainer
│   ├── mistake_history.py        # SQLite Data Access Layer (DAL)
│   ├── practice_generator.py     # Weakness-targeted problem synthesis
│   └── progress.py               # Accuracy, streaks, and analytics engine
├── math_engine/
│   ├── ast.py                    # AST hierarchy & Mermaid serialization
│   ├── error_classifier.py       # 13-category error diagnosis engine
│   ├── latex_parser.py           # LaTeX to SymPy AST pipeline
│   ├── solver.py                 # Deterministic SymPy solver
│   ├── step_generator.py         # Human-readable derivation generator
│   └── verifier.py               # Step-by-step equivalence checker
├── ml/
│   ├── dataset.py                # HuggingFace loader with lazy .select()
│   ├── inference.py              # MathRecognizer inference pipeline
│   ├── model.py                  # CNN Encoder + 2D PosEnc + Transformer Decoder
│   ├── preprocessing.py          # Whitespace crop, padding & line segmentation
│   ├── tokenizer.py              # 130-token LaTeX tokenizer
│   └── training.py               # Trainer loop, CosineAnnealing, CER & EM metrics
├── tests/                        # 46 automated unit & integration tests
├── ARCHITECTURE.md               # Detailed architectural specification
├── config.yaml                   # Global hyperparameters & configuration
├── colab_train.ipynb             # Google Colab GPU training notebook
├── eval.py                       # Research evaluation suite
├── requirements.txt              # Pinned Python dependencies
├── run.py                        # Web server launcher
└── train.py                      # CLI training entry point
>>>>>>> 19aa8a5 (Fix handwriting recognition training)
```

---

<<<<<<< HEAD
## 🚀 How to Run on Your Computer

Follow these steps to run the complete project locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/harshitakhobare-eng/DL_CP.git
cd DL_CP
```
=======
## 🚀 Quick Start & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/YOUR_USERNAME/MathSolverr.git
cd MathSolverr

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
>>>>>>> 19aa8a5 (Fix handwriting recognition training)

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
### 4. Download Model Weights

1. Go to the **Releases** tab on this GitHub repository.
2. Download `best_model.pt`.
3. Create a folder named `checkpoints` inside the project and place the file there:
   ```
   DL_CP/checkpoints/best_model.pt
   ```

### 5. Launch the Application

```bash
python run.py
```

Now open your web browser and visit:
👉 **`http://127.0.0.1:8000`**

---

## 🧪 Running Automated Tests

To run the complete test suite (46 unit and integration tests covering the ML pipeline, parser, solver, verifier, and API):
=======
### 2. Download Pre-trained Weights
Download `best_model.pt` from the Releases section and place it inside:
```
MathSolverr/checkpoints/best_model.pt
```

### 3. Launch the Application
```bash
python run.py
```
Open your browser and navigate to: **`http://127.0.0.1:8000`**

---

## 💻 Interactive Web Interface

The UI is built with a **Warm Academic** design language (Lora serif typography, cream/paper background, clean spacing rhythm):

* **Solve Tab**: Upload handwritten expressions or input LaTeX to receive step-by-step derivations and live AST diagrams.
* **Check Solution Tab**: Audit multi-step solutions to flag errors with mathematical explanations.
* **Explain Tab**: Detailed 6-point pedagogical breakdown of mistakes.
* **History Tab**: Searchable audit log of historical mistakes with timestamps.
* **Practice Tab**: Synthesizes targeted practice questions based on past weaknesses.
* **Dashboard Tab**: Real-time accuracy metrics, streak counters, and error taxonomy distribution.

---

## 🧪 Testing & Evaluation

Run the complete test suite (46 unit & integration tests):
>>>>>>> 19aa8a5 (Fix handwriting recognition training)

```bash
pytest tests/ -v
```

<<<<<<< HEAD
---

## 💻 Tech Stack

* **Deep Learning**: PyTorch, Torchvision, Hugging Face Datasets
* **Math Engine**: SymPy, Python `re` AST parser
* **Backend API**: FastAPI, Uvicorn, SQLite
* **Frontend**: HTML5, CSS3 (Warm Academic design), Vanilla JavaScript, KaTeX, Mermaid.js, Chart.js
=======
Run the research evaluation benchmark:
```bash
python eval.py --checkpoint checkpoints/best_model.pt
```

### Measured Evaluation Results:
* **Symbolic Solver**: `100.0%` (6/6 passing)
* **Step Verification**: `100.0%` (5/5 passing)
* **Error Classification**: `75.0%` (3/4 passing)
* **Practice Generation Correctness**: `100.0%` (9/9 verified)
* **Automated Unit Tests**: `46/46 passed`
>>>>>>> 19aa8a5 (Fix handwriting recognition training)

---

## 📄 License
<<<<<<< HEAD

This project is licensed under the MIT License.
```
=======
This project is licensed under the MIT License.
>>>>>>> 19aa8a5 (Fix handwriting recognition training)
