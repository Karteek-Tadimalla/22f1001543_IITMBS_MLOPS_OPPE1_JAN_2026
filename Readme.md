# 22f1001543_IITMBS_MLOPS_OPPE1_JAN_2026

End-to-end MLOps pipeline for stock movement prediction as part of the IITM BS MLOps OPPE (Jan 2026).  
The project covers data versioning, experiment tracking, model selection, and CI-based evaluation using a single exported best model artifact.

---

## Project Structure

Key directories and files:

- `data/`
  - `raw/` – original input data (not committed if large).
  - `processed/v0/`
    - `train.parquet` – training split.
    - `test.parquet` – test split used in CI.
- `models/`
  - `best_model.joblib` – exported best model (scikit-learn compatible).
- `src/`
  - `hpt_train.py` – hyperparameter tuning and training with MLflow logging (used locally).
  - `register_model.py` – optional registration of best run in MLflow (used locally).
  - `ci_evaluate.py` – CI evaluation script loading `best_model.joblib`.
  - `ci_sanity_checks.py` – CI sanity checks on evaluation metrics.
- `.github/workflows/`
  - `ci.yml` – GitHub Actions workflow that runs evaluation and posts a CML report.
- `.gitignore`
  - Ignores local MLflow artifacts (`mlruns/`, `mlflow.db`) and virtualenv noise.

---

## Workflow Overview

### 1. Local Experimentation (MLflow)

Locally, MLflow is used only for **experiment tracking and model selection**:

- `hpt_train.py` runs hyperparameter tuning and logs:
  - parameters,
  - metrics (e.g., accuracy, F1),
  - and model artifacts to the local MLflow tracking directory.
- The best run is chosen from MLflow (e.g., by highest F1 score).
- The trained best model is exported to a standalone file:

```python
import os
import joblib

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.joblib")
```

This separates **experiment history (local)** from **deployment artifact (versioned in git)**.

---

### 2. Best Model Artifact

The file `models/best_model.joblib` is the **single source of truth** for CI:

- It contains the final chosen model in a format compatible with scikit-learn’s `joblib.load`. [web:200][web:201]
- MLflow metadata (`mlruns/`, `mlflow.db`) is **not** tracked in git and remains local only.
- This keeps the repository small and makes CI reproducible without requiring MLflow backend access. [web:201][web:206]

---

### 3. CI Evaluation

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on each push and pull request:

1. Checks out the repo.
2. Sets up Python and installs dependencies from `requirements.txt` plus `cml`.
3. Runs `src/ci_evaluate.py`.
4. Runs `src/ci_sanity_checks.py`.
5. Creates a CML comment with the evaluation report (`reports/ci_metrics.md`) on the PR/commit.

#### `ci_evaluate.py` (core logic)

- Loads the test data:

```python
import pandas as pd

df = pd.read_parquet("data/processed/v0/test.parquet")
y = df["target"]
X = df.drop(columns=["target"])
```

- Loads the model:

```python
import joblib

model = joblib.load("models/best_model.joblib")
```

- Computes metrics (accuracy, F1, precision, recall) and writes:
  - `reports/ci_metrics.json` – machine-readable metrics.
  - `reports/ci_metrics.md` – human-readable markdown report.

This ensures CI always evaluates the **current best committed model** on a fixed test set and surfaces metrics automatically.

---

### 4. CI Sanity Checks

`src/ci_sanity_checks.py` enforces simple quality gates:

- Loads `reports/ci_metrics.json`.
- Fails the job (`exit 1`) if:
  - F1 score is below a defined threshold (e.g., 0.50).
  - Accuracy is below a defined threshold.
- Prints “Sanity checks passed.” when metrics meet expectations.

This prevents regressions where a newly committed best model is significantly worse than previous versions.

---

### 5. MLflow Artifacts and Git

To keep the repo clean:

- `mlruns/` and `mlflow.db` are **ignored** in `.gitignore` and removed from git tracking via:

```bash
git rm -r --cached mlruns
```

- MLflow tracking data stays local for experimentation and exam inspection, but is not part of the git-managed artifact set.
- Only the **final exported model** (`models/best_model.joblib`) and CI/reporting scripts are under version control.

---

## How to Run Locally

### 1. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train and export best model (local)

```bash
python src/hpt_train.py \
  --train_path data/processed/v0/train.parquet \
  --test_path data/processed/v0/test.parquet \
  --out_json reports/hpt_best_v0.json

# Inside training / selection code, ensure best_model is saved to:
# models/best_model.joblib
```

### 3. Run CI evaluation locally (optional)

```bash
python src/ci_evaluate.py
python src/ci_sanity_checks.py
cat reports/ci_metrics.md
```

This reproduces what GitHub Actions does during CI.

---

## Design Rationale

- **MLflow for research, not deployment**: MLflow is used to explore hyperparameters and track experiments locally; the CI pipeline is intentionally kept independent of MLflow infrastructure by relying on a single exported model artifact. [web:152][web:201]
- **Small, versioned artifact**: `best_model.joblib` is small, easy to version, and simple for CI to load using scikit-learn patterns. [web:200][web:201][web:206]
- **Clean git history**: Ignoring `mlruns/` and `mlflow.db` avoids pushing thousands of small MLflow files and keeps the repo focused on code, data schema, and final artifacts.

---
