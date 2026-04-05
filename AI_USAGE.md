# AI Usage Documentation

## AI Tools Utilized and Conversation History
> List all GenAI / LLM tools used during the exam  
> Provide **public share links** to AI chats or attach conversation files if links are not available  

- Tool Name: Perplexity (powered by GPT-5.1)
    - Purpose :  
      Helped design the ML experiment tracking and CI flow for the OPPE project: MLflow configuration, exporting the best model as `best_model.joblib`, simplifying CI evaluation and sanity checks, and cleaning up git tracking of `mlruns` and `mlflow.db`. [web:120][web:146][web:152]
    - Shared Chat Link :  
      Not available (this tool instance does not support public share links).
    - Notes (optional) :  
      All relevant prompts and responses that influenced the solution are included below in Section 3.

---

## 3️⃣ Prompts and Responses Used
> Include **all prompts** that contributed to solving the exam tasks  
> Include **all responses** in case public share links are not available

### Tool Name #1: Perplexity (powered by GPT-5.1)

#### Prompt 1
> `but any other way`

**Response (summary):**  
Explained there are two main patterns for CI to access the best MLflow model:  
1) Export the best model as a small artifact (e.g., `model.joblib`) and make CI load that file.  
2) Share the MLflow tracking/registry backend with CI so it can query runs or registry directly.  
Recommended option (1) as simpler and more robust for the exam, with local MLflow for HPO and versioning, and CI reading an exported best model artifact. [web:120][web:126]

---

#### Prompt 2
> `no it should search the ml runs though right so should i commit mlruns as well`

**Response (summary):**  
Clarified that if CI must “search MLflow runs” to pick the best model, it needs access to the same MLflow backend as local. Suggested using a SQLite backend (`sqlite:///mlflow.db`) as a shared backend store and committing `mlflow.db` plus minimal model registry artifacts under `mlruns/models/...`. Showed how to set `MLFLOW_TRACKING_URI` to the SQLite URI in both local environment and CI, and load models using URIs like `models:/stock_movement_model/1` or `runs:/<run_id>/model`. [web:146][web:120][web:134]

---

#### Prompt 3
> Shell log showing:
> ```bash
> echo "mlruns/models" >> .gitignore
> git add .gitignore
> git commit -m "Ignore local MLflow Model runs directory"
> ```

**Response (summary):**  
Noted that ignoring `mlruns/models` is the opposite of what’s needed if CI must read from the MLflow model registry, because that directory holds the registered model artifacts. Recommended using:
```gitignore
mlruns/
!mlruns/models/
```
and committing `mlruns/models` and `mlflow.db` so CI can access model artifacts while still ignoring bulky run data. [web:146]

---

#### Prompt 4
> `give proper code entirely what to do next from creataion`

**Response (summary):**  
Provided a full plan to use a SQLite-backed MLflow tracking server:  
- Set `MLFLOW_TRACKING_URI="sqlite:///mlflow.db"` and run `mlflow db upgrade mlflow.db`.  
- Start `mlflow server` with `--backend-store-uri sqlite:///mlflow.db` and `--default-artifact-root ./mlruns`.  
- Re-run `hpt_train.py` and `register_model.py` against that server so runs and registry entries live in `mlflow.db`.  
- In CI, set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` and use `mlflow.search_runs()` to pick the best run (e.g., max `f1_score`), then load the model via `mlflow.pyfunc.load_model("runs:/<best_run_id>/model")`. [web:168][web:146][web:141][web:138]

---

#### Prompt 5
> `no i re ran i tafter that what i have mmlflow.db`

**Response (summary):**  
Explained that having `mlflow.db` alone does not guarantee runs are in it; if the training script still logs to default `./mlruns`, `mlflow.search_runs()` against `sqlite:///mlflow.db` will be empty. Suggested verifying with:
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
print(mlflow.search_runs().head())
```
and, if empty, re-running `hpt_train.py` with `MLFLOW_TRACKING_URI="sqlite:///mlflow.db"` ensuring the script doesn’t override the tracking URI internally. [web:168][web:170][web:152]

---

#### Prompt 6
> Shell output showing `mlruns` structure and empty `search_runs()` on SQLite

**Response (summary):**  
Diagnosed that old runs are still stored in the file-based `mlruns/` backend and `mlflow.db` is empty, so querying SQLite returns an empty DataFrame. Suggested abandoning the SQLite migration for the OPPE and instead using `MLFLOW_TRACKING_URI="mlruns"` everywhere, then committing the `mlruns/` directory so CI can search and load runs directly from the file-backed store. [web:177][web:152][web:138]

---

#### Prompt 7
> Shell logs: `MLFLOW_TRACKING_URI="mlruns"` and `mlflow.search_runs()` still empty

**Response (summary):**  
Concluded that training scripts weren’t actually logging to the `mlruns` URI. Provided a minimal debug script:
```python
with mlflow.start_run(run_name="debug_run") as run:
    mlflow.log_metric("debug_metric", 1.23)
```
and showed that this produced a visible run in `search_runs()`, proving MLflow and the file backend work. Recommended adding:
```python
import os, mlflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
```
at the top of training scripts and re-running HPO so real runs appear under `mlruns`. [web:152][web:183][web:178]

---

#### Prompt 8
> Shell log of debug run showing `debug_run` in search results

**Response (summary):**  
Confirmed MLflow + `mlruns` were now functioning. Suggested standardizing on `mlruns` as tracking URI in all project scripts, then provided a `ci_evaluate.py` that:
- Sets tracking URI from env (default `mlruns`),
- Calls `mlflow.search_runs(search_all_experiments=True, order_by=["metrics.f1_score DESC"])`,
- Takes the best run’s `run_id`,
- Loads the model via `mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")`,
- Evaluates on test data and writes metrics.  
Also provided a basic `ci_sanity_checks.py` and GitHub Actions workflow for CI. [web:152][web:178][web:144]

---

#### Prompt 9
> Git diff showing many `mlruns/...` files being renamed/deleted and model moved to `models/best_model.joblib`  
> Comment: “this will probably take tmrw”

**Response (summary):**  
Advised stopping that heavy push and not uploading the entire `mlruns/` tree. Recommended instead:
- Use `git rm -r --cached mlruns` to remove MLflow artifacts from tracking but keep them locally.  
- Add `mlruns/` and `mlflow.db` to `.gitignore`.  
- Switch to a simpler approach where MLflow is used locally to pick the best model, then the chosen model is exported as `best_model.joblib` and only that file is committed and used by CI. [web:184][web:190][web:214][web:216]

---

#### Prompt 10
> `so give only best model code`

**Response (summary):**  
Provided a minimal pattern for exporting the best scikit‑learn model:
```python
import os, joblib
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.joblib")
```
and, if only available via MLflow, loading it with:
```python
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}/model")
underlying_model = getattr(pyfunc_model, "_model_impl", pyfunc_model)
joblib.dump(underlying_model, "models/best_model.joblib")
```
Then gave `ci_evaluate.py` that just loads `models/best_model.joblib`, runs predictions on `data/processed/v0/test.parquet`, computes metrics, and writes `ci_metrics.json` and `ci_metrics.md`, plus a simple sanity check script and CI workflow that no longer depend on MLflow tracking store. [web:200][web:201][web:206]

---

#### Prompt 11
> User pasted `import mlflow` etc. directly into bash and got `command not found` errors

**Response (summary):**  
Clarified that Python code must be run via `python` or a `.py` file, not pasted directly into bash. Provided a one‑shot command using a here‑document:
```bash
python - << 'EOF'
import os, mlflow, joblib
os.makedirs("models", exist_ok=True)
best_run_id = "f943b26d8c98473985faea593a0340c7"
model_uri = f"runs:/{best_run_id}/model"
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
underlying_model = getattr(pyfunc_model, "_model_impl", pyfunc_model)
joblib.dump(underlying_model, "models/best_model.joblib")
print("Saved models/best_model.joblib")
EOF
```
so the best model artifact is created correctly. [web:200][web:206]

---

#### Prompt 12
> `next be quick`

**Response (summary):**  
Gave a concise sequence to finalize the best-model-based CI:  
- `ls models/best_model.joblib` to verify file,  
- Add `.gitignore` entries for `mlruns/` and `mlflow.db`,  
- Create `src/ci_evaluate.py` and `src/ci_sanity_checks.py` using `joblib.load("models/best_model.joblib")`,  
- Create `.github/workflows/ci.yml` that installs deps, runs evaluation and sanity checks, then posts a CML report,  
- Test locally and push. [web:200][web:201][web:206]

---

#### Prompt 13
> Git diff with many `mlruns/...` deletes and rename of `model.pkl` to `models/best_model.joblib`

**Response (summary):**  
Explained that git renaming `mlruns/.../model.pkl` to `models/best_model.joblib` and deleting `mlruns` registry paths is exactly the desired result: git history sheds MLflow artifacts and keeps only the exported best model. Advised letting push finish and then confirming on GitHub that `models/best_model.joblib`, CI scripts, and `.gitignore` are correct. [web:184][web:215]

---

#### Prompt 14
> `mlruns isnt under gitignore`

**Response (summary):**  
Provided commands to append:
```gitignore
mlruns/
mlflow.db
```
to `.gitignore`, then `git add .gitignore`, and if needed:
```bash
git rm -r --cached mlruns
git commit -m "Ignore local MLflow tracking artifacts"
git push origin main
```
so `mlruns` is no longer tracked and is ignored in future commits. [web:214][web:221][web:225]

---

#### Prompt 15
> `its not working somethings wrong` and `why is it 2245 items / 2245 mlruns isnt hsowing anything`

**Response (summary):**  
Clarified that numbers like “Enumerating objects: 2254” and “Counting objects: 2254” refer to total git objects (additions, deletions, trees, history), not just `mlruns`. Showed how to verify `mlruns` is no longer tracked with:
```bash
git ls-files mlruns | head
```
and explained that a non-zero object count is normal even after untracking `mlruns`; what matters is that `git ls-files mlruns` returns nothing and `.gitignore` includes `mlruns/`, ensuring future commits don’t re-add it. [web:215][web:216][web:220][web:223]
