# Credit Card Fraud Detection — Model B (Linear/Neural Track)

**ITI113 — MLOps for Trustworthy AI | Team 04**
**Owner:** Raju Vanitha (S401)

This directory contains the Linear/Neural modelling track — one of two independently developed
pipelines in the team's credit card fraud detection project (Model A, tree-based, is maintained
separately). Model B compares a Logistic Regression baseline against a Multi-Layer Perceptron
(MLP), covering the full lifecycle from exploratory analysis through hyperparameter tuning,
explainability, deployment testing, and retraining.

## Problem Statement

Fraudulent transactions represent 0.58% of the ~1.3M-row development dataset (a ratio of
approximately 1:173). This imbalance governs every design decision in this track: **PR-AUC**, not
accuracy, is the primary evaluation metric, and decision thresholds are selected from validation
data via a fine-grained scan rather than a fixed 0.50 default.

## Data
Dataset Repository Directory: Kaggle Credit Card Fraud Detection (https://www.kaggle.com/datasets/kartik2112/fraud-detection)

## Repository Structure

### Core Notebooks (run in this order)

| # | Notebook | Purpose |
|---|---|---|
| 1 | `team0401_Vanitha_EDA_Final.ipynb` | Exploratory analysis — six hypotheses tested against evidence |
| 2 | `team0401_Vanitha_Linear_Neural_Feature_Engineering_Final.ipynb` | Feature engineering — leakage-safe, train-only fitting |
| 3 | `team0401_Vanitha_Baseline_Model_B_Final.ipynb` | Training, and MLflow registration |
| 4 | `team0401_Vanitha_Final_Model_B_Final.ipynb` | Training, hyperparameter tuning, evaluation, and MLflow registration |
| 5 | `team0401_Vanitha_shap_explainability_Final.ipynb` | SHAP explainability — executed against both registered candidates |
| 6 | `team0401_Vanitha_Final_Model_B_Retrain_Pipeline_Final.ipynb` | Retraining and champion-promotion pipeline |
| 7 | `team0401_Vanitha_Final_Model_B_Inference_Pipeline_Final.ipynb` | Demonstrates the deployment artifact end to end |
| 8 | `team0401_Vanitha_Final_Model_B_Robustness_Final.ipynb` | Robustness testing against the real deployed code |

### Deployment Artifact

| File | Purpose |
|---|---|
| `inference.py` | Standalone SageMaker inference script (`model_fn` / `input_fn` / `predict_fn` / `output_fn`). This is the actual artifact notebooks 6 and 7 test against — not a notebook-level reimplementation. |
| `mlflow_utils.py` | Standalone SageMaker utility script.
It is an ML experiment-tracking and AWS integration utility that supports the project.


## Architecture Notes

- **Self-contained pipeline.** The registered model is a genuine `sklearn.Pipeline`
  (`M03FeatureTransformer` + classifier), not a bare estimator — scaling and one-hot encoding
  happen inside the model itself. Any script loading the pickled model must have
  `M03FeatureTransformer` defined in its own `__main__` namespace first (see notebooks 4, 5, 6).
- **MLflow registry.** Registered under `ITI113-team04-{STUDENT_ID}-ModelB-LinearNeural`. The
  retraining pipeline (notebook 5) uses a **fixed** alias, `champion-candidate`, deliberately
  different from notebook 3's dynamic `model_family`-based alias — see [Known Issues](#known-issues-and-open-items).

## Key Results

Held-out test performance, at each model's own F1-optimal threshold:

| Metric | Logistic Regression | MLP |
|---|---|---|
| PR-AUC | 0.3179 | **0.9083** |
| Precision | 40.1% | 88.0% |
| Recall | 40.2% | 80.8% |
| Confusion matrix (TN / FP / FN / TP) | 256,932 / 902 / 897 / 604 | 257,668 / 166 / 288 / 1,213 |

**MLP was selected** as the deployed candidate on the strength of test PR-AUC, corroborated by a
precision-sensitivity check: at an elevated 90% precision requirement, MLP retains 81.2% recall
while Logistic Regression collapses to 0.6% recall.

## Governance Findings

**Explainability (SHAP).** Both models identify transaction amount (`amt_log`) as their most
important feature; their rankings diverge substantially beyond that point (Spearman correlation
approximately -0.04, against a 0.70 sanity baseline for Logistic Regression against its own
coefficients). Cardholder age was not found to be an important driver for either model. Both
models independently flag the same transaction as high-confidence fraud despite it being a genuine
false positive, driven in both cases by transaction amount alone.

**Robustness testing**, run against the real deployed `inference.py`:
- A missing `category` value is silently scored identically to a genuinely unrecognised category
  — a real data-quality gap, not flagged by the current deployment code.
- No input-range validation exists for any field — implausible values (e.g. `age=150`,
  `txn_velocity_1h=999`) produce confident predictions rather than being rejected.

## Known Issues and Open Items

| Issue | Status |
|---|---|
| No deliberate input validation in `inference.py` (category, value ranges) | Open — identified by robustness testing |
| Retraining pipeline has only been exercised against the original training file | Open — next run should use genuinely new data |
| `KernelExplainer` (MLP) SHAP rankings are not stable run-to-run at `nsamples=100` | Documented in notebook 4; the overall divergence finding holds across runs, exact values do not |

## Requirements

- AWS SageMaker Studio environment with access to the team's S3 bucket
  (`nyp-26s1-iti113/iti113/team04/`) and the SageMaker MLflow tracking app
- Python packages: `scikit-learn`, `pandas`, `numpy`, `shap`, `statsmodels`, `mlflow==3.15.1`,
  `boto3`, `joblib` (installed at the top of each notebook)
- `mlflow_utils` (shared team module) for MLflow session initialization

## Notebook Versioning

Only the latest version of each notebook is listed above and should be committed. Earlier
iterations (visible in commit history / prior exports) reflect incremental fixes — including a
sklearn `clone()` bug in the hyperparameter search, MLflow deprecation warnings, a coarse
threshold-search grid, and a baseline→final naming migration — and are not needed once superseded.
