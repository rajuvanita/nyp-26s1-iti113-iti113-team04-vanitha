
## Notebooks (run order)

1. **team0402_EDA_CreditCardFraud_v1.0.ipynb**
   Independent EDA for the Tree-Based track. States fraud-indicator
   hypotheses up front, then tests each against the raw transaction data,
   judging whether the evidence supports, weakens, or contradicts it.
   Output feeds the feature-engineering step below.

2. **team0402_XGBoost_SMOTE_Feature_Engineering_v1.0.ipynb**
   Turns the EDA's feature candidates into a leakage-controlled,
   model-ready feature matrix for XGBoost + SMOTE. Reads its feature list
   directly from the EDA's exported candidates file rather than
   re-deciding it, so the two notebooks can't silently drift apart.

3. **team0402_ModelA_Final_v2.0.ipynb**
   Develops, evaluates, and registers the final Model A (XGBoost + SMOTE).
   Keeps the milestone baseline as a fixed benchmark, runs a
   validation-only candidate comparison, freezes the operating threshold
   before touching test data, retrains on the combined dev set, and
   reports final test performance.

4. **team0402_ModelA_Retrain_Pipeline v2.0.ipynb**
   A separate retraining pipeline that re-applies the frozen
   feature/model decisions to new data, without re-deriving them. Kept
   apart from EDA/Feature Engineering deliberately: those represent a
   one-time human decision, while retraining is what should run
   automatically when new data arrives.

5. **team0402_Fairness_Bias_Detection_v2.0.ipynb**
   Audits Model A's held-out test predictions for fairness across age,
   gender, and spending tier. Region/state is explicitly excluded, since
   `state` was dropped during EDA and is absent from the downstream
   dataset.

6. **team0402_ModelA_Robustness_Final_v2.0.ipynb**
   Robustness and deployment validation, v2. Unlike an earlier version
   that tested a reimplementation of preprocessing, this version calls
   the actual deployed `inference.py` (`model_fn` / `predict_fn`)
   directly, so results reflect the real deployment artifact.

## Config / support files

- **mlflow_app_config_team04_s402.json** — Team/student-specific MLflow
  App configuration (region, app ARN, experiment name, S3 artifact
  store, and resource tags) used to connect to the shared tracking
  server.
- **team0402_mlflow_utils.py** — Shared helper module (`initialize_mlflow()`)
  that connects to the existing SageMaker MLflow App, sets the
  experiment, and generates a presigned MLflow UI URL. Imported by the
  notebooks above rather than duplicating connection logic in each one.

