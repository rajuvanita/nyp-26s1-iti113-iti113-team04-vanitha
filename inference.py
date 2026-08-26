"""
inference.py — Model B (Linear/Neural) SageMaker inference script.
Vanitha — Linear/Neural Pipeline Track (Logistic Regression + MLP)

Standard SageMaker script-mode contract: model_fn / input_fn / predict_fn / output_fn.
This is the real deployment artifact — team0401_Vanitha_Final_Model_B_Robustness.ipynb imports
and calls this module directly, the same principle Model A's robustness notebook uses ("test the
actual deployed code, not a reimplementation that can silently drift out of sync").

Architectural note carried over from the Inference Pipeline notebook this was extracted from:
Model B's registered model is a genuine sklearn.Pipeline (M03FeatureTransformer + classifier), not
a bare classifier — scaling and one-hot encoding happen INSIDE the loaded model itself. That means
predict_fn here does not need Model A's separate _apply_target_encoding()-style preprocessing step;
it hands the 13 raw engineered fields straight to the loaded Pipeline.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class M03FeatureTransformer(BaseEstimator, TransformerMixin):
    """Copied verbatim from team0401_Vanitha_Final_Model_B.ipynb (M06) — must match exactly, so
    pickle can resolve this class while reconstructing the loaded Pipeline object. Not
    instantiated directly in this script; the loaded Pipeline already carries its own fitted
    scaler/ohe internally.
    """

    def __init__(self, scaler, ohe, continuous_features, ohe_columns, passthrough_features):
        self.scaler = scaler
        self.ohe = ohe
        self.continuous_features = continuous_features
        self.ohe_columns = ohe_columns
        self.passthrough_features = passthrough_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaled = pd.DataFrame(
            self.scaler.transform(X[self.continuous_features]),
            columns=self.continuous_features,
            index=X.index,
        )
        encoded = pd.DataFrame(
            self.ohe.transform(X[['category']]),
            columns=self.ohe_columns,
            index=X.index,
        )
        passthrough = X[self.passthrough_features].reset_index(drop=True)
        result = pd.concat(
            [scaled.reset_index(drop=True), encoded.reset_index(drop=True), passthrough],
            axis=1,
        )
        result.index = X.index
        return result

    def get_feature_names_out(self, input_features=None):
        return np.array(self.continuous_features + self.ohe_columns + self.passthrough_features)


DEPLOYMENT_CONTRACT_FILENAME = "modelb_deployment_contract.json"
PREPROCESSING_BUNDLE_FILENAME = "modelb_preprocessing_bundle.joblib"
MODEL_FILENAME = "model.pkl"


def model_fn(model_dir):
    """SageMaker calls this once per worker with the local directory the model.tar.gz was
    extracted into. Loads the pickled Pipeline, the deployment contract, and the preprocessing
    bundle (kept for its operating_threshold and metadata — not needed to preprocess anything,
    since the loaded Pipeline already does that internally).
    """
    model_dir = Path(model_dir)

    with open(model_dir / MODEL_FILENAME, "rb") as f:
        model = joblib.load(f)

    with open(model_dir / DEPLOYMENT_CONTRACT_FILENAME, "r", encoding="utf-8") as f:
        contract = json.load(f)

    preprocessing_bundle = joblib.load(model_dir / PREPROCESSING_BUNDLE_FILENAME)

    return {
        "model": model,
        "contract": contract,
        "preprocessing_bundle": preprocessing_bundle,
        "operating_threshold": float(contract["operating_threshold"]),
    }


def input_fn(request_body, content_type="application/json"):
    """Deserializes an incoming request into a DataFrame of raw engineered fields.
    Accepts either a single record ({"amt": ..., ...}) or a batch ([{...}, {...}]).
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type!r}. Only application/json is accepted.")

    payload = json.loads(request_body)
    records = payload if isinstance(payload, list) else [payload]
    return pd.DataFrame(records)


def predict_fn(input_data, model_bundle):
    """input_data: a DataFrame of raw engineered fields (RAW_FIELDS_REQUIRED, from the
    deployment contract) — matches Feature-Engineering-level output, same as the Inference
    Pipeline notebook's predict_from_engineered_features(). Returns a DataFrame with
    fraud_probability and is_fraud_predicted columns, one row per input row.
    """
    contract = model_bundle["contract"]
    model = model_bundle["model"]
    threshold = model_bundle["operating_threshold"]
    raw_fields_required = contract["raw_input_fields_required"]

    df = input_data.copy()
    missing = [c for c in raw_fields_required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields per the deployment contract: {missing}")

    X = df[raw_fields_required]

    # The loaded Pipeline's 'features' step (M03FeatureTransformer) performs scaling and
    # one-hot encoding internally — no manual preprocessing needed here.
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)

    return pd.DataFrame({
        "fraud_probability": probability,
        "is_fraud_predicted": prediction,
    }, index=df.index)


def output_fn(prediction, accept="application/json"):
    """Serializes predict_fn's output DataFrame back to the response body."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept!r}. Only application/json is produced.")
    return prediction.to_json(orient="records"), accept