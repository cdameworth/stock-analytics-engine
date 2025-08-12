# build_artifacts_for_sagemaker.py
# Creates two artifacts for SageMaker (sklearn container):
#   1) stock_prediction_compatible_model.tar.gz  (model files only)
#   2) code.tar.gz                               (inference.py only)

import os, tarfile, json, shutil
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

MODEL_DIR = "model_artifacts"
CODE_DIR = "code"
MODEL_TAR = "stock_prediction_compatible_model.tar.gz"
CODE_TAR = "code.tar.gz"

INFERENCE_PY = r'''import joblib
import json
import numpy as np
import os
import logging

logger = logging.getLogger()

def model_fn(model_dir):
    """Load the model artifacts placed by SageMaker at /opt/ml/model"""
    try:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        # scaler is optional; keep for future compatibility
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        meta_path = os.path.join(model_dir, "metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        logger.info("Model loaded successfully")
        return {"model": model, "scaler": scaler, "metadata": metadata}
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, content_type):
    """Accepts JSON:
        {"instances": [[f1, f2, ...], [f1, f2, ...]]}
       or a direct JSON array for a single instance.
    """
    try:
        if content_type != "application/json":
            raise ValueError(f"Unsupported content type: {content_type}")

        payload = json.loads(request_body)
        if isinstance(payload, dict) and "instances" in payload:
            return np.array(payload["instances"])
        # single array case
        if isinstance(payload, list):
            return np.array([payload])
        raise ValueError("JSON must be {'instances': [[...], ...]} or a single array")
    except Exception as e:
        logger.error(f"Error parsing input: {str(e)}")
        raise

def predict_fn(input_data, model_bundle):
    """Runs prediction and clips to [0, 1]."""
    try:
        model = model_bundle["model"]
        preds = model.predict(input_data)
        preds = np.clip(preds, 0, 1)
        logger.info(f"Generated {len(preds)} predictions")
        return preds.tolist()
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise

def output_fn(predictions, accept):
    """Returns {"predictions":[...]} as JSON."""
    try:
        if accept != "application/json":
            raise ValueError(f"Unsupported accept: {accept}")
        return json.dumps({"predictions": predictions})
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        raise
'''

def build_model_and_artifacts():
    # === 1) Train a small synthetic model (same as your previous logic) ===
    np.random.seed(42)
    n_samples = 15000
    price_to_ma5_ratio   = np.random.normal(1.0, 0.08, n_samples)
    price_to_ma20_ratio  = np.random.normal(1.0, 0.12, n_samples)
    volatility           = np.abs(np.random.normal(0.025, 0.015, n_samples))
    volume_normalized    = np.random.uniform(0, 1, n_samples)
    market_sentiment     = np.random.uniform(0, 1, n_samples)
    market_volatility    = np.abs(np.random.normal(0.025, 0.01, n_samples))
    market_correlation   = np.random.uniform(0, 1, n_samples)

    X = np.column_stack([
        price_to_ma5_ratio,
        price_to_ma20_ratio,
        volatility,
        volume_normalized,
        market_sentiment,
        market_volatility,
        market_correlation
    ])

    y = (
        0.5
        + 0.2  * np.clip((price_to_ma5_ratio - 1) * 5, -0.2, 0.2)
        + 0.15 * np.clip((price_to_ma20_ratio - 1) * 3, -0.15, 0.15)
        - 0.1  * np.clip(volatility * 20, 0, 0.1)
        + 0.08 * volume_normalized
        + 0.15 * market_sentiment
        - 0.05 * np.clip(market_volatility * 20, 0, 0.05)
        + 0.05 * market_correlation
        + np.random.normal(0, 0.03, n_samples)
    )
    y = np.clip(y, 0, 1)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ).fit(X, y)

    scaler = StandardScaler().fit(X)

    # === 2) Write model files ===
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    meta = {
        "model_type": "RandomForestRegressor",
        "feature_count": 7,
        "feature_names": [
            "price_to_ma5_ratio",
            "price_to_ma20_ratio",
            "volatility",
            "volume_normalized",
            "market_sentiment",
            "market_volatility",
            "market_correlation"
        ],
        "training_samples": int(n_samples),
        "model_score_r2_in_sample": float(model.score(X, y)),
        "prediction_range": [0.0, 1.0],
        "built_at": datetime.utcnow().isoformat() + "Z"
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # === 3) Create the model tar (no code inside) ===
    if os.path.exists(MODEL_TAR):
        os.remove(MODEL_TAR)
    with tarfile.open(MODEL_TAR, "w:gz") as tar:
        # put files at tar root as SageMaker expects
        tar.add(os.path.join(MODEL_DIR, "model.pkl"), arcname="model.pkl")
        tar.add(os.path.join(MODEL_DIR, "scaler.pkl"), arcname="scaler.pkl")
        tar.add(os.path.join(MODEL_DIR, "metadata.json"), arcname="metadata.json")

    # === 4) Create code dir and code tar (only inference.py) ===
    if os.path.exists(CODE_DIR):
        shutil.rmtree(CODE_DIR)
    os.makedirs(CODE_DIR, exist_ok=True)
    with open(os.path.join(CODE_DIR, "inference.py"), "w") as f:
        f.write(INFERENCE_PY)

    if os.path.exists(CODE_TAR):
        os.remove(CODE_TAR)
    with tarfile.open(CODE_TAR, "w:gz") as tar:
        tar.add(os.path.join(CODE_DIR, "inference.py"), arcname="inference.py")

    # Keep folders for versioning/inspection; don’t delete.
    return {
        "model_tar": os.path.abspath(MODEL_TAR),
        "code_tar": os.path.abspath(CODE_TAR),
        "model_dir": os.path.abspath(MODEL_DIR),
        "code_dir": os.path.abspath(CODE_DIR),
        "r2_in_sample": meta["model_score_r2_in_sample"]
    }

if __name__ == "__main__":
    out = build_model_and_artifacts()
    print("\n=== Artifacts ready ===")
    print(f"Model tar: {out['model_tar']}")
    print(f"Code tar : {out['code_tar']}")
    print(f"R^2 (in-sample): {out['r2_in_sample']:.4f}")
    print("\nNext:")
    print("  aws s3 cp stock_prediction_compatible_model.tar.gz s3://<YOUR_BUCKET>/stock_prediction_compatible_model.tar.gz")
    print("  aws s3 cp code.tar.gz s3://<YOUR_BUCKET>/code/code.tar.gz")
