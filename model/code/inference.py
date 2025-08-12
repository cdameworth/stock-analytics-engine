import joblib
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
