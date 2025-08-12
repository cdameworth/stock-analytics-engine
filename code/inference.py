import joblib
import json
import numpy as np
import os
import logging

logger = logging.getLogger()

def model_fn(model_dir):
    """Load the model artifacts"""
    try:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        logger.info("Model loaded successfully")
        return {
            "model": model,
            "scaler": scaler,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def input_fn(request_body, content_type):
    """Parse the input data from your Lambda function"""
    try:
        if content_type == 'application/json':
            input_data = json.loads(request_body)
            
            # Handle the format your Lambda sends: {"instances": [[val1, val2, ...]]}
            if 'instances' in input_data:
                instances = input_data['instances']
                return np.array(instances)
            else:
                # Handle direct array format
                return np.array([input_data])
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        logger.error(f"Error parsing input: {str(e)}")
        raise e

def predict_fn(input_data, model_dict):
    """Make predictions using the loaded model"""
    try:
        model = model_dict["model"]
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Ensure predictions are in valid range [0, 1]
        predictions = np.clip(predictions, 0, 1)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions.tolist()
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise e

def output_fn(predictions, content_type):
    """Format the output for your Lambda function"""
    try:
        if content_type == 'application/json':
            # Return in the format your Lambda expects
            return json.dumps({
                "predictions": predictions
            })
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        raise e