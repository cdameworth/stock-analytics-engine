"""
Native AWS Lambda ML Inference Handler
Replaces the broken ml_inference_simple_fallback with AWS native implementation.

This module provides:
- Native AWS Lambda handler using boto3
- Simple rule-based ML predictions for stock recommendations
- Integration with DynamoDB for storing predictions
- Error handling and logging
"""

import json
import logging
import os
import boto3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import uuid

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
s3_client = boto3.client('s3')

# Environment variables
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'stock-recommendations')
S3_BUCKET = os.environ.get('S3_BUCKET', '')

def lambda_handler(event, context):
    """
    Main Lambda handler for ML inference using native AWS services.

    Expected event format:
    {
        "symbol": "AAPL",
        "current_price": 150.00,
        "data": {
            "volume": 1000000,
            "market_cap": 2500000000000,
            "pe_ratio": 25.5
        }
    }
    """
    try:
        logger.info(f"ML Inference request: {json.dumps(event, default=decimal_default)}")

        # Extract parameters from event
        symbol = event.get('symbol', '')
        current_price = float(event.get('current_price', 0))
        data = event.get('data', {})

        if not symbol:
            raise ValueError("Symbol is required")

        if current_price <= 0:
            raise ValueError("Valid current_price is required")

        # Generate ML prediction using rule-based model
        prediction_result = generate_stock_prediction(symbol, current_price, data)

        # Store prediction in DynamoDB
        stored_result = store_prediction(prediction_result)

        logger.info(f"Generated prediction for {symbol}: {prediction_result['recommendation_type']} (confidence: {prediction_result['confidence']})")

        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction_result,
                'stored': stored_result,
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': context.aws_request_id
            }, default=decimal_default)
        }

    except Exception as e:
        logger.error(f"ML Inference error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': context.aws_request_id
            })
        }

def generate_stock_prediction(symbol: str, current_price: float, data: Dict) -> Dict:
    """
    Generate stock prediction using rule-based ML model.

    This is a simplified ML model that uses financial indicators to make predictions.
    In production, this could be replaced with actual ML models.
    """
    try:
        # Extract features from data
        volume = float(data.get('volume', 1000000))
        market_cap = float(data.get('market_cap', 1000000000))
        pe_ratio = float(data.get('pe_ratio', 20))
        price_change_24h = float(data.get('price_change_24h', 0))

        # Rule-based prediction logic
        score = 0.5  # Base score
        confidence = 0.5  # Base confidence

        # Volume indicator (higher volume = more confidence)
        if volume > 5000000:
            score += 0.1
            confidence += 0.1
        elif volume < 500000:
            score -= 0.1
            confidence -= 0.05

        # P/E ratio indicator
        if 10 <= pe_ratio <= 25:  # Reasonable P/E range
            score += 0.15
            confidence += 0.1
        elif pe_ratio > 40:  # Overvalued
            score -= 0.2
            confidence += 0.05

        # Recent price momentum
        if price_change_24h > 2:  # Strong positive momentum
            score += 0.2
            confidence += 0.1
        elif price_change_24h < -2:  # Strong negative momentum
            score -= 0.2
            confidence += 0.1

        # Market cap consideration
        if market_cap > 100000000000:  # Large cap (more stable)
            confidence += 0.15
        elif market_cap < 1000000000:  # Small cap (more volatile)
            confidence -= 0.1
            score += 0.05  # But potentially higher returns

        # Normalize scores
        score = max(0.0, min(1.0, score))
        confidence = max(0.3, min(0.9, confidence))

        # Determine recommendation type
        if score >= 0.65:
            recommendation_type = 'BUY'
            target_price = current_price * (1 + (score - 0.5) * 0.4)  # Up to 20% upside
        elif score <= 0.35:
            recommendation_type = 'SELL'
            target_price = current_price * (1 - (0.5 - score) * 0.3)  # Up to 15% downside
        else:
            recommendation_type = 'HOLD'
            target_price = current_price * (0.98 + (score - 0.4) * 0.04)  # Small adjustment

        # Calculate risk level
        if confidence >= 0.7:
            risk_level = 'LOW'
        elif confidence >= 0.5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        # Generate rationale
        rationale = generate_rationale(symbol, score, confidence, data)

        return {
            'symbol': symbol,
            'recommendation_type': recommendation_type,
            'prediction_score': round(score, 3),
            'confidence': round(confidence, 3),
            'current_price': current_price,
            'target_price': round(target_price, 2),
            'risk_level': risk_level,
            'rationale': rationale,
            'model_version': 'native_v1.0',
            'features_used': ['volume', 'pe_ratio', 'price_change_24h', 'market_cap']
        }

    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {str(e)}")
        # Return neutral prediction on error
        return {
            'symbol': symbol,
            'recommendation_type': 'HOLD',
            'prediction_score': 0.5,
            'confidence': 0.3,
            'current_price': current_price,
            'target_price': current_price,
            'risk_level': 'HIGH',
            'rationale': f'Prediction error: {str(e)}',
            'model_version': 'native_v1.0_fallback',
            'features_used': []
        }

def generate_rationale(symbol: str, score: float, confidence: float, data: Dict) -> str:
    """Generate human-readable rationale for the prediction."""
    rationale_parts = []

    volume = float(data.get('volume', 1000000))
    pe_ratio = float(data.get('pe_ratio', 20))
    price_change_24h = float(data.get('price_change_24h', 0))

    # Volume analysis
    if volume > 5000000:
        rationale_parts.append("High trading volume indicates strong interest")
    elif volume < 500000:
        rationale_parts.append("Low trading volume suggests limited interest")

    # P/E analysis
    if 10 <= pe_ratio <= 25:
        rationale_parts.append("P/E ratio is within reasonable range")
    elif pe_ratio > 40:
        rationale_parts.append("High P/E ratio suggests potential overvaluation")
    elif pe_ratio < 10:
        rationale_parts.append("Low P/E ratio may indicate undervaluation")

    # Momentum analysis
    if price_change_24h > 2:
        rationale_parts.append("Strong positive momentum in recent trading")
    elif price_change_24h < -2:
        rationale_parts.append("Negative momentum in recent trading")

    # Confidence qualifier
    if confidence >= 0.7:
        rationale_parts.append("High confidence in prediction")
    elif confidence < 0.5:
        rationale_parts.append("Lower confidence due to mixed signals")

    return ". ".join(rationale_parts) + "."

def store_prediction(prediction: Dict) -> bool:
    """Store prediction result in DynamoDB."""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)

        # Create recommendation record
        recommendation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        item = {
            'recommendation_id': recommendation_id,
            'symbol': prediction['symbol'],
            'recommendation_type': prediction['recommendation_type'],
            'prediction_score': Decimal(str(prediction['prediction_score'])),
            'confidence': Decimal(str(prediction['confidence'])),
            'current_price': Decimal(str(prediction['current_price'])),
            'target_price': Decimal(str(prediction['target_price'])),
            'risk_level': prediction['risk_level'],
            'rationale': prediction['rationale'],
            'timestamp': timestamp,
            'model_version': prediction['model_version'],
            'features_used': prediction['features_used'],
            'ranking': 1  # Default ranking
        }

        # Add TTL (expire after 7 days)
        ttl_timestamp = int((datetime.utcnow() + timedelta(days=7)).timestamp())
        item['ttl'] = ttl_timestamp

        table.put_item(Item=item)

        logger.info(f"Stored prediction for {prediction['symbol']} with ID {recommendation_id}")
        return True

    except Exception as e:
        logger.error(f"Error storing prediction: {str(e)}")
        return False

def decimal_default(obj):
    """JSON serializer for Decimal objects."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# Health check handler for testing
def health_check():
    """Simple health check for the Lambda function."""
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'model_version': 'native_v1.0',
            'timestamp': datetime.utcnow().isoformat()
        })
    }