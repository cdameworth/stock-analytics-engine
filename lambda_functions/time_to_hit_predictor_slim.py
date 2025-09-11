"""
Time-to-Hit Prediction System - Slim Version
Basic prediction without scientific computing dependencies
"""

import json
import boto3
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

# Environment variables
TIME_PREDICTIONS_TABLE = os.environ.get('TIME_PREDICTIONS_TABLE', 'time-to-hit-predictions')
S3_DATA_BUCKET = os.environ.get('S3_DATA_BUCKET', 'stock-analytics-data-lake')

def lambda_handler(event, context):
    """
    Simplified time-to-hit prediction handler
    Returns basic time estimates without complex ML
    """
    try:
        logger.info(f"Time-to-hit prediction request: {event}")
        
        # Parse input - handle both direct calls and API Gateway events
        if 'body' in event:
            # API Gateway event structure
            body = json.loads(event['body']) if event.get('body') else {}
            symbol = body.get('symbol')
            current_price = float(body.get('current_price', 0))
            target_price = float(body.get('target_price', 0))
            recommendation_data = body.get('recommendation_data', {})
        else:
            # Direct Lambda invocation
            symbol = event.get('symbol')
            current_price = float(event.get('current_price', 0))
            target_price = float(event.get('target_price', 0))
            recommendation_data = event.get('recommendation_data', {})
        
        if not symbol or current_price <= 0 or target_price <= 0:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Invalid input parameters',
                    'message': 'Symbol, current_price, and target_price are required'
                })
            }
        
        # Calculate price change percentage
        price_change_pct = ((target_price - current_price) / current_price) * 100
        
        # Basic time estimation logic based on price change magnitude
        if abs(price_change_pct) < 2:
            expected_days_min, expected_days_max = 5, 15
            confidence = "high"
        elif abs(price_change_pct) < 5:
            expected_days_min, expected_days_max = 10, 30
            confidence = "medium"
        elif abs(price_change_pct) < 10:
            expected_days_min, expected_days_max = 15, 60
            confidence = "medium"
        else:
            expected_days_min, expected_days_max = 30, 120
            confidence = "low"
        
        # Create prediction record for storage
        prediction_record = {
            'prediction_id': f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'symbol': symbol,
            'target_price': Decimal(str(target_price)),
            'current_price': Decimal(str(current_price)),
            'expected_days_min': expected_days_min,
            'expected_days_max': expected_days_max,
            'probability_distribution': {
                'p25': Decimal(str(expected_days_min)),
                'p50': Decimal(str((expected_days_min + expected_days_max) / 2)),
                'p75': Decimal(str(expected_days_max))
            },
            'confidence_level': confidence,
            'market_regime': 'normal',
            'key_factors': ['price_change_magnitude', 'basic_estimation'],
            'risk_factors': ['simplified_model'],
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'status': 'pending',
            'validation_data': {}
        }
        
        # Store prediction for validation
        try:
            table = dynamodb.Table(TIME_PREDICTIONS_TABLE)
            table.put_item(Item=prediction_record)
            logger.info(f"Stored time prediction for validation tracking: {prediction_record['prediction_id']}")
        except Exception as e:
            logger.error(f"Error storing prediction for validation: {str(e)}")
        
        # Return prediction response
        response = {
            'symbol': symbol,
            'expected_days_range': [expected_days_min, expected_days_max],
            'expected_timeline': f"{expected_days_min}-{expected_days_max} days",
            'confidence_level': confidence,
            'timing_summary': f"Expected to reach ${target_price:.2f} in {expected_days_min}-{expected_days_max} days",
            'price_change_percentage': round(price_change_pct, 2),
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'model_version': 'slim_1.0',
            'key_factors': ['price_change_magnitude', 'basic_estimation'],
            'risk_factors': ['simplified_model'],
            'market_regime': 'normal',
            'probability_distribution': {
                '7_days': 0.2 if confidence == 'high' else 0.1,
                '15_days': 0.5 if confidence == 'high' else 0.3,
                '30_days': 0.8 if confidence != 'low' else 0.6
            }
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Error in time prediction: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }