"""
Stock Recommendations API Lambda Function
Serves stock recommendations via API Gateway
"""

import json
import boto3
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import os

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
redis_client = None

# Configuration
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
REDIS_ENDPOINT = os.environ.get('REDIS_ENDPOINT')

def lambda_handler(event, context):
    """
    Main Lambda handler function for API Gateway
    """
    try:
        # Initialize Redis connection if available
        if REDIS_ENDPOINT:
            import redis
            global redis_client
            redis_client = redis.Redis(host=REDIS_ENDPOINT, port=6379, decode_responses=True)
        
        # Parse request
        http_method = event['httpMethod']
        path = event['path']
        query_params = event.get('queryStringParameters') or {}
        
        logger.info(f"Processing {http_method} request to {path}")
        
        # Route request
        if http_method == 'GET' and path == '/recommendations':
            return get_recommendations(query_params)
        elif http_method == 'GET' and path.startswith('/recommendations/'):
            symbol = path.split('/')[-1].upper()
            return get_recommendation_by_symbol(symbol, query_params)
        else:
            return {
                'statusCode': 404,
                'headers': get_cors_headers(),
                'body': json.dumps({'error': 'Endpoint not found'})
            }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Internal server error',
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def get_recommendations(query_params):
    """
    Get stock recommendations with optional filtering
    """
    try:
        # Parse query parameters
        limit = int(query_params.get('limit', 10))
        limit = min(limit, 50)  # Cap at 50
        
        recommendation_type = query_params.get('type', '').upper()
        risk_level = query_params.get('risk', '').upper()
        min_confidence = float(query_params.get('min_confidence', 0.0))
        
        # Check cache first
        cache_key = f"recommendations:{recommendation_type}:{risk_level}:{min_confidence}:{limit}"
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info("Returning cached recommendations")
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': cached_result
            }
        
        # Query DynamoDB
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Get recent recommendations (last 24 hours)
        cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        # Scan with filters (in production, you'd want to use GSI for better performance)
        filter_expression = "attribute_exists(recommendation_id) AND #ts > :cutoff"
        expression_values = {':cutoff': cutoff_time}
        expression_names = {'#ts': 'timestamp'}
        
        if recommendation_type in ['BUY', 'SELL', 'HOLD']:
            filter_expression += " AND recommendation_type = :rec_type"
            expression_values[':rec_type'] = recommendation_type
        
        if risk_level in ['LOW', 'MEDIUM', 'HIGH']:
            filter_expression += " AND risk_level = :risk"
            expression_values[':risk'] = risk_level
        
        if min_confidence > 0:
            filter_expression += " AND confidence >= :min_conf"
            expression_values[':min_conf'] = Decimal(str(min_confidence))
        
        response = table.scan(
            FilterExpression=filter_expression,
            ExpressionAttributeNames=expression_names,
            ExpressionAttributeValues=expression_values,
            Limit=limit * 2  # Get more to account for filtering
        )
        
        items = response.get('Items', [])
        
        # Sort by prediction score and ranking
        items.sort(key=lambda x: (float(x.get('prediction_score', 0)), x.get('ranking', 999)), reverse=True)
        
        # Limit results
        items = items[:limit]
        
        # Format response
        recommendations = []
        for item in items:
            rec = format_recommendation(item)
            recommendations.append(rec)
        
        response_data = {
            'recommendations': recommendations,
            'count': len(recommendations),
            'filters_applied': {
                'type': recommendation_type or 'all',
                'risk_level': risk_level or 'all',
                'min_confidence': min_confidence,
                'limit': limit
            },
            'timestamp': datetime.utcnow().isoformat(),
            'cache_duration': 300  # 5 minutes
        }
        
        response_body = json.dumps(response_data, default=decimal_default)
        
        # Cache the result
        cache_result(cache_key, response_body, 300)  # 5 minutes
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': response_body
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': str(e)})
        }

def get_recommendation_by_symbol(symbol, query_params):
    """
    Get recommendation for specific stock symbol
    """
    try:
        # Check cache first
        cache_key = f"recommendation:{symbol}"
        cached_result = get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Returning cached recommendation for {symbol}")
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': cached_result
            }
        
        # Query DynamoDB
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        # Get recent recommendations for this symbol
        cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        
        response = table.scan(
            FilterExpression="symbol = :symbol AND #ts > :cutoff",
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={
                ':symbol': symbol,
                ':cutoff': cutoff_time
            }
        )
        
        items = response.get('Items', [])
        
        if not items:
            return {
                'statusCode': 404,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'error': f'No recent recommendations found for {symbol}',
                    'symbol': symbol
                })
            }
        
        # Get the most recent recommendation
        latest_item = max(items, key=lambda x: x['timestamp'])
        
        # Format response
        recommendation = format_recommendation(latest_item)
        
        # Add historical data if requested
        include_history = query_params.get('include_history', 'false').lower() == 'true'
        if include_history:
            # Sort by timestamp, most recent first
            historical_items = sorted(items, key=lambda x: x['timestamp'], reverse=True)
            recommendation['history'] = [format_recommendation(item) for item in historical_items[:10]]
        
        response_data = {
            'recommendation': recommendation,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        response_body = json.dumps(response_data, default=decimal_default)
        
        # Cache the result
        cache_result(cache_key, response_body, 180)  # 3 minutes
        
        logger.info(f"Returning recommendation for {symbol}")
        
        return {
            'statusCode': 200,
            'headers': get_cors_headers(),
            'body': response_body
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendation for {symbol}: {str(e)}")
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': str(e), 'symbol': symbol})
        }

def format_recommendation(item):
    """
    Format DynamoDB item into API response format
    """
    try:
        return {
            'recommendation_id': item.get('recommendation_id'),
            'symbol': item.get('symbol'),
            'recommendation_type': item.get('recommendation_type'),
            'prediction_score': float(item.get('prediction_score', 0)),
            'confidence': float(item.get('confidence', 0)),
            'current_price': float(item.get('current_price', 0)),
            'target_price': float(item.get('target_price', 0)),
            'price_change_prediction': float(item.get('target_price', 0)) - float(item.get('current_price', 0)),
            'risk_level': item.get('risk_level'),
            'ranking': item.get('ranking'),
            'rationale': item.get('rationale', ''),
            'timestamp': item.get('timestamp'),
            'metadata': item.get('metadata', {})
        }
    except Exception as e:
        logger.error(f"Error formatting recommendation: {str(e)}")
        return {
            'error': 'Error formatting recommendation data',
            'raw_item': str(item)
        }

def get_from_cache(key):
    """
    Get data from Redis cache
    """
    try:
        if redis_client:
            return redis_client.get(key)
        return None
    except Exception as e:
        logger.error(f"Error getting from cache: {str(e)}")
        return None

def cache_result(key, data, ttl=300):
    """
    Store data in Redis cache
    """
    try:
        if redis_client:
            redis_client.setex(key, ttl, data)
            logger.debug(f"Cached result with key: {key}")
    except Exception as e:
        logger.error(f"Error caching result: {str(e)}")

def get_cors_headers():
    """
    Get CORS headers for API responses
    """
    return {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
        'Cache-Control': 'no-cache'
    }

def decimal_default(obj):
    """
    JSON serializer for Decimal objects
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")