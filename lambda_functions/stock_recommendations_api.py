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
from typing import Dict, Optional

# OpenTelemetry initialization for SigNoz
def init_otel():
    """Initialize OpenTelemetry for SigNoz Cloud"""
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        from opentelemetry.instrumentation.boto3sqs import Boto3SQSInstrumentor
        from opentelemetry.instrumentation.botocore import BotocoreInstrumentor

        # Get SigNoz configuration from environment
        signoz_endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'https://ingest.us.signoz.cloud:443')
        signoz_headers = os.environ.get('OTEL_EXPORTER_OTLP_HEADERS', '')
        service_name = os.environ.get('OTEL_SERVICE_NAME', 'stock-recommendations-api')

        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.namespace": "stock-analytics",
            "deployment.environment": "development"
        })

        # Configure tracing
        trace_exporter = OTLPSpanExporter(
            endpoint=signoz_endpoint,
            headers=(dict([header.split('=', 1) for header in signoz_headers.split(',') if '=' in header]) if signoz_headers else {}),
            insecure=False
        )

        tracer_provider = TracerProvider(resource=resource)
        span_processor = BatchSpanProcessor(trace_exporter)
        tracer_provider.add_span_processor(span_processor)
        trace.set_tracer_provider(tracer_provider)

        # Configure metrics
        metric_exporter = OTLPMetricExporter(
            endpoint=signoz_endpoint,
            headers=(dict([header.split('=', 1) for header in signoz_headers.split(',') if '=' in header]) if signoz_headers else {}),
            insecure=False
        )

        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        # Set basic trace context propagator
        set_global_textmap(TraceContextTextMapPropagator())

        # Instrument AWS services
        BotocoreInstrumentor().instrument()

        print(f"OpenTelemetry initialized for SigNoz: {signoz_endpoint}")
        return True

    except Exception as e:
        print(f"Failed to initialize OpenTelemetry: {e}")
        return False

# Initialize OTEL
OTEL_ENABLED = init_otel()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')
redis_client = None

VALKEY_ENDPOINT = os.environ.get('VALKEY_ENDPOINT') or os.environ.get('REDIS_ENDPOINT')
cache_client = None
cache_available = False
cache_init_attempted = False

# Configuration
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
REDIS_ENDPOINT = os.environ.get('REDIS_ENDPOINT')
TIME_PREDICTOR_FUNCTION = os.environ.get('TIME_PREDICTOR_FUNCTION', 'time-to-hit-predictor')

# Recommendation expiry configuration (in hours)
RECOMMENDATION_TTL_HOURS = int(os.environ.get('RECOMMENDATION_TTL_HOURS', '24'))  # Default: 24 hours
RECOMMENDATION_MAX_AGE_HOURS = int(os.environ.get('RECOMMENDATION_MAX_AGE_HOURS', '48'))  # Absolute maximum: 48 hours

def _init_cache():
    global cache_client, cache_available, cache_init_attempted
    if cache_init_attempted or not VALKEY_ENDPOINT:
        return
    cache_init_attempted = True
    try:
        import redis  # redis-py works with Valkey
        cache_client = redis.Redis(
            host=VALKEY_ENDPOINT,
            port=6379,
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=2,
            ssl=True,
            ssl_check_hostname=False,
            ssl_cert_reqs=None
        )
        cache_client.ping()
        cache_available = True
        logger.info("Valkey cache initialized")
    except ImportError:
        logger.info("redis/valkey client library not packaged; cache disabled")
    except Exception as e:
        logger.warning(f"Valkey cache init failed ({e}); disabled")

def get_recommendation_cutoff():
    """
    Get cutoff time for active recommendations based on configurable TTL.
    Filters out stale recommendations while preserving historical data.
    """
    now = datetime.utcnow()

    # Primary cutoff: TTL-based expiry for active recommendations
    primary_cutoff = now - timedelta(hours=RECOMMENDATION_TTL_HOURS)

    # Weekend adjustment: if it's Monday and the TTL would exclude Friday data,
    # extend the cutoff to include Friday's close
    current_weekday = now.weekday()  # Monday=0, Sunday=6

    if current_weekday == 0:  # Monday
        # If TTL is less than 72 hours, extend to include Friday's data
        friday_cutoff = now - timedelta(days=3)  # Go back to Friday
        if primary_cutoff > friday_cutoff and RECOMMENDATION_TTL_HOURS < 72:
            cutoff = friday_cutoff
            logger.info(f"Monday adjustment: Extended cutoff to include Friday data")
        else:
            cutoff = primary_cutoff
    else:
        cutoff = primary_cutoff

    # Absolute maximum age safety check
    max_cutoff = now - timedelta(hours=RECOMMENDATION_MAX_AGE_HOURS)
    if cutoff < max_cutoff:
        cutoff = max_cutoff
        logger.info(f"Applied maximum age limit: {RECOMMENDATION_MAX_AGE_HOURS} hours")

    logger.info(f"Recommendation cutoff: {cutoff.isoformat()} (TTL: {RECOMMENDATION_TTL_HOURS}h, Max: {RECOMMENDATION_MAX_AGE_HOURS}h)")
    return cutoff.isoformat()

def get_time_prediction(symbol: str, current_price: float, target_price: float, recommendation_data: Dict) -> Optional[Dict]:
    """
    Get time-to-hit prediction for a recommendation
    """
    try:
        # Check cache first
        cache_key = f"time_prediction:{symbol}:{current_price}:{target_price}"
        cached_prediction = get_from_cache(cache_key)
        if cached_prediction:
            logger.info(f"Returning cached time prediction for {symbol}")
            return json.loads(cached_prediction)
        
        # Call time predictor Lambda function
        payload = {
            'symbol': symbol,
            'current_price': current_price,
            'target_price': target_price,
            'recommendation_data': recommendation_data
        }
        
        response = lambda_client.invoke(
            FunctionName=TIME_PREDICTOR_FUNCTION,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload, default=decimal_default)
        )
        
        if response['StatusCode'] == 200:
            result = json.loads(response['Payload'].read())
            if result.get('statusCode') == 200:
                prediction_data = json.loads(result['body'])
                # The prediction data is directly in the response body
                time_prediction = prediction_data
                
                # Cache the result
                cache_result(cache_key, json.dumps(time_prediction, default=decimal_default), 1800)  # Cache for 30 minutes
                
                logger.info(f"Generated time prediction for {symbol}: {time_prediction.get('expected_timeline', 'unknown')}")
                return time_prediction
            else:
                logger.warning(f"Time predictor returned error for {symbol}: {result.get('body', 'Unknown error')}")
        else:
            logger.error(f"Lambda invocation failed for {symbol}: Status {response['StatusCode']}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting time prediction for {symbol}: {str(e)}")
        return None

def enhance_recommendation_with_time_prediction(recommendation: Dict, include_time_prediction: bool = True) -> Dict:
    """
    Enhance a recommendation with time-to-hit prediction
    """
    try:
        if not include_time_prediction:
            return recommendation
        
        symbol = recommendation.get('symbol')
        current_price = recommendation.get('current_price')
        target_price = recommendation.get('target_price')
        
        if not symbol or not current_price or not target_price:
            logger.warning(f"Missing required data for time prediction: {symbol}")
            return recommendation
        
        # Get time prediction
        time_prediction = get_time_prediction(
            symbol=symbol,
            current_price=float(current_price),
            target_price=float(target_price),
            recommendation_data=recommendation
        )
        
        if time_prediction:
            # Add time prediction to recommendation
            recommendation['time_to_hit_prediction'] = {
                'expected_timeline': time_prediction.get('expected_timeline', 'Unknown'),
                'confidence_level': time_prediction.get('confidence_level', 'medium'),
                'key_factors': time_prediction.get('key_factors', []),
                'risk_factors': time_prediction.get('risk_factors', []),
                'probability_milestones': {
                    '1_week': time_prediction.get('probability_distribution', {}).get('7_days', 0.3),
                    '2_weeks': time_prediction.get('probability_distribution', {}).get('15_days', 0.6),
                    '1_month': time_prediction.get('probability_distribution', {}).get('30_days', 0.8)
                },
                'market_regime': time_prediction.get('market_regime', 'unknown'),
                'prediction_timestamp': time_prediction.get('prediction_timestamp')
            }
            
            # Add user-friendly summary
            expected_timeline = time_prediction.get('expected_timeline', '')
            confidence = time_prediction.get('confidence_level', 'medium')
            
            if 'days' in expected_timeline.lower():
                recommendation['timing_summary'] = f"Expected in {expected_timeline} ({confidence} confidence)"
            else:
                recommendation['timing_summary'] = f"Timeline: {expected_timeline} ({confidence} confidence)"
        else:
            # Add placeholder when prediction fails
            recommendation['time_to_hit_prediction'] = {
                'expected_timeline': 'Analysis pending',
                'confidence_level': 'low',
                'timing_summary': 'Time analysis in progress'
            }
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Error enhancing recommendation with time prediction: {str(e)}")
        return recommendation


def lambda_handler(event, context):
    """
    Main Lambda handler function for API Gateway (REST / HTTP API) or test invoke.
    """
    # Create trace for this request if OTEL is enabled
    tracer = None
    span = None

    if OTEL_ENABLED:
        try:
            from opentelemetry import trace
            tracer = trace.get_tracer(__name__)
            span = tracer.start_span("lambda_handler")
            span.set_attribute("aws.lambda.function_name", context.function_name if context else "unknown")
            span.set_attribute("aws.lambda.request_id", context.aws_request_id if context else "unknown")
        except Exception as e:
            logger.warning(f"Failed to create OTEL span: {e}")

    try:
        _init_cache()

        # Support test invokes with none/empty event
        if event is None:
            event = {}

        # Derive HTTP method & path robustly
        http_method = (
            event.get('httpMethod') or
            event.get('requestContext', {}).get('http', {}).get('method') or
            'GET'
        )
        path = (
            event.get('path') or
            event.get('rawPath') or
            '/'
        )
        query_params = event.get('queryStringParameters') or {}

        # CORS preflight
        if http_method == 'OPTIONS':
            return {
                'statusCode': 204,
                'headers': get_cors_headers(),
                'body': ''
            }

        # Simple health endpoint
        if path in ['/', '/health']:
            return {
                'statusCode': 200,
                'headers': get_cors_headers(),
                'body': json.dumps({
                    'status': 'ok',
                    'cache': 'ready' if cache_available else 'disabled',
                    'timestamp': datetime.utcnow().isoformat()
                })
            }
        if path.startswith('/recommendations/') and len(path.split('/')) == 3:
            symbol = path.split('/')[2].upper()
            return get_recommendation_by_symbol(symbol, query_params)

        if path == '/recommendations':
            return get_recommendations(query_params)

        return {
            'statusCode': 404,
            'headers': get_cors_headers(),
            'body': json.dumps({'error': 'Not found', 'path': path})
        }

    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
        return {
            'statusCode': 500,
            'headers': get_cors_headers(),
            'body': json.dumps({
                'error': 'Internal server error',
                'timestamp': datetime.utcnow().isoformat()
            })
        }
    finally:
        # Close the OTEL span
        if span:
            try:
                span.end()
            except Exception as e:
                logger.warning(f"Failed to end OTEL span: {e}")

def get_from_cache(key):
    """
    Get data from Valkey/Redis cache
    """
    try:
        if cache_available and cache_client:
            return cache_client.get(key)
        return None
    except Exception as e:
        logger.error(f"Error getting from cache: {str(e)}")
        return None

def cache_result(key, data, ttl=300):
    """
    Store data in Valkey/Redis cache
    """
    try:
        if cache_available and cache_client:
            cache_client.setex(key, ttl, data)
            logger.debug(f"Cached result {key}")
    except Exception as e:
        logger.error(f"Error caching result: {str(e)}")

def get_recommendations(query_params):
    """
    Get stock recommendations with optional filtering
    """
    try:
        # Parse query parameters
        limit = int(query_params.get('limit', 50))
        limit = min(limit, 50)  # Cap at 50
        
        recommendation_type = query_params.get('type', '').upper()
        risk_level = query_params.get('risk', '').upper()
        min_confidence = float(query_params.get('min_confidence', 0.0))
        include_time_prediction = query_params.get('include_time_prediction', 'true').lower() == 'true'
        
        # Check cache first
        cache_key = f"recommendations:{recommendation_type}:{risk_level}:{min_confidence}:{limit}:{include_time_prediction}"
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
        
        # Get active recommendations (within TTL)
        cutoff_time = get_recommendation_cutoff()
        
        # Scan with filters (in production, you'd want to use GSI for better performance)
        # Explicitly filter for proper recommendations by requiring essential fields
        filter_expression = "attribute_exists(recommendation_id) AND attribute_exists(recommendation_type) AND attribute_exists(prediction_score) AND #ts > :cutoff"
        expression_values = {':cutoff': cutoff_time}
        expression_names = {'#ts': 'timestamp'}
        
        logger.info(f"Filter expression: {filter_expression}")
        logger.info(f"Expression values: {expression_values}")
        
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
            ExpressionAttributeValues=expression_values
        )
        
        logger.info(f"DynamoDB scan response: Count={response.get('Count', 0)}, ScannedCount={response.get('ScannedCount', 0)}")
        items = response.get('Items', [])
        logger.info(f"Items retrieved: {len(items)}")
        
        # Log first few items for debugging
        if items:
            logger.info(f"First item keys: {list(items[0].keys())}")
            logger.info(f"First item sample: {dict(list(items[0].items())[:3])}")
        else:
            logger.info("No items returned from DynamoDB scan")
        
        # Sort by prediction score and ranking
        items.sort(key=lambda x: (float(x.get('prediction_score', 0)), x.get('ranking', 999)), reverse=True)
        
        # Limit results
        items = items[:limit]
        
        # Format response
        recommendations = []
        for item in items:
            rec = format_recommendation(item)
            # Enhance with time prediction if requested
            rec = enhance_recommendation_with_time_prediction(rec, include_time_prediction)
            recommendations.append(rec)
        
        response_data = {
            'recommendations': recommendations,
            'count': len(recommendations),
            'filters_applied': {
                'type': recommendation_type or 'all',
                'risk_level': risk_level or 'all',
                'min_confidence': min_confidence,
                'limit': limit,
                'time_predictions_included': include_time_prediction,
                'recommendation_ttl_hours': RECOMMENDATION_TTL_HOURS,
                'max_age_hours': RECOMMENDATION_MAX_AGE_HOURS,
                'cutoff_time': cutoff_time
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
            'body': json.dumps({'error': str(e)}, default=decimal_default)
        }

def get_recommendation_by_symbol(symbol, query_params):
    """
    Get recommendation for specific stock symbol
    """
    try:
        # Parse query parameters
        include_time_prediction = query_params.get('include_time_prediction', 'true').lower() == 'true'
        # Check cache first  
        cache_key = f"recommendation:{symbol}:{include_time_prediction}"
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
        
        # Get active recommendations for this symbol (within TTL)
        cutoff_time = get_recommendation_cutoff()
        
        response = table.scan(
            FilterExpression="symbol = :symbol AND #ts > :cutoff AND attribute_exists(recommendation_type) AND attribute_exists(prediction_score)",
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
        # Enhance with time prediction if requested
        recommendation = enhance_recommendation_with_time_prediction(recommendation, include_time_prediction)
        
        # Add historical data if requested
        include_history = query_params.get('include_history', 'false').lower() == 'true'
        if include_history:
            # Sort by timestamp, most recent first
            historical_items = sorted(items, key=lambda x: x['timestamp'], reverse=True)
            # Format historical items (without time predictions to avoid overload)
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
            'body': json.dumps({'error': str(e), 'symbol': symbol}, default=decimal_default)
        }

def format_recommendation(item):
    """
    Format DynamoDB item into API response format
    """
    try:
        return {
            'recommendation_id': item.get('recommendation_id'),
            'symbol': item.get('symbol'),
            'prediction': item.get('recommendation_type'),  # Map recommendation_type to prediction field for Swagger spec
            'recommendation_type': item.get('recommendation_type'),  # Keep for backward compatibility
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