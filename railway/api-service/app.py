"""
Flask application wrapper for Stock Analytics Engine Lambda functions.
Converts AWS Lambda functions to REST API endpoints for Railway deployment.
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from datetime import datetime

# Add lambda_functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Lambda function handlers
from lambda_functions import stock_recommendations_api
from lambda_functions import dual_prediction_reporting_api
from lambda_functions import custom_stock_request_api

# Import shared utilities
from lambda_functions.shared.lambda_utils import LambdaResponse
from lambda_functions.shared.error_handling import StructuredLogger

app = Flask(__name__)
logger = StructuredLogger(__name__)

# Mock Lambda context for compatibility
class MockLambdaContext:
    def __init__(self):
        self.function_name = "railway-api-service"
        self.memory_limit_in_mb = 2048
        self.invoked_function_arn = "arn:aws:lambda:railway:api-service"
        self.aws_request_id = None

    def get_remaining_time_in_millis(self):
        return 300000  # 5 minutes

def convert_flask_to_lambda_event(request_obj, path_params=None):
    """Convert Flask request to AWS Lambda event format."""
    event = {
        'httpMethod': request_obj.method,
        'path': request_obj.path,
        'headers': dict(request_obj.headers),
        'queryStringParameters': dict(request_obj.args) if request_obj.args else None,
        'pathParameters': path_params,
        'body': request_obj.get_data(as_text=True) if request_obj.data else None,
        'isBase64Encoded': False,
        'requestContext': {
            'requestId': request_obj.environ.get('REQUEST_ID', 'railway-request'),
            'identity': {
                'sourceIp': request_obj.remote_addr
            }
        }
    }
    return event

def convert_lambda_to_flask_response(lambda_response):
    """Convert AWS Lambda response to Flask response."""
    if isinstance(lambda_response, dict):
        status_code = lambda_response.get('statusCode', 200)
        body = lambda_response.get('body', '{}')
        headers = lambda_response.get('headers', {})

        # Parse body if it's a string
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass

        response = jsonify(body)
        response.status_code = status_code

        # Add headers
        for key, value in headers.items():
            response.headers[key] = value

        return response
    else:
        return jsonify({'error': 'Invalid response format'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Railway."""
    return jsonify({
        'status': 'healthy',
        'service': 'stock-analytics-api',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Stock Analytics Engine API',
        'version': '1.0.0',
        'endpoints': {
            'recommendations': '/recommendations',
            'recommendations_by_symbol': '/recommendations/{symbol}',
            'analytics_dashboard': '/analytics/dashboard',
            'analytics_detailed': '/analytics/detailed',
            'custom_request': '/custom-request'
        },
        'documentation': 'https://github.com/your-repo/stock-analytics-engine'
    }), 200

# Stock recommendations endpoints
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get all stock recommendations."""
    event = convert_flask_to_lambda_event(request)
    context = MockLambdaContext()

    try:
        lambda_response = stock_recommendations_api.lambda_handler(event, context)
        return convert_lambda_to_flask_response(lambda_response)
    except Exception as e:
        logger.log_error(f"Error in recommendations endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations/<symbol>', methods=['GET'])
def get_recommendation_by_symbol(symbol):
    """Get recommendation for specific stock symbol."""
    event = convert_flask_to_lambda_event(request, path_params={'symbol': symbol.upper()})
    context = MockLambdaContext()

    try:
        lambda_response = stock_recommendations_api.lambda_handler(event, context)
        return convert_lambda_to_flask_response(lambda_response)
    except Exception as e:
        logger.log_error(f"Error in recommendation by symbol endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Analytics endpoints
@app.route('/analytics/dashboard', methods=['GET'])
def analytics_dashboard():
    """Get analytics dashboard data."""
    event = convert_flask_to_lambda_event(request)
    context = MockLambdaContext()

    try:
        lambda_response = dual_prediction_reporting_api.lambda_handler(event, context)
        return convert_lambda_to_flask_response(lambda_response)
    except Exception as e:
        logger.log_error(f"Error in analytics dashboard endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics/detailed', methods=['GET'])
def analytics_detailed():
    """Get detailed analytics data."""
    event = convert_flask_to_lambda_event(request)
    event['path'] = '/analytics/detailed'  # Update path for routing
    context = MockLambdaContext()

    try:
        lambda_response = dual_prediction_reporting_api.lambda_handler(event, context)
        return convert_lambda_to_flask_response(lambda_response)
    except Exception as e:
        logger.log_error(f"Error in detailed analytics endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Custom stock request endpoint
@app.route('/custom-request', methods=['POST'])
def custom_request():
    """Process custom stock analysis request."""
    event = convert_flask_to_lambda_event(request)
    context = MockLambdaContext()

    try:
        lambda_response = custom_stock_request_api.lambda_handler(event, context)
        return convert_lambda_to_flask_response(lambda_response)
    except Exception as e:
        logger.log_error(f"Error in custom request endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.log_error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    logger.log_info(f"Starting Stock Analytics API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
