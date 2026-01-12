"""
Flask application wrapper for Stock Analytics Engine Lambda functions.
Converts AWS Lambda functions to REST API endpoints for Railway deployment.

Endpoints:
    Core:
        GET  /                           - API information
        GET  /health                     - Health check
        GET  /recommendations            - All stock recommendations
        GET  /recommendations/{symbol}   - Single symbol recommendation
        POST /custom-request             - Custom analysis

    Analytics (Legacy):
        GET  /analytics/dashboard        - Analytics dashboard
        GET  /analytics/detailed         - Detailed analytics

    Accuracy Tracking (New):
        GET  /analytics/calibration      - Confidence calibration report
        GET  /analytics/symbols          - Symbol-level accuracy rankings
        GET  /analytics/symbols/{symbol} - Specific symbol performance
        GET  /analytics/deployment-gates - Deployment gate decisions
        GET  /analytics/error-distribution - Error magnitude analysis
        GET  /analytics/audit-trail      - Accuracy audit log
        GET  /analytics/market-correlation - Accuracy by market condition
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from datetime import datetime

# Set AWS region for boto3 compatibility (even though we use PostgreSQL on Railway)
# This prevents import errors from Lambda functions that still have boto3 imports
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ.setdefault('AWS_REGION', 'us-east-1')

# Add lambda_functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Lambda function handlers with error handling for Railway compatibility
try:
    from lambda_functions import stock_recommendations_api
    from lambda_functions import dual_prediction_reporting_api
    LAMBDA_FUNCTIONS_AVAILABLE = True
except Exception as e:
    LAMBDA_FUNCTIONS_AVAILABLE = False
    stock_recommendations_api = None
    dual_prediction_reporting_api = None
    print(f"Warning: Lambda functions not fully available: {e}")

# Import shared utilities
from lambda_functions.shared.lambda_utils import LambdaResponse
from lambda_functions.shared.error_handling import StructuredLogger

# Import accuracy tracking modules (these use PostgreSQL, fully Railway-compatible)
from lambda_functions.shared.accuracy_tracking import (
    ConfidenceCalibrationTracker,
    SymbolAccuracyAggregator,
    DeploymentGate,
    AccuracyAuditLogger,
    MarketConditionTracker,
    ErrorDistributionAnalyzer
)

# Import database for health checks
from lambda_functions.shared.database import db, is_database_available

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
    status = {
        'status': 'healthy',
        'service': 'stock-analytics-api',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'database': 'not_configured'
    }

    # Check database health if available
    if is_database_available():
        try:
            db_health = db.health_check()
            status['database'] = 'healthy' if db_health.get('healthy') else 'unhealthy'
            status['database_latency_ms'] = db_health.get('latency_ms')
        except Exception as e:
            status['database'] = 'error'
            status['database_error'] = str(e)

    return jsonify(status), 200


# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Stock Analytics Engine API',
        'version': '2.0.0',
        'platform': 'Railway',
        'endpoints': {
            'core': {
                'recommendations': '/recommendations',
                'recommendations_by_symbol': '/recommendations/{symbol}',
                'custom_request': '/custom-request'
            },
            'analytics': {
                'dashboard': '/analytics/dashboard',
                'detailed': '/analytics/detailed',
                'calibration': '/analytics/calibration',
                'symbols': '/analytics/symbols',
                'symbol_detail': '/analytics/symbols/{symbol}',
                'deployment_gates': '/analytics/deployment-gates',
                'error_distribution': '/analytics/error-distribution',
                'audit_trail': '/analytics/audit-trail',
                'market_correlation': '/analytics/market-correlation'
            }
        },
        'documentation': 'https://github.com/your-repo/stock-analytics-engine'
    }), 200

# Stock recommendations endpoints
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get all stock recommendations."""
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        return jsonify({
            'error': 'Lambda functions not available',
            'message': 'Use /analytics endpoints for PostgreSQL-based data'
        }), 503

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
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        return jsonify({
            'error': 'Lambda functions not available',
            'message': 'Use /analytics endpoints for PostgreSQL-based data'
        }), 503

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
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        return jsonify({
            'error': 'Legacy analytics not available',
            'message': 'Use /analytics/calibration, /analytics/symbols, etc. for PostgreSQL-based analytics'
        }), 503

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
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        return jsonify({
            'error': 'Legacy analytics not available',
            'message': 'Use /analytics/calibration, /analytics/symbols, etc. for PostgreSQL-based analytics'
        }), 503

    event = convert_flask_to_lambda_event(request)
    event['path'] = '/analytics/detailed'  # Update path for routing
    context = MockLambdaContext()

    try:
        lambda_response = dual_prediction_reporting_api.lambda_handler(event, context)
        return convert_lambda_to_flask_response(lambda_response)
    except Exception as e:
        logger.log_error(f"Error in detailed analytics endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Custom stock request endpoint (disabled - module not available)
@app.route('/custom-request', methods=['POST'])
def custom_request():
    """Process custom stock analysis request (not implemented on Railway)."""
    return jsonify({
        'error': 'Feature not available',
        'message': 'Custom requests are not supported in Railway deployment'
    }), 501


# ============================================================
# ACCURACY TRACKING ENDPOINTS
# ============================================================

@app.route('/analytics/calibration', methods=['GET'])
def analytics_calibration():
    """Get confidence calibration metrics (ECE)."""
    try:
        lookback_days = request.args.get('lookback_days', 30, type=int)
        model_type = request.args.get('model_type', 'price_prediction')

        tracker = ConfidenceCalibrationTracker()
        result = tracker.get_calibration_report(model_type=model_type, lookback_days=lookback_days)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in calibration endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/symbols', methods=['GET'])
def analytics_symbols():
    """Get symbol-level accuracy rankings."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')
        limit = request.args.get('limit', 50, type=int)

        aggregator = SymbolAccuracyAggregator()
        result = aggregator.get_symbol_ranking(model_type=model_type, limit=limit)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in symbols endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/symbols/<symbol>', methods=['GET'])
def analytics_symbol_detail(symbol):
    """Get detailed accuracy for specific symbol."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')

        aggregator = SymbolAccuracyAggregator()
        result = aggregator.get_symbol_detail(symbol.upper(), model_type=model_type)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in symbol detail endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/deployment-gates', methods=['GET'])
def analytics_deployment_gates():
    """Get recent deployment gate decisions."""
    try:
        limit = request.args.get('limit', 10, type=int)
        model_type = request.args.get('model_type')

        gate = DeploymentGate()
        result = gate.get_recent_decisions(model_type=model_type, limit=limit)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in deployment gates endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/deployment-gates/latest', methods=['GET'])
def analytics_latest_gate():
    """Get latest deployment gate status."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')

        gate = DeploymentGate()
        result = gate.get_latest_gate(model_type=model_type)

        if result:
            return jsonify(result), 200
        return jsonify({'error': 'No gate data available'}), 404
    except Exception as e:
        logger.log_error(f"Error in latest gate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/deployment-gates/statistics', methods=['GET'])
def analytics_gate_statistics():
    """Get deployment gate statistics."""
    try:
        lookback_days = request.args.get('lookback_days', 30, type=int)

        gate = DeploymentGate()
        result = gate.get_gate_statistics(lookback_days=lookback_days)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in gate statistics endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/error-distribution', methods=['GET'])
def analytics_error_distribution():
    """Get error magnitude distribution."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')

        analyzer = ErrorDistributionAnalyzer()
        result = analyzer.get_distribution(model_type=model_type)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in error distribution endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/error-distribution/trend', methods=['GET'])
def analytics_error_trend():
    """Get error distribution trend."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')
        days = request.args.get('days', 14, type=int)

        analyzer = ErrorDistributionAnalyzer()
        result = analyzer.get_error_trend(model_type=model_type, days=days)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in error trend endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/audit-trail', methods=['GET'])
def analytics_audit_trail():
    """Get accuracy audit trail."""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        model_type = request.args.get('model_type')
        event_type = request.args.get('event_type')
        limit = request.args.get('limit', 100, type=int)

        audit_logger = AccuracyAuditLogger()
        result = audit_logger.get_audit_trail(
            start_date=start_date,
            end_date=end_date,
            model_type=model_type,
            event_type=event_type,
            limit=limit
        )

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in audit trail endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/audit-trail/verify', methods=['GET'])
def analytics_verify_claim():
    """Verify historical accuracy claim."""
    try:
        timestamp = request.args.get('timestamp')
        model_type = request.args.get('model_type', 'price_prediction')

        if not timestamp:
            return jsonify({'error': 'timestamp parameter required'}), 400

        audit_logger = AccuracyAuditLogger()
        result = audit_logger.verify_historical_claim(timestamp, model_type)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in verify claim endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/market-correlation', methods=['GET'])
def analytics_market_correlation():
    """Get accuracy by market condition."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')

        tracker = MarketConditionTracker()
        result = tracker.get_correlation_report(model_type=model_type)

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in market correlation endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/market-conditions', methods=['GET'])
def analytics_market_conditions():
    """Get current market conditions."""
    try:
        tracker = MarketConditionTracker()
        result = tracker.get_current_conditions()

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(f"Error in market conditions endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/market-conditions/history', methods=['GET'])
def analytics_market_history():
    """Get market regime history."""
    try:
        days = request.args.get('days', 30, type=int)

        tracker = MarketConditionTracker()
        result = tracker.get_regime_history(days=days)

        return jsonify({'history': result, 'days': days}), 200
    except Exception as e:
        logger.log_error(f"Error in market history endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/retraining-candidates', methods=['GET'])
def analytics_retraining_candidates():
    """Get symbols that need retraining."""
    try:
        model_type = request.args.get('model_type', 'price_prediction')

        aggregator = SymbolAccuracyAggregator()
        candidates = aggregator.identify_retraining_candidates(model_type=model_type)

        return jsonify({
            'candidates': candidates,
            'count': len(candidates),
            'model_type': model_type,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.log_error(f"Error in retraining candidates endpoint: {str(e)}")
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
