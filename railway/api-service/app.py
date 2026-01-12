"""
Flask API for Stock Analytics Engine - Railway-native implementation.
No AWS dependencies - uses PostgreSQL for all data storage.

Endpoints:
    Core:
        GET  /                           - API information
        GET  /health                     - Health check
        GET  /recommendations            - All stock recommendations
        GET  /recommendations/{symbol}   - Single symbol recommendation
        GET  /predictions/price          - Price predictions
        GET  /predictions/time           - Time-to-hit predictions
        GET  /predictions/dashboard      - Predictions dashboard

    Accuracy Tracking:
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
from flask import Flask, request, jsonify
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Railway-native services (no AWS dependencies)
from services.recommendations import RecommendationsService, get_service as get_recommendations_service
from services.predictions import PredictionsService, get_service as get_predictions_service

# Import shared utilities
from lambda_functions.shared.error_handling import StructuredLogger

# Import accuracy tracking modules (PostgreSQL-based)
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


# ============================================================
# CORE ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Railway."""
    status = {
        'status': 'healthy',
        'service': 'stock-analytics-api',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0-railway',
        'platform': 'Railway',
        'database': 'not_configured'
    }

    # Check database health
    if is_database_available():
        try:
            db_health = db.health_check()
            status['database'] = 'healthy' if db_health.get('healthy') else 'unhealthy'
            status['database_latency_ms'] = db_health.get('latency_ms')
        except Exception as e:
            status['database'] = 'error'
            status['database_error'] = str(e)

    return jsonify(status), 200


@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Stock Analytics Engine API',
        'version': '3.0.0-railway',
        'platform': 'Railway (PostgreSQL)',
        'aws_dependencies': False,
        'endpoints': {
            'core': {
                'health': '/health',
                'recommendations': '/recommendations',
                'recommendations_by_symbol': '/recommendations/{symbol}',
                'price_predictions': '/predictions/price',
                'time_predictions': '/predictions/time',
                'predictions_dashboard': '/predictions/dashboard'
            },
            'analytics': {
                'calibration': '/analytics/calibration',
                'symbols': '/analytics/symbols',
                'symbol_detail': '/analytics/symbols/{symbol}',
                'deployment_gates': '/analytics/deployment-gates',
                'error_distribution': '/analytics/error-distribution',
                'audit_trail': '/analytics/audit-trail',
                'market_correlation': '/analytics/market-correlation',
                'retraining_candidates': '/analytics/retraining-candidates'
            }
        }
    }), 200


# ============================================================
# RECOMMENDATIONS ENDPOINTS (Railway-native)
# ============================================================

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get all stock recommendations."""
    try:
        limit = request.args.get('limit', 100, type=int)
        service = get_recommendations_service()
        result = service.get_all_recommendations(limit=limit)

        if result['success']:
            return jsonify(result), 200
        return jsonify(result), 500

    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_recommendations'})
        return jsonify({'error': str(e)}), 500


@app.route('/recommendations/<symbol>', methods=['GET'])
def get_recommendation_by_symbol(symbol):
    """Get recommendation for specific stock symbol."""
    try:
        service = get_recommendations_service()
        result = service.get_recommendation_by_symbol(symbol.upper())

        if result['success']:
            return jsonify(result), 200
        return jsonify(result), 404

    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_recommendation_by_symbol', 'symbol': symbol})
        return jsonify({'error': str(e)}), 500


@app.route('/recommendations/buy', methods=['GET'])
def get_buy_recommendations():
    """Get all BUY recommendations."""
    try:
        limit = request.args.get('limit', 50, type=int)
        service = get_recommendations_service()
        result = service.get_recommendations_by_type('BUY', limit=limit)
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_buy_recommendations'})
        return jsonify({'error': str(e)}), 500


@app.route('/recommendations/sell', methods=['GET'])
def get_sell_recommendations():
    """Get all SELL recommendations."""
    try:
        limit = request.args.get('limit', 50, type=int)
        service = get_recommendations_service()
        result = service.get_recommendations_by_type('SELL', limit=limit)
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_sell_recommendations'})
        return jsonify({'error': str(e)}), 500


@app.route('/recommendations/high-confidence', methods=['GET'])
def get_high_confidence_recommendations():
    """Get high confidence recommendations."""
    try:
        min_confidence = request.args.get('min_confidence', 0.7, type=float)
        limit = request.args.get('limit', 20, type=int)
        service = get_recommendations_service()
        result = service.get_high_confidence_recommendations(
            min_confidence=min_confidence,
            limit=limit
        )
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_high_confidence_recommendations'})
        return jsonify({'error': str(e)}), 500


# ============================================================
# PREDICTIONS ENDPOINTS (Railway-native)
# ============================================================

@app.route('/predictions/price', methods=['GET'])
def get_price_predictions():
    """Get price predictions."""
    try:
        symbol = request.args.get('symbol')
        status = request.args.get('status')
        limit = request.args.get('limit', 100, type=int)

        service = get_predictions_service()
        result = service.get_price_predictions(
            symbol=symbol,
            status=status,
            limit=limit
        )
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_price_predictions'})
        return jsonify({'error': str(e)}), 500


@app.route('/predictions/time', methods=['GET'])
def get_time_predictions():
    """Get time-to-hit predictions."""
    try:
        symbol = request.args.get('symbol')
        status = request.args.get('status')
        limit = request.args.get('limit', 100, type=int)

        service = get_predictions_service()
        result = service.get_time_predictions(
            symbol=symbol,
            status=status,
            limit=limit
        )
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_time_predictions'})
        return jsonify({'error': str(e)}), 500


@app.route('/predictions/pending', methods=['GET'])
def get_pending_validations():
    """Get predictions pending validation."""
    try:
        service = get_predictions_service()
        result = service.get_pending_validations()
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'get_pending_validations'})
        return jsonify({'error': str(e)}), 500


@app.route('/predictions/dashboard', methods=['GET'])
def predictions_dashboard():
    """Get predictions analytics dashboard."""
    try:
        service = get_predictions_service()
        result = service.get_analytics_dashboard()
        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'predictions_dashboard'})
        return jsonify({'error': str(e)}), 500


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
        logger.log_error(e, context={'endpoint': 'analytics_calibration'})
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
        logger.log_error(e, context={'endpoint': 'analytics_symbols'})
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
        logger.log_error(e, context={'endpoint': 'analytics_symbol_detail', 'symbol': symbol})
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
        logger.log_error(e, context={'endpoint': 'analytics_deployment_gates'})
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
        logger.log_error(e, context={'endpoint': 'analytics_error_distribution'})
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
        logger.log_error(e, context={'endpoint': 'analytics_audit_trail'})
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
        logger.log_error(e, context={'endpoint': 'analytics_market_correlation'})
        return jsonify({'error': str(e)}), 500


@app.route('/analytics/market-conditions', methods=['GET'])
def analytics_market_conditions():
    """Get current market conditions."""
    try:
        tracker = MarketConditionTracker()
        result = tracker.get_current_conditions()

        return jsonify(result), 200
    except Exception as e:
        logger.log_error(e, context={'endpoint': 'analytics_market_conditions'})
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
        logger.log_error(e, context={'endpoint': 'analytics_retraining_candidates'})
        return jsonify({'error': str(e)}), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.log_error(Exception(str(error)), context={'handler': '500_error'})
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    logger.log_info(f"Starting Stock Analytics API (Railway-native) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
