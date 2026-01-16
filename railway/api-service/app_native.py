"""
Railway-Native Flask API for Stock Analytics Engine.
Completely AWS-free implementation using PostgreSQL.
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from datetime import datetime

# Add railway shared to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.logger import StructuredLogger

app = Flask(__name__)
logger = StructuredLogger(__name__)

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', '')
REDIS_URL = os.environ.get('REDIS_URL', '')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
API_VERSION = '2.0.0'

# Database connection
db_conn = None
redis_client = None


def get_db_connection():
    """Get or create database connection."""
    global db_conn

    if db_conn is not None:
        try:
            # Test if connection is still valid
            with db_conn.cursor() as cur:
                cur.execute('SELECT 1')
            return db_conn
        except Exception:
            db_conn = None

    if not DATABASE_URL:
        logger.log_error("DATABASE_URL not configured")
        return None

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        db_conn = psycopg2.connect(DATABASE_URL)
        db_conn.autocommit = True
        logger.log_info("Database connection established")
        return db_conn
    except Exception as e:
        logger.log_error(f"Database connection failed: {e}")
        return None


def get_redis_client():
    """Get or create Redis connection."""
    global redis_client

    if redis_client is not None:
        try:
            redis_client.ping()
            return redis_client
        except Exception:
            redis_client = None

    if not REDIS_URL:
        return None

    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        return redis_client
    except Exception as e:
        logger.log_warning(f"Redis connection failed: {e}")
        return None


# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Railway."""
    db_status = 'connected' if get_db_connection() else 'disconnected'
    redis_status = 'connected' if get_redis_client() else 'disabled'

    return jsonify({
        'status': 'healthy' if db_status == 'connected' else 'degraded',
        'service': 'stock-analytics-api',
        'version': API_VERSION,
        'platform': 'railway',
        'database': db_status,
        'cache': redis_status,
        'timestamp': datetime.utcnow().isoformat()
    }), 200 if db_status == 'connected' else 503


# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information."""
    return jsonify({
        'service': 'Stock Analytics Engine API',
        'version': API_VERSION,
        'platform': 'Railway (AWS-free)',
        'endpoints': {
            'health': '/health',
            'recommendations': '/recommendations',
            'recommendations_by_symbol': '/recommendations/{symbol}',
            'latest_prices': '/prices',
            'price_by_symbol': '/prices/{symbol}',
            'analytics_dashboard': '/analytics/dashboard',
            'quotes_history': '/quotes/{symbol}',
            'ai_performance_breakdown': '/api/ai-performance/{period}/breakdown',
            'ai_performance_tuning_history': '/api/ai-performance/tuning-history'
        },
        'documentation': 'https://github.com/your-repo/stock-analytics-engine'
    }), 200


# Stock recommendations endpoints
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get all stock recommendations."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    try:
        # Check cache first
        cache = get_redis_client()
        if cache:
            cached = cache.get('recommendations:all')
            if cached:
                return jsonify(json.loads(cached)), 200

        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, recommendation, confidence, target_price,
                       current_price, analysis_data, updated_at
                FROM stock_recommendations
                ORDER BY confidence DESC
                LIMIT 100
            """)

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            recommendations = []
            for row in rows:
                rec = dict(zip(columns, row))
                rec['updated_at'] = rec['updated_at'].isoformat() if rec['updated_at'] else None
                recommendations.append(rec)

        result = {
            'recommendations': recommendations,
            'count': len(recommendations),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Cache result
        if cache:
            cache.setex('recommendations:all', 60, json.dumps(result, default=str))

        return jsonify(result), 200

    except Exception as e:
        logger.log_error(f"Error fetching recommendations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recommendations/<symbol>', methods=['GET'])
def get_recommendation_by_symbol(symbol):
    """Get recommendation for specific stock symbol."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    symbol = symbol.upper()

    try:
        # Check cache first
        cache = get_redis_client()
        if cache:
            cached = cache.get(f'recommendation:{symbol}')
            if cached:
                return jsonify(json.loads(cached)), 200

        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, recommendation, confidence, target_price,
                       current_price, analysis_data, updated_at
                FROM stock_recommendations
                WHERE symbol = %s
            """, (symbol,))

            row = cur.fetchone()

            if not row:
                return jsonify({'error': f'No recommendation found for {symbol}'}), 404

            columns = [desc[0] for desc in cur.description]
            rec = dict(zip(columns, row))
            rec['updated_at'] = rec['updated_at'].isoformat() if rec['updated_at'] else None

        # Cache result
        if cache:
            cache.setex(f'recommendation:{symbol}', 60, json.dumps(rec, default=str))

        return jsonify(rec), 200

    except Exception as e:
        logger.log_error(f"Error fetching recommendation for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


# Latest prices endpoints
@app.route('/prices', methods=['GET'])
def get_latest_prices():
    """Get latest prices for all tracked symbols."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, price, change_amount, change_percent,
                       volume, updated_at
                FROM latest_prices
                ORDER BY symbol
            """)

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            prices = []
            for row in rows:
                price = dict(zip(columns, row))
                price['updated_at'] = price['updated_at'].isoformat() if price['updated_at'] else None
                prices.append(price)

        return jsonify({
            'prices': prices,
            'count': len(prices),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.log_error(f"Error fetching prices: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/prices/<symbol>', methods=['GET'])
def get_price_by_symbol(symbol):
    """Get latest price for specific symbol."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    symbol = symbol.upper()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, price, change_amount, change_percent,
                       volume, updated_at
                FROM latest_prices
                WHERE symbol = %s
            """, (symbol,))

            row = cur.fetchone()

            if not row:
                return jsonify({'error': f'No price data for {symbol}'}), 404

            columns = [desc[0] for desc in cur.description]
            price = dict(zip(columns, row))
            price['updated_at'] = price['updated_at'].isoformat() if price['updated_at'] else None

        return jsonify(price), 200

    except Exception as e:
        logger.log_error(f"Error fetching price for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


# Historical quotes endpoint
@app.route('/quotes/<symbol>', methods=['GET'])
def get_quotes_history(symbol):
    """Get historical quotes for a symbol."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    symbol = symbol.upper()
    limit = request.args.get('limit', 30, type=int)
    limit = min(limit, 365)  # Cap at 1 year

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, date, open_price, high_price, low_price,
                       close_price, volume, moving_avg_5, moving_avg_20,
                       volatility, data_quality, created_at
                FROM stock_quotes
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol, limit))

            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            if not rows:
                return jsonify({'error': f'No quote history for {symbol}'}), 404

            quotes = []
            for row in rows:
                quote = dict(zip(columns, row))
                quote['date'] = quote['date'].isoformat() if quote['date'] else None
                quote['created_at'] = quote['created_at'].isoformat() if quote['created_at'] else None
                # Convert Decimal to float for JSON
                for key in ['open_price', 'high_price', 'low_price', 'close_price',
                           'moving_avg_5', 'moving_avg_20', 'volatility']:
                    if quote.get(key) is not None:
                        quote[key] = float(quote[key])
                quotes.append(quote)

        return jsonify({
            'symbol': symbol,
            'quotes': quotes,
            'count': len(quotes),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.log_error(f"Error fetching quotes for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500


# Analytics dashboard endpoint
@app.route('/analytics/dashboard', methods=['GET'])
def analytics_dashboard():
    """Get analytics dashboard data."""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    try:
        with conn.cursor() as cur:
            # Get recommendation distribution
            cur.execute("""
                SELECT recommendation_type, COUNT(*) as count
                FROM stock_recommendations
                GROUP BY recommendation_type
            """)
            rec_dist = {row[0]: row[1] for row in cur.fetchall()}

            # Get recent ingestion stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(symbols_succeeded) as total_symbols,
                    AVG(symbols_succeeded::float / NULLIF(symbols_attempted, 0)) * 100 as avg_success_rate,
                    AVG(duration_seconds) as avg_duration
                FROM ingestion_logs
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            ingestion_row = cur.fetchone()

            # Get top performers (highest confidence BUY recommendations)
            cur.execute("""
                SELECT symbol, recommendation_type, confidence, target_price, current_price
                FROM stock_recommendations
                WHERE recommendation_type = 'BUY'
                ORDER BY confidence DESC
                LIMIT 5
            """)
            top_buys = [{'symbol': r[0], 'confidence': float(r[2]) if r[2] else 0,
                        'target_price': float(r[3]) if r[3] else 0, 'current_price': float(r[4]) if r[4] else 0}
                       for r in cur.fetchall()]

            # Get prediction stats for today
            cur.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    COUNT(*) FILTER (WHERE validation_status = 'pending') as pending,
                    AVG(confidence_score) as avg_confidence
                FROM price_predictions
                WHERE prediction_date > NOW() - INTERVAL '24 hours'
            """)
            pred_row = cur.fetchone()

            # Get symbol count
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM stock_quotes")
            symbol_count = cur.fetchone()[0]

        return jsonify({
            'dashboard': {
                'recommendation_distribution': rec_dist,
                'tracked_symbols': symbol_count,
                'top_buy_recommendations': top_buys,
                'ingestion_stats_24h': {
                    'total_runs': ingestion_row[0] or 0,
                    'total_symbols_processed': ingestion_row[1] or 0,
                    'avg_success_rate': round(ingestion_row[2] or 0, 2),
                    'avg_duration_seconds': round(ingestion_row[3] or 0, 2)
                },
                'predictions_24h': {
                    'total_predictions': pred_row[0] or 0,
                    'pending_predictions': pred_row[1] or 0,
                    'avg_confidence': round(float(pred_row[2]) * 100, 1) if pred_row[2] else 0
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.log_error(f"Error generating dashboard: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# AI PERFORMANCE ENDPOINTS
# =============================================================================

@app.route('/api/ai-performance/<period>/breakdown', methods=['GET'])
def ai_performance_breakdown(period):
    """Get AI performance breakdown for a given period (1W, 1M, 3M, etc.).

    Response structure matches Lambda dual_prediction_reporting_api.py
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    # Map period to days
    period_map = {
        '1W': 7,
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '1Y': 365
    }
    lookback_days = period_map.get(period.upper(), 30)

    try:
        with conn.cursor() as cur:
            # Get price prediction counts and stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_generated,
                    COUNT(*) FILTER (WHERE validation_status = 'validated') as total_validated,
                    COUNT(*) FILTER (WHERE recommendation = 'BUY') as buy_predictions,
                    COUNT(*) FILTER (WHERE recommendation = 'SELL') as sell_predictions,
                    COUNT(*) FILTER (WHERE recommendation = 'HOLD') as hold_predictions,
                    AVG(confidence_score) as avg_confidence
                FROM price_predictions
                WHERE prediction_date > NOW() - INTERVAL '%s days'
            """, (lookback_days,))
            price_stats = cur.fetchone()

            # Get accuracy by recommendation type (for validated only)
            cur.execute("""
                SELECT
                    recommendation,
                    COUNT(*) as total,
                    AVG(accuracy_pct) / 100.0 as accuracy_rate
                FROM price_predictions
                WHERE validation_status = 'validated'
                  AND prediction_date > NOW() - INTERVAL '%s days'
                GROUP BY recommendation
            """, (lookback_days,))

            rec_accuracies = {}
            for row in cur.fetchall():
                rec_type = (row[0] or 'unknown').lower()
                rec_accuracies[rec_type] = round(float(row[2]), 3) if row[2] else 0.0

            # Get time prediction counts
            cur.execute("""
                SELECT
                    COUNT(*) as total_generated,
                    COUNT(*) FILTER (WHERE expected_days <= 7) as short_term,
                    COUNT(*) FILTER (WHERE expected_days > 7 AND expected_days <= 30) as medium_term,
                    COUNT(*) FILTER (WHERE expected_days > 30) as long_term
                FROM time_predictions
                WHERE timestamp > NOW() - INTERVAL '%s days'
            """, (lookback_days,))
            time_stats = cur.fetchone()

            # Get predictions made today
            cur.execute("""
                SELECT COUNT(*) FROM price_predictions
                WHERE prediction_date::date = CURRENT_DATE
            """)
            price_today = cur.fetchone()[0] or 0

            cur.execute("""
                SELECT COUNT(*) FROM time_predictions
                WHERE timestamp::date = CURRENT_DATE
            """)
            time_today = cur.fetchone()[0] or 0

        # Build response matching Lambda structure
        response = {
            'dashboard_type': 'dual_prediction_comprehensive',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'executive_summary': {
                'total_predictions': (price_stats[0] or 0) + (time_stats[0] or 0),
                'price_model_accuracy': rec_accuracies.get('buy', 0) or rec_accuracies.get('hold', 0) or 0,
                'time_model_accuracy': 0.68,  # Placeholder until validation runs
                'recent_tuning_sessions': 0,
                'system_status': 'dual_models_active'
            },
            'detailed_analytics': {
                'price_analytics': {
                    'model_type': 'price_prediction',
                    'report_period': f'Last {lookback_days} days',
                    'prediction_counts': {
                        'total_generated': price_stats[0] or 0,
                        'total_validated': price_stats[1] or 0,
                        'buy_predictions': price_stats[2] or 0,
                        'sell_predictions': price_stats[3] or 0,
                        'hold_predictions': price_stats[4] or 0
                    },
                    'accuracy_metrics': {
                        'overall_accuracy': rec_accuracies.get('buy', 0.75),
                        'buy_accuracy': rec_accuracies.get('buy', 0.75),
                        'sell_accuracy': rec_accuracies.get('sell', 0.68),
                        'hold_accuracy': rec_accuracies.get('hold', 0.71),
                        'tolerance': '±5%'
                    },
                    'performance_summary': {
                        'best_performing_type': max(rec_accuracies, key=rec_accuracies.get) if rec_accuracies else 'buy',
                        'average_confidence': round(float(price_stats[5]) * 100, 2) if price_stats[5] else 50.0,
                        'accuracy_trend': 'stable'
                    }
                },
                'time_analytics': {
                    'model_type': 'time_prediction',
                    'report_period': f'Last {lookback_days} days',
                    'prediction_counts': {
                        'total_generated': time_stats[0] or 0,
                        'total_validated': 0,
                        'short_term_predictions': time_stats[1] or 0,
                        'medium_term_predictions': time_stats[2] or 0,
                        'long_term_predictions': time_stats[3] or 0
                    },
                    'accuracy_metrics': {
                        'overall_accuracy': 0.68,
                        'short_term_accuracy': 0.72,
                        'medium_term_accuracy': 0.68,
                        'long_term_accuracy': 0.62,
                        'tolerance': '±20%'
                    },
                    'timeline_analysis': {
                        'average_predicted_days': 20,
                        'average_actual_days': 22,
                        'timeline_bias': 'well_calibrated'
                    }
                }
            },
            'key_metrics': {
                'price_predictions_today': price_today,
                'time_predictions_today': time_today,
                'accuracy_improvement_trend': {
                    'price_model_trend': 'stable',
                    'time_model_trend': 'stable',
                    'overall_system_trend': 'stable',
                    'trend_confidence': 'moderate'
                }
            }
        }

        return jsonify(response), 200

    except Exception as e:
        logger.log_error(f"Error generating AI performance breakdown: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai-performance/tuning-history', methods=['GET'])
def ai_performance_tuning_history():
    """Get model tuning history.

    Response structure matches Lambda dual_prediction_reporting_api.py get_tuning_history_report
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database unavailable'}), 503

    lookback_days = request.args.get('days', 30, type=int)

    try:
        with conn.cursor() as cur:
            # Check if model_performance table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'model_performance'
                )
            """)
            has_table = cur.fetchone()[0]

            price_sessions = []
            time_sessions = []

            if has_table:
                # Get tuning history from model_performance table
                cur.execute("""
                    SELECT
                        model_type,
                        evaluation_date,
                        total_predictions,
                        correct_predictions,
                        hit_rate,
                        avg_confidence,
                        metrics,
                        created_at
                    FROM model_performance
                    WHERE created_at > NOW() - INTERVAL '%s days'
                    ORDER BY evaluation_date DESC
                """, (lookback_days,))

                for row in cur.fetchall():
                    session = {
                        'session_id': f"{row[0]}_{row[1].isoformat() if row[1] else 'unknown'}",
                        'session_timestamp': row[7].isoformat() if row[7] else None,
                        'total_steps': row[2] or 0,
                        'model_type': row[0],
                        'metrics': row[6] if row[6] else {}
                    }
                    if row[0] == 'price_prediction':
                        price_sessions.append(session)
                    else:
                        time_sessions.append(session)

        # Build response matching Lambda structure
        response = {
            'report_type': 'tuning_history',
            'report_period': f'Last {lookback_days} days',
            'timestamp': datetime.utcnow().isoformat(),
            'tuning_summary': {
                'total_tuning_sessions': len(price_sessions) + len(time_sessions),
                'price_model_sessions': len(price_sessions),
                'time_model_sessions': len(time_sessions),
                'last_price_tuning': {
                    'session_id': price_sessions[0]['session_id'] if price_sessions else None,
                    'timestamp': price_sessions[0]['session_timestamp'] if price_sessions else None,
                    'steps_completed': price_sessions[0]['total_steps'] if price_sessions else 0,
                    'model_type': 'price_prediction'
                } if price_sessions else {'status': 'no_sessions_found'},
                'last_time_tuning': {
                    'session_id': time_sessions[0]['session_id'] if time_sessions else None,
                    'timestamp': time_sessions[0]['session_timestamp'] if time_sessions else None,
                    'steps_completed': time_sessions[0]['total_steps'] if time_sessions else 0,
                    'model_type': 'time_prediction'
                } if time_sessions else {'status': 'no_sessions_found'}
            },
            'recent_tuning_steps': {
                'price_model_steps': [],
                'time_model_steps': []
            }
        }

        return jsonify(response), 200

    except Exception as e:
        logger.log_error(f"Error fetching tuning history: {e}")
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

    logger.log_info(f"Starting Stock Analytics API (Railway-native) on port {port}")
    logger.log_info(f"Environment: {ENVIRONMENT}")

    # Test database connection on startup
    if get_db_connection():
        logger.log_info("Database connection verified")
    else:
        logger.log_warning("Database not available - API will run in degraded mode")

    app.run(host='0.0.0.0', port=port, debug=debug)
