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
            'quotes_history': '/quotes/{symbol}'
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
                SELECT recommendation, COUNT(*) as count
                FROM stock_recommendations
                GROUP BY recommendation
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
                SELECT symbol, recommendation, confidence, target_price, current_price
                FROM stock_recommendations
                WHERE recommendation = 'BUY'
                ORDER BY confidence DESC
                LIMIT 5
            """)
            top_buys = [{'symbol': r[0], 'confidence': float(r[2]),
                        'target_price': float(r[3]), 'current_price': float(r[4])}
                       for r in cur.fetchall()]

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
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.log_error(f"Error generating dashboard: {e}")
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
