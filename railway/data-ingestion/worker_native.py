"""
Railway-Native Data Ingestion Worker.
Completely AWS-free implementation using PostgreSQL for storage.
"""

import os
import sys
import time
import json
import schedule
import urllib.request
import urllib.parse
import socket
from datetime import datetime, time as dt_time
from decimal import Decimal
import pytz

# Add railway shared to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.logger import StructuredLogger

logger = StructuredLogger(__name__)

# Configuration from environment
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
DATABASE_URL = os.environ.get('DATABASE_URL', '')
REDIS_URL = os.environ.get('REDIS_URL', '')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
MAX_SYMBOLS_PER_RUN = int(os.environ.get('MAX_SYMBOLS_PER_RUN', '3'))
MARKET_INTERVAL_MINUTES = int(os.environ.get('MARKET_INTERVAL_MINUTES', '5'))
EVENING_INTERVAL_MINUTES = int(os.environ.get('EVENING_INTERVAL_MINUTES', '10'))
PER_CALL_TIMEOUT = int(os.environ.get('PER_CALL_TIMEOUT', '8'))

# Stock universe (simplified for Railway)
MAJOR_INDEXES = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']

POPULAR_STOCKS = [
    # Mega Cap
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    # Tech
    'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'UBER', 'SHOP',
    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN',
    # Consumer
    'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'COST', 'TGT', 'LOW',
    # Energy/Industrial
    'XOM', 'CVX', 'BA', 'CAT', 'GE', 'HON', 'UNP', 'LMT'
]

# Database connection
db_conn = None
redis_client = None


def init_database():
    """Initialize PostgreSQL connection and create tables if needed."""
    global db_conn

    if not DATABASE_URL:
        logger.log_error("DATABASE_URL not configured")
        return False

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        db_conn = psycopg2.connect(DATABASE_URL)
        db_conn.autocommit = True

        # Create tables if they don't exist
        # Note: Uses trading_day column to match existing Railway database schema
        with db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_quotes (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    price DECIMAL(10, 2) NOT NULL,
                    open_price DECIMAL(10, 2),
                    high_price DECIMAL(10, 2),
                    low_price DECIMAL(10, 2),
                    volume BIGINT,
                    previous_close DECIMAL(10, 2),
                    change_amount DECIMAL(10, 2),
                    change_percent DECIMAL(6, 2),
                    trading_day DATE,
                    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(symbol, trading_day)
                );

                CREATE INDEX IF NOT EXISTS idx_stock_quotes_symbol ON stock_quotes(symbol);
                CREATE INDEX IF NOT EXISTS idx_stock_quotes_trading_day ON stock_quotes(trading_day DESC);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS latest_prices (
                    symbol VARCHAR(10) PRIMARY KEY,
                    price DECIMAL(12,4),
                    change_amount DECIMAL(12,4),
                    change_percent DECIMAL(8,4),
                    volume BIGINT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_recommendations (
                    symbol VARCHAR(10) PRIMARY KEY,
                    recommendation VARCHAR(20),
                    confidence DECIMAL(5,4),
                    target_price DECIMAL(12,4),
                    current_price DECIMAL(12,4),
                    analysis_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS price_predictions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    prediction_date DATE NOT NULL,
                    predicted_price DECIMAL(12,4),
                    actual_price DECIMAL(12,4),
                    confidence DECIMAL(5,4),
                    model_version VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, prediction_date, model_version)
                );

                CREATE INDEX IF NOT EXISTS idx_price_predictions_symbol ON price_predictions(symbol);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_logs (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(50),
                    symbols_attempted INT,
                    symbols_succeeded INT,
                    duration_seconds DECIMAL(8,2),
                    errors JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        logger.log_info("Database initialized successfully")
        return True

    except ImportError:
        logger.log_error("psycopg2 not installed - run: pip install psycopg2-binary")
        return False
    except Exception as e:
        logger.log_error(f"Database initialization failed: {e}")
        return False


def init_cache():
    """Initialize Redis connection for caching."""
    global redis_client

    if not REDIS_URL:
        logger.log_info("Redis not configured - caching disabled")
        return False

    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.log_info("Redis cache initialized")
        return True
    except ImportError:
        logger.log_info("redis package not installed - caching disabled")
        return False
    except Exception as e:
        logger.log_warning(f"Redis connection failed: {e}")
        return False


def is_market_hours():
    """Check if current time is during market hours (9 AM - 4 PM EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    if now.weekday() >= 5:  # Weekend
        return False

    market_open = dt_time(9, 0)
    market_close = dt_time(16, 0)
    return market_open <= now.time() <= market_close


def is_evening_hours():
    """Check if current time is evening processing hours (5 PM - 11 PM EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    if now.weekday() >= 5:
        return False

    evening_start = dt_time(17, 0)
    evening_end = dt_time(23, 0)
    return evening_start <= now.time() <= evening_end


def fetch_stock_data(symbol):
    """Fetch stock data from Alpha Vantage API."""
    if not ALPHA_VANTAGE_API_KEY:
        logger.log_error("Alpha Vantage API key not configured")
        return None

    # Check cache first
    if redis_client:
        try:
            cached = redis_client.get(f"stock_data:{symbol}")
            if cached:
                logger.log_info(f"Cache hit for {symbol}")
                return json.loads(cached)
        except Exception as e:
            logger.log_warning(f"Cache read error: {e}")

    try:
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': 'compact',
            'apikey': ALPHA_VANTAGE_API_KEY
        }

        query = urllib.parse.urlencode(params)
        full_url = f"{url}?{query}"

        with urllib.request.urlopen(full_url, timeout=PER_CALL_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        if 'Error Message' in data:
            logger.log_warning(f"API error for {symbol}: {data['Error Message']}")
            return None

        if 'Note' in data:
            logger.log_warning(f"Rate limit for {symbol}: {data['Note']}")
            return None

        # Cache the result
        if redis_client:
            try:
                redis_client.setex(f"stock_data:{symbol}", 300, json.dumps(data))
            except Exception as e:
                logger.log_warning(f"Cache write error: {e}")

        return data

    except Exception as e:
        logger.log_error(f"Fetch error for {symbol}: {e}")
        return None


def process_stock_data(symbol, raw_data):
    """Process raw Alpha Vantage data into structured format."""
    try:
        if 'Time Series (Daily)' not in raw_data:
            return None

        ts = raw_data['Time Series (Daily)']
        dates = sorted(ts.keys(), reverse=True)

        if not dates:
            return None

        latest_date = dates[0]
        latest = ts[latest_date]

        # Get current and previous close
        close_price = float(latest['4. close'])
        open_price = float(latest['1. open'])

        # Get previous close from day before if available
        previous_close = close_price
        if len(dates) > 1:
            prev_day = ts[dates[1]]
            previous_close = float(prev_day['4. close'])

        # Calculate change
        change_amount = close_price - previous_close
        change_percent = (change_amount / previous_close * 100) if previous_close > 0 else 0

        return {
            'symbol': symbol,
            'trading_day': latest_date,
            'price': close_price,
            'open': open_price,
            'high': float(latest['2. high']),
            'low': float(latest['3. low']),
            'volume': int(latest['5. volume']),
            'previous_close': previous_close,
            'change': change_amount,
            'change_percent': change_percent,
            'fetched_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.log_error(f"Process error for {symbol}: {e}")
        return None


def store_quote(symbol, data):
    """Store processed quote in PostgreSQL."""
    if not db_conn or not data:
        return False

    try:
        with db_conn.cursor() as cur:
            # Insert/update stock quote - matches existing Railway database schema
            cur.execute("""
                INSERT INTO stock_quotes
                    (symbol, price, open_price, high_price, low_price,
                     volume, previous_close, change_amount, change_percent,
                     trading_day, fetched_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, trading_day)
                DO UPDATE SET
                    price = EXCLUDED.price,
                    open_price = EXCLUDED.open_price,
                    high_price = EXCLUDED.high_price,
                    low_price = EXCLUDED.low_price,
                    volume = EXCLUDED.volume,
                    previous_close = EXCLUDED.previous_close,
                    change_amount = EXCLUDED.change_amount,
                    change_percent = EXCLUDED.change_percent,
                    fetched_at = EXCLUDED.fetched_at
            """, (
                data['symbol'],
                data['price'],
                data['open'],
                data['high'],
                data['low'],
                data['volume'],
                data['previous_close'],
                data['change'],
                data['change_percent'],
                data['trading_day'],
                data['fetched_at']
            ))

            # Update latest price
            cur.execute("""
                INSERT INTO latest_prices (symbol, price, change_amount, change_percent, volume, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    price = EXCLUDED.price,
                    change_amount = EXCLUDED.change_amount,
                    change_percent = EXCLUDED.change_percent,
                    volume = EXCLUDED.volume,
                    updated_at = CURRENT_TIMESTAMP
            """, (symbol, data['price'], data['change'], data['change_percent'], data['volume']))

        return True

    except Exception as e:
        logger.log_error(f"Store error for {symbol}: {e}")
        return False


def generate_recommendation(symbol, data):
    """Generate a simple recommendation based on price momentum."""
    if not data:
        return None

    try:
        current_price = data['price']
        change_percent = data['change_percent']

        # Simple momentum-based recommendation
        if change_percent > 2:
            recommendation = 'BUY'
            confidence = min(0.8, 0.5 + change_percent / 10)
        elif change_percent < -2:
            recommendation = 'SELL'
            confidence = min(0.8, 0.5 + abs(change_percent) / 10)
        else:
            recommendation = 'HOLD'
            confidence = 0.5

        target_price = current_price * (1.05 if recommendation == 'BUY' else 0.95 if recommendation == 'SELL' else 1.0)

        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': round(confidence, 4),
            'target_price': round(target_price, 2),
            'current_price': current_price,
            'analysis': {
                'change_percent': change_percent,
                'trend': 'bullish' if change_percent > 0 else 'bearish' if change_percent < 0 else 'neutral'
            }
        }

    except Exception as e:
        logger.log_error(f"Recommendation error for {symbol}: {e}")
        return None


def store_recommendation(symbol, rec):
    """Store recommendation in PostgreSQL."""
    if not db_conn or not rec:
        return False

    try:
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_recommendations
                    (symbol, recommendation, confidence, target_price, current_price, analysis_data, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (symbol)
                DO UPDATE SET
                    recommendation = EXCLUDED.recommendation,
                    confidence = EXCLUDED.confidence,
                    target_price = EXCLUDED.target_price,
                    current_price = EXCLUDED.current_price,
                    analysis_data = EXCLUDED.analysis_data,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                rec['symbol'], rec['recommendation'], rec['confidence'],
                rec['target_price'], rec['current_price'],
                json.dumps(rec['analysis'])
            ))
        return True
    except Exception as e:
        logger.log_error(f"Store recommendation error for {symbol}: {e}")
        return False


def log_ingestion_run(run_id, attempted, succeeded, duration, errors):
    """Log ingestion run metrics to database."""
    if not db_conn:
        return

    try:
        with db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ingestion_logs (run_id, symbols_attempted, symbols_succeeded, duration_seconds, errors)
                VALUES (%s, %s, %s, %s, %s)
            """, (run_id, attempted, succeeded, duration, json.dumps(errors)))
    except Exception as e:
        logger.log_error(f"Log ingestion error: {e}")


def run_data_ingestion(max_symbols=None):
    """Execute data ingestion cycle."""
    if max_symbols is None:
        max_symbols = MAX_SYMBOLS_PER_RUN

    start_time = time.time()
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger.log_info(f"Starting data ingestion cycle (max_symbols={max_symbols})")

    # Determine symbols to process
    current_hour = datetime.utcnow().hour
    rotation_seed = current_hour // 4

    # Always include some indexes
    idx_count = min(2, max_symbols // 3)
    idx_symbols = MAJOR_INDEXES[:idx_count]

    # Rotate through stocks
    remaining = max_symbols - len(idx_symbols)
    stock_start = (rotation_seed * remaining) % len(POPULAR_STOCKS)
    rotated = POPULAR_STOCKS[stock_start:] + POPULAR_STOCKS[:stock_start]
    stock_symbols = rotated[:remaining]

    all_symbols = idx_symbols + stock_symbols

    succeeded = 0
    errors = []

    for symbol in all_symbols:
        try:
            # Fetch data
            raw_data = fetch_stock_data(symbol)
            if not raw_data:
                errors.append({'symbol': symbol, 'error': 'fetch_failed'})
                continue

            # Process data
            processed = process_stock_data(symbol, raw_data)
            if not processed:
                errors.append({'symbol': symbol, 'error': 'process_failed'})
                continue

            # Store quote
            if store_quote(symbol, processed):
                # Generate and store recommendation
                rec = generate_recommendation(symbol, processed)
                if rec:
                    store_recommendation(symbol, rec)
                succeeded += 1
                logger.log_info(f"Processed {symbol}: ${processed['price']:.2f}")
            else:
                errors.append({'symbol': symbol, 'error': 'store_failed'})

            # Rate limiting - be nice to the API
            time.sleep(0.5)

        except Exception as e:
            errors.append({'symbol': symbol, 'error': str(e)})
            logger.log_error(f"Error processing {symbol}: {e}")

    duration = round(time.time() - start_time, 2)

    # Log the run
    log_ingestion_run(run_id, len(all_symbols), succeeded, duration, errors)

    if errors:
        logger.log_info(f"Data ingestion finished with {len(errors)} errors")
    else:
        logger.log_info(f"Data ingestion completed: {succeeded}/{len(all_symbols)} symbols in {duration}s")

    return succeeded, len(errors)


def run_market_hours_job():
    """Job for market hours - runs only during trading hours."""
    if is_market_hours():
        logger.log_info("Market hours active - running data ingestion")
        run_data_ingestion()
    else:
        logger.log_info("Outside market hours - skipping ingestion")


def run_evening_job():
    """Job for evening hours - runs during extended processing."""
    if is_evening_hours():
        logger.log_info("Evening hours active - running data ingestion")
        run_data_ingestion()
    else:
        logger.log_info("Outside evening hours - skipping ingestion")


def setup_schedules():
    """Set up the job schedules."""
    schedule.every(MARKET_INTERVAL_MINUTES).minutes.do(run_market_hours_job)
    logger.log_info(f"Scheduled market hours job: every {MARKET_INTERVAL_MINUTES} minutes")

    schedule.every(EVENING_INTERVAL_MINUTES).minutes.do(run_evening_job)
    logger.log_info(f"Scheduled evening job: every {EVENING_INTERVAL_MINUTES} minutes")

    # End of day comprehensive run at 4:30 PM EST
    schedule.every().day.at("16:30").do(
        lambda: run_data_ingestion() if datetime.now(pytz.timezone('US/Eastern')).weekday() < 5 else None
    )
    logger.log_info("Scheduled end-of-day comprehensive run at 4:30 PM EST")


def main():
    """Main worker loop."""
    logger.log_info("Starting Stock Analytics Data Ingestion Worker (Railway-native)")
    logger.log_info(f"Environment: {ENVIRONMENT}")
    logger.log_info(f"Stock universe: {len(POPULAR_STOCKS)} symbols")
    logger.log_info(f"Alpha Vantage API key configured: {bool(ALPHA_VANTAGE_API_KEY)}")

    # Initialize database
    logger.log_info("Initializing database tables")
    if not init_database():
        logger.log_error("Failed to initialize database - exiting")
        sys.exit(1)

    # Initialize cache (optional)
    init_cache()

    # Setup schedules
    setup_schedules()

    # Run initial ingestion on startup
    logger.log_info("Running initial data ingestion")
    run_data_ingestion()

    # Main loop
    logger.log_info("Worker running - waiting for scheduled jobs")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            logger.log_info("Worker shutting down gracefully")
            if db_conn:
                db_conn.close()
            break
        except Exception as e:
            logger.log_error(f"Error in worker loop: {e}")
            time.sleep(60)


if __name__ == '__main__':
    main()
