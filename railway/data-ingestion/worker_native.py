"""
Railway-Native Data Ingestion Worker.
Completely AWS-free implementation using PostgreSQL for storage.
"""

import os
import sys
import time
import json
import math
import schedule
import urllib.request
import urllib.parse
import socket
from datetime import datetime, time as dt_time, timedelta
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
MAX_SYMBOLS_PER_RUN = int(os.environ.get('MAX_SYMBOLS_PER_RUN', '45'))  # All stocks per run with premium API
MARKET_INTERVAL_MINUTES = int(os.environ.get('MARKET_INTERVAL_MINUTES', '5'))
EVENING_INTERVAL_MINUTES = int(os.environ.get('EVENING_INTERVAL_MINUTES', '10'))
PER_CALL_TIMEOUT = int(os.environ.get('PER_CALL_TIMEOUT', '8'))
API_CALL_DELAY = float(os.environ.get('API_CALL_DELAY', '0.2'))  # 200ms delay = ~300 calls/min capacity (well under 75/min limit per symbol)

# Stock universe - expanded for comprehensive market coverage
# Target: ~450 symbols with smart daily rotation
# Premium API allows 75 calls/min; Free tier allows 5 calls/min (500/day)

MAJOR_INDEXES = [
    # US Market ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VXX', 'ARKK', 'XLF', 'XLE',
    'XLK', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE',
    # International/Bond ETFs
    'EEM', 'EFA', 'VWO', 'TLT', 'LQD', 'HYG', 'GLD', 'SLV', 'USO'
]

POPULAR_STOCKS = [
    # === MEGA CAP (Market Cap > $200B) ===
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
    'UNH', 'JNJ', 'V', 'XOM', 'JPM', 'WMT', 'MA', 'PG', 'HD', 'CVX',
    'LLY', 'MRK', 'ABBV', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO',
    'ACN', 'ABT', 'DHR', 'WFC', 'CRM', 'LIN', 'TXN', 'PM', 'VZ', 'NEE',

    # === LARGE CAP TECH ===
    'ADBE', 'AMD', 'INTC', 'ORCL', 'QCOM', 'IBM', 'NOW', 'INTU', 'AMAT', 'ADI',
    'LRCX', 'MU', 'SNPS', 'CDNS', 'KLAC', 'MRVL', 'NXPI', 'FTNT', 'PANW', 'CRWD',
    'DDOG', 'ZS', 'OKTA', 'NET', 'SNOW', 'PLTR', 'PATH', 'MDB', 'TEAM',

    # === LARGE CAP FINANCIALS ===
    'BAC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'CB', 'PNC',
    'USB', 'TFC', 'AIG', 'MET', 'PRU', 'AFL', 'ALL', 'TRV', 'CME', 'ICE',
    'SPGI', 'MCO', 'MSCI', 'FIS', 'FISV', 'GPN', 'ADP', 'PYPL', 'COIN',

    # === LARGE CAP HEALTHCARE ===
    'PFE', 'BMY', 'AMGN', 'GILD', 'VRTX', 'REGN', 'MRNA', 'BIIB', 'ILMN', 'DXCM',
    'ISRG', 'SYK', 'BDX', 'MDT', 'ZBH', 'BSX', 'EW', 'A', 'IQV', 'MTD',
    'CI', 'ELV', 'HUM', 'CNC', 'MCK', 'CAH', 'COR', 'CVS', 'WBA', 'HCA',

    # === LARGE CAP CONSUMER ===
    'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR', 'ORLY', 'AZO',
    'BBY', 'ULTA', 'LULU', 'GAP', 'KSS', 'M', 'JWN', 'DRI', 'CMG', 'YUM',
    'DPZ', 'WING', 'SHAK', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'MAR', 'HLT', 'H',

    # === LARGE CAP INDUSTRIAL ===
    'BA', 'CAT', 'GE', 'HON', 'UNP', 'LMT', 'RTX', 'DE', 'MMM', 'UPS',
    'FDX', 'CSX', 'NSC', 'WM', 'RSG', 'EMR', 'ETN', 'ITW', 'ROK', 'PH',
    'CMI', 'PCAR', 'ODFL', 'JBHT', 'XPO', 'CHRW', 'EXPD', 'DAL', 'UAL', 'LUV',

    # === LARGE CAP ENERGY ===
    'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL', 'DVN',
    'FANG', 'HES', 'APA', 'MRO', 'KMI', 'WMB', 'OKE', 'ET', 'EPD', 'TRGP',

    # === LARGE CAP UTILITIES/REITS ===
    'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ED', 'WEC', 'ES',
    'PEG', 'AWK', 'AEE', 'CMS', 'DTE', 'FE', 'PPL', 'EVRG', 'NI', 'AES',
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'DLR', 'O', 'WELL', 'AVB', 'EQR',

    # === LARGE CAP MATERIALS ===
    'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM',
    'PPG', 'ALB', 'LYB', 'EMN', 'CE', 'CF', 'MOS', 'FMC', 'IFF', 'CTVA',

    # === LARGE CAP COMMUNICATIONS ===
    'NFLX', 'DIS', 'CMCSA', 'T', 'TMUS', 'CHTR', 'PARA', 'WBD', 'FOX', 'FOXA',
    'TTWO', 'EA', 'RBLX', 'U', 'MTCH', 'SPOT', 'PINS', 'SNAP',

    # === MID CAP GROWTH ===
    'UBER', 'LYFT', 'SHOP', 'ETSY', 'W', 'CHWY', 'PTON', 'HOOD', 'SOFI', 'UPST',
    'AFRM', 'BILL', 'HUBS', 'TWLO', 'TTD', 'ROKU', 'ZM', 'DOCU', 'VEEV', 'PAYC',
    'PCTY', 'WDAY', 'RNG', 'ZI', 'ESTC', 'CFLT', 'GTLB', 'DOCN', 'FROG', 'S',

    # === MID CAP VALUE ===
    'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'GOEV',
    'AAL', 'JBLU', 'ALK', 'SAVE', 'HA', 'SKYW', 'ALGT', 'LTM', 'RYAAY',
    'CLF', 'X', 'AA', 'STLD', 'RS', 'ATI', 'CMC', 'TX', 'MT', 'VALE',

    # === SMALL CAP SPECULATIVE ===
    'AMC', 'GME', 'BB', 'EXPR', 'KOSS', 'SNDL', 'TLRY', 'CGC',
    'ACB', 'CRON', 'HEXO', 'OGI', 'VFF', 'GRWG', 'CURLF', 'TCNNF', 'GTBIF', 'CRLBF',

    # === BIOTECH/PHARMA ===
    'SGEN', 'ALNY', 'BMRN', 'INCY', 'EXEL', 'SRPT', 'IONS', 'NBIX', 'RARE', 'FOLD',
    'ARWR', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'BLUE', 'SGMO', 'FATE', 'KYMR',

    # === SEMICONDUCTORS ===
    'ASML', 'TSM', 'SOXL', 'SOXS', 'ON', 'SWKS', 'QRVO', 'MPWR', 'ALGM', 'WOLF',
    'CRUS', 'SLAB', 'DIOD', 'AMBA', 'SITM', 'POWI', 'SMTC', 'AOSL', 'FORM', 'RMBS',

    # === CYBERSECURITY ===
    'CYBR', 'TENB', 'VRNS', 'QLYS', 'RPD', 'SAIL', 'SWI', 'GEN',

    # === AI/ML FOCUSED ===
    'AI', 'BBAI', 'SOUN', 'GFAI', 'PRCT', 'AITX', 'VERI', 'AISP'
]

# Remove duplicates while preserving order
POPULAR_STOCKS = list(dict.fromkeys(POPULAR_STOCKS))

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

            # Note: price_predictions and time_predictions tables are expected to exist
            # from the main schema.sql migration. We verify they exist but don't create them.
            # If they don't exist, the store_*_prediction functions will gracefully fail.
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'price_predictions'
                );
            """)
            has_price_table = cur.fetchone()[0]
            if has_price_table:
                logger.log_info("price_predictions table exists")
            else:
                logger.log_warning("price_predictions table does not exist - predictions will be skipped")

            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'time_predictions'
                );
            """)
            has_time_table = cur.fetchone()[0]
            if has_time_table:
                logger.log_info("time_predictions table exists")
            else:
                logger.log_warning("time_predictions table does not exist - predictions will be skipped")

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
    """Check if current time is during market hours (9:30 AM - 4 PM EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    if now.weekday() >= 5:  # Weekend
        return False

    market_open = dt_time(9, 30)  # US market opens at 9:30 AM ET
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
        import uuid
        recommendation_id = str(uuid.uuid4())

        with db_conn.cursor() as cur:
            # Check if recommendation for this symbol exists
            cur.execute("SELECT id FROM stock_recommendations WHERE symbol = %s", (symbol,))
            existing = cur.fetchone()

            if existing:
                # Update existing
                cur.execute("""
                    UPDATE stock_recommendations SET
                        recommendation_type = %s,
                        confidence = %s,
                        target_price = %s,
                        current_price = %s,
                        metadata = %s,
                        timestamp = CURRENT_TIMESTAMP
                    WHERE symbol = %s
                """, (
                    rec['recommendation'],
                    rec['confidence'],
                    rec['target_price'],
                    rec['current_price'],
                    json.dumps(rec['analysis']),
                    symbol
                ))
            else:
                # Insert new
                cur.execute("""
                    INSERT INTO stock_recommendations
                        (recommendation_id, symbol, recommendation_type, confidence,
                         target_price, current_price, metadata, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """, (
                    recommendation_id,
                    rec['symbol'],
                    rec['recommendation'],
                    rec['confidence'],
                    rec['target_price'],
                    rec['current_price'],
                    json.dumps(rec['analysis'])
                ))
        return True
    except Exception as e:
        logger.log_error(f"Store recommendation error for {symbol}: {e}")
        return False


# =============================================================================
# PRICE PREDICTION MODEL
# =============================================================================

# Sector mappings for prediction algorithm
TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'NFLX']
FINANCE_STOCKS = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL']
HEALTHCARE_STOCKS = ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN']
LARGE_CAP = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 'UNH']


def get_sector_strength(symbol):
    """Get sector relative strength for prediction."""
    if symbol in TECH_STOCKS:
        return 0.01  # Tech slightly positive
    elif symbol in FINANCE_STOCKS:
        return -0.005  # Finance slightly negative
    elif symbol in HEALTHCARE_STOCKS:
        return 0.015  # Healthcare positive
    else:
        return 0.0  # Neutral for others


def get_market_trend():
    """Get overall market trend (simplified)."""
    return 0.005  # Slightly positive market


def calculate_volatility(symbol):
    """Calculate stock volatility based on category."""
    if symbol in LARGE_CAP:
        return 0.15  # 15% volatility for large cap
    else:
        return 0.25  # 25% volatility for others


def calculate_base_trend(change_percent):
    """Calculate base trend from price change (simulating RSI/MACD behavior)."""
    # Simulate RSI-like signal based on recent price change
    if change_percent < -3:
        rsi_signal = 0.08  # Oversold - bullish
    elif change_percent > 3:
        rsi_signal = -0.06  # Overbought - bearish
    else:
        rsi_signal = -change_percent / 100  # Linear scaling

    # Simulate MACD-like signal
    macd_signal = max(min(change_percent * 0.01, 0.05), -0.05)

    # Bollinger-like position (assume middle)
    bollinger_signal = 0

    return rsi_signal + macd_signal + bollinger_signal


def generate_price_prediction(symbol, data, timeframe_days=30):
    """
    Generate price target prediction using technical and fundamental analysis.
    Ported from Lambda price_prediction_model.py.
    """
    if not data:
        return None

    try:
        current_price = data['price']
        change_percent = data['change_percent']

        # Fundamental factors
        sector_strength = get_sector_strength(symbol)
        market_trend = get_market_trend()
        volatility = calculate_volatility(symbol)

        # Price prediction algorithm
        base_trend = calculate_base_trend(change_percent)
        sector_adjustment = sector_strength * 0.02  # ±2% sector impact
        market_adjustment = market_trend * 0.015    # ±1.5% market impact
        volume_impact = 0  # Simplified - no volume ratio available

        # Calculate total expected return
        total_expected_return = base_trend + sector_adjustment + market_adjustment + volume_impact

        # Apply timeframe scaling
        timeframe_multiplier = math.sqrt(timeframe_days / 30.0)
        adjusted_return = total_expected_return * timeframe_multiplier

        # Calculate target price
        target_price = current_price * (1 + adjusted_return)

        # Determine recommendation
        if adjusted_return > 0.05:
            recommendation = 'BUY'
        elif adjusted_return < -0.05:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'

        # Calculate confidence based on signal strength
        signal_strength = abs(adjusted_return)
        confidence = min(0.5 + (signal_strength * 2), 0.95)

        # Price range estimation
        price_range_pct = volatility * 0.5
        price_low = target_price * (1 - price_range_pct)
        price_high = target_price * (1 + price_range_pct)

        # Key factors driving prediction
        factors = []
        if abs(change_percent) > 2:
            factors.append('momentum_signal' if change_percent > 0 else 'reversal_signal')
        if sector_strength > 0.01:
            factors.append('sector_strength')
        elif sector_strength < -0.01:
            factors.append('sector_weakness')
        if symbol in LARGE_CAP:
            factors.append('large_cap_stability')

        return {
            'symbol': symbol,
            'target_price': round(target_price, 2),
            'current_price': current_price,
            'recommendation': recommendation,
            'confidence': round(confidence, 4),
            'expected_return': round(adjusted_return * 100, 2),
            'price_range': {
                'low': round(price_low, 2),
                'high': round(price_high, 2)
            },
            'factors': factors,
            'timeframe_days': timeframe_days,
            'model_version': 'price_v1.0_railway'
        }

    except Exception as e:
        logger.log_error(f"Price prediction error for {symbol}: {e}")
        return None


def store_price_prediction(symbol, prediction):
    """Store price prediction in PostgreSQL.

    Matches railway/schema.sql price_predictions table structure.
    """
    if not db_conn or not prediction:
        return False

    try:
        validation_date = datetime.utcnow() + timedelta(days=prediction['timeframe_days'])

        with db_conn.cursor() as cur:
            # Insert new prediction (no upsert - each prediction is unique with UUID)
            cur.execute("""
                INSERT INTO price_predictions
                    (symbol, predicted_price, confidence_score, recommendation,
                     prediction_date, validation_date, validation_status)
                VALUES (%s, %s, %s, %s, NOW(), %s, 'pending')
            """, (
                symbol,
                prediction['target_price'],
                prediction['confidence'],
                prediction['recommendation'],
                validation_date
            ))
        logger.log_info(f"Stored price prediction for {symbol}: ${prediction['target_price']:.2f} ({prediction['recommendation']})")
        return True
    except Exception as e:
        logger.log_error(f"Store price prediction error for {symbol}: {e}")
        return False


# =============================================================================
# TIME-TO-HIT PREDICTION MODEL
# =============================================================================

def generate_time_prediction(symbol, data, target_price=None):
    """
    Generate time-to-hit prediction.
    Ported from Lambda time_to_hit_predictor_slim.py.
    """
    if not data:
        return None

    try:
        current_price = data['price']

        # If no target price provided, use recommendation target (+/- 5%)
        if target_price is None:
            change_percent = data['change_percent']
            if change_percent > 0:
                target_price = current_price * 1.05  # 5% up target
            else:
                target_price = current_price * 0.95  # 5% down target

        # Calculate price change percentage to target
        price_change_pct = ((target_price - current_price) / current_price) * 100

        # Time estimation based on price change magnitude
        if abs(price_change_pct) < 2:
            expected_days_min, expected_days_max = 5, 15
            confidence_level = 'high'
            confidence_score = 0.85
        elif abs(price_change_pct) < 5:
            expected_days_min, expected_days_max = 10, 30
            confidence_level = 'medium'
            confidence_score = 0.70
        elif abs(price_change_pct) < 10:
            expected_days_min, expected_days_max = 15, 60
            confidence_level = 'medium'
            confidence_score = 0.55
        else:
            expected_days_min, expected_days_max = 30, 120
            confidence_level = 'low'
            confidence_score = 0.40

        predicted_days = (expected_days_min + expected_days_max) // 2
        expected_hit_date = datetime.utcnow().date() + timedelta(days=predicted_days)

        return {
            'symbol': symbol,
            'target_price': round(target_price, 2),
            'current_price': current_price,
            'price_change_pct': round(price_change_pct, 2),
            'predicted_days_min': expected_days_min,
            'predicted_days_max': expected_days_max,
            'predicted_days': predicted_days,
            'expected_hit_date': expected_hit_date,
            'confidence_level': confidence_level,
            'confidence_score': round(confidence_score, 4),
            'model_version': 'time_v1.0_railway'
        }

    except Exception as e:
        logger.log_error(f"Time prediction error for {symbol}: {e}")
        return None


def store_time_prediction(symbol, prediction):
    """Store time-to-hit prediction in PostgreSQL.

    Handles both the full schema (from schema.sql) and simpler schemas.
    """
    if not db_conn or not prediction:
        return False

    try:
        with db_conn.cursor() as cur:
            # Check if target_price column exists in time_predictions table
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'time_predictions' AND column_name = 'target_price'
            """)
            has_target_price = cur.fetchone() is not None

            if has_target_price:
                # Full schema from schema.sql
                cur.execute("""
                    INSERT INTO time_predictions
                        (symbol, target_price, predicted_days, confidence_score,
                         prediction_date, expected_hit_date, validation_status)
                    VALUES (%s, %s, %s, %s, NOW(), %s, 'pending')
                """, (
                    symbol,
                    prediction['target_price'],
                    prediction['predicted_days'],
                    prediction['confidence_score'],
                    prediction['expected_hit_date']
                ))
            else:
                # Simpler schema without target_price - check what columns exist
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'time_predictions'
                """)
                columns = [row[0] for row in cur.fetchall()]

                # Handle the Railway schema: id, prediction_id, symbol, expected_days, confidence_level, prediction_data, timestamp
                if 'symbol' in columns and 'expected_days' in columns:
                    import uuid
                    prediction_id = str(uuid.uuid4())

                    # Build prediction_data JSON with full prediction details
                    prediction_data = {
                        'target_price': prediction['target_price'],
                        'current_price': prediction['current_price'],
                        'price_change_pct': prediction['price_change_pct'],
                        'predicted_days_min': prediction['predicted_days_min'],
                        'predicted_days_max': prediction['predicted_days_max'],
                        'expected_hit_date': prediction['expected_hit_date'].isoformat() if prediction['expected_hit_date'] else None,
                        'model_version': prediction['model_version']
                    }

                    cur.execute("""
                        INSERT INTO time_predictions
                            (prediction_id, symbol, expected_days, confidence_level, prediction_data, timestamp)
                        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """, (
                        prediction_id,
                        symbol,
                        prediction['predicted_days'],
                        prediction['confidence_level'],
                        json.dumps(prediction_data)
                    ))
                elif 'symbol' in columns and 'predicted_days' in columns:
                    # Alternative schema with predicted_days
                    insert_cols = ['symbol', 'predicted_days']
                    insert_vals = [symbol, prediction['predicted_days']]

                    if 'confidence_score' in columns:
                        insert_cols.append('confidence_score')
                        insert_vals.append(prediction['confidence_score'])
                    elif 'confidence' in columns:
                        insert_cols.append('confidence')
                        insert_vals.append(prediction['confidence_score'])

                    if 'expected_hit_date' in columns:
                        insert_cols.append('expected_hit_date')
                        insert_vals.append(prediction['expected_hit_date'])

                    if 'prediction_date' in columns:
                        insert_cols.append('prediction_date')
                        insert_vals.append(datetime.utcnow())

                    placeholders = ', '.join(['%s'] * len(insert_vals))
                    col_str = ', '.join(insert_cols)
                    cur.execute(f"""
                        INSERT INTO time_predictions ({col_str})
                        VALUES ({placeholders})
                    """, insert_vals)
                else:
                    logger.log_warning(f"time_predictions table has unexpected schema: {columns}")
                    return False

        logger.log_info(f"Stored time prediction for {symbol}: {prediction['predicted_days']} days to ${prediction['target_price']:.2f}")
        return True
    except Exception as e:
        logger.log_error(f"Store time prediction error for {symbol}: {e}")
        return False


# =============================================================================
# LOGGING AND INGESTION
# =============================================================================

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


def get_symbols_not_updated_today():
    """Get symbols that haven't been updated today from database.

    This enables smart rotation to ensure each symbol gets updated once per day
    within API rate limits (500/day free tier, or higher with premium).
    """
    if not db_conn:
        return []

    try:
        with db_conn.cursor() as cur:
            # Get symbols updated today
            cur.execute("""
                SELECT DISTINCT symbol FROM latest_prices
                WHERE updated_at::date = CURRENT_DATE
            """)
            updated_today = {row[0] for row in cur.fetchall()}

            # Return symbols not yet updated
            all_symbols = set(MAJOR_INDEXES + POPULAR_STOCKS)
            return list(all_symbols - updated_today)
    except Exception as e:
        logger.log_error(f"Error getting symbols not updated today: {e}")
        return []


def run_data_ingestion(max_symbols=None):
    """Execute data ingestion cycle with smart rotation.

    Strategy: Process symbols that haven't been updated today first.
    This ensures each symbol gets updated once per day across the
    available API calls (500/day free tier, or more with premium).

    With ~450 symbols and premium API (75 calls/min):
    - Can process all symbols in ~6 minutes
    - Running every 5 min = full market coverage

    With ~450 symbols and free tier (500 calls/day):
    - Each symbol gets called ~1x per day
    - Market hours (9am-4pm = 7 hours) = ~70 calls/hour
    """
    if max_symbols is None:
        max_symbols = MAX_SYMBOLS_PER_RUN

    start_time = time.time()
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Get symbols that need updating today (smart rotation)
    pending_symbols = get_symbols_not_updated_today()

    if pending_symbols:
        logger.log_info(f"Found {len(pending_symbols)} symbols not yet updated today")
    else:
        # All symbols updated - use time-based rotation for refresh
        pending_symbols = MAJOR_INDEXES + POPULAR_STOCKS
        logger.log_info("All symbols updated today - refreshing with rotation")

    # Prioritize: indexes first, then by alphabetical for determinism
    indexes_pending = [s for s in pending_symbols if s in MAJOR_INDEXES]
    stocks_pending = [s for s in pending_symbols if s not in MAJOR_INDEXES]
    stocks_pending.sort()  # Alphabetical for deterministic ordering

    # Always include 1-2 indexes per run for benchmark tracking
    idx_count = min(2, len(indexes_pending), max_symbols // 3)
    idx_symbols = indexes_pending[:idx_count]

    # Fill remaining with stocks
    remaining = max_symbols - len(idx_symbols)
    stock_symbols = stocks_pending[:remaining]

    all_symbols = idx_symbols + stock_symbols

    logger.log_info(f"Starting data ingestion cycle: {len(all_symbols)} symbols "
                   f"(indexes: {len(idx_symbols)}, stocks: {len(stock_symbols)})")

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

                # Generate and store price prediction
                price_pred = generate_price_prediction(symbol, processed)
                if price_pred:
                    store_price_prediction(symbol, price_pred)
                    # Use price prediction target for time prediction
                    target_price = price_pred['target_price']
                else:
                    # Fallback to simple 5% target
                    target_price = processed['price'] * 1.05

                # Generate and store time-to-hit prediction
                time_pred = generate_time_prediction(symbol, processed, target_price)
                if time_pred:
                    store_time_prediction(symbol, time_pred)

                succeeded += 1
                pred_info = f"price=${price_pred['target_price']:.2f}" if price_pred else "no prediction"
                logger.log_info(f"Processed {symbol}: ${processed['price']:.2f} ({pred_info})")
            else:
                errors.append({'symbol': symbol, 'error': 'store_failed'})

            # Rate limiting - use configurable delay (premium API allows faster)
            time.sleep(API_CALL_DELAY)

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
