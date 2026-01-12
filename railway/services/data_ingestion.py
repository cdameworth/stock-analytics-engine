"""
Railway-native Stock Data Ingestion Service.
Fetches stock data from Alpha Vantage API and stores in PostgreSQL.
No AWS dependencies - designed for Railway deployment.
"""

import os
import logging
import urllib.request
import urllib.parse
import json
import time
import socket
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# PostgreSQL database connection
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lambda_functions.shared.database import db, is_database_available

# Configuration from environment
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
CONNECT_TEST_HOST = os.environ.get('CONNECT_TEST_HOST', 'www.alphavantage.co')
CONNECT_TEST_PORT = 443
CONNECT_TEST_TIMEOUT = int(os.environ.get('CONNECT_TEST_TIMEOUT', '2'))
PER_CALL_TIMEOUT = int(os.environ.get('PER_CALL_TIMEOUT', '10'))
MAX_SYMBOLS_PER_RUN = int(os.environ.get('MAX_SYMBOLS_PER_RUN', '5'))
API_CALLS_PER_MINUTE = int(os.environ.get('API_CALLS_PER_MINUTE', '5'))

# Stock universe
MAJOR_INDEXES = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']

POPULAR_STOCKS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY',
    # Consumer
    'WMT', 'HD', 'MCD', 'SBUX', 'NKE', 'COST',
    # Industrial/Energy
    'XOM', 'CVX', 'CAT', 'BA', 'GE', 'HON',
    # Growth
    'AMD', 'CRM', 'ADBE', 'NFLX', 'PYPL', 'SQ',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'SCHD',
    # Additional
    'DIS', 'KO', 'PEP', 'IBM', 'INTC', 'ORCL'
]

# Remove duplicates
POPULAR_STOCKS = list(dict.fromkeys(POPULAR_STOCKS))


class DataIngestionService:
    """Railway-native data ingestion service using PostgreSQL."""

    def __init__(self):
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.last_api_call = 0
        self.api_call_count = 0

    def _rate_limit(self):
        """Enforce API rate limiting."""
        now = time.time()
        if now - self.last_api_call < (60 / API_CALLS_PER_MINUTE):
            sleep_time = (60 / API_CALLS_PER_MINUTE) - (now - self.last_api_call)
            logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_api_call = time.time()
        self.api_call_count += 1

    def _internet_reachable(self) -> bool:
        """Check if Alpha Vantage API is reachable."""
        try:
            with socket.create_connection(
                (CONNECT_TEST_HOST, CONNECT_TEST_PORT),
                CONNECT_TEST_TIMEOUT
            ):
                return True
        except Exception as e:
            logger.error(f"No outbound connectivity: {e}")
            return False

    def fetch_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current quote for a symbol from Alpha Vantage."""
        if not self.api_key:
            logger.error("No Alpha Vantage API key configured")
            return None

        self._rate_limit()

        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }

        url = f"{ALPHA_VANTAGE_BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'StockAnalyticsEngine/1.0')

            with urllib.request.urlopen(request, timeout=PER_CALL_TIMEOUT) as response:
                data = json.loads(response.read().decode('utf-8'))

            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                return {
                    'symbol': symbol,
                    'price': float(quote.get('05. price', 0)),
                    'open': float(quote.get('02. open', 0)),
                    'high': float(quote.get('03. high', 0)),
                    'low': float(quote.get('04. low', 0)),
                    'volume': int(quote.get('06. volume', 0)),
                    'previous_close': float(quote.get('08. previous close', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%').replace('%', ''),
                    'latest_trading_day': quote.get('07. latest trading day'),
                    'fetched_at': datetime.utcnow().isoformat()
                }
            elif 'Note' in data:
                logger.warning(f"API rate limit reached: {data['Note']}")
                return None
            elif 'Error Message' in data:
                logger.error(f"API error for {symbol}: {data['Error Message']}")
                return None
            else:
                logger.warning(f"No data for {symbol}: {data}")
                return None

        except urllib.error.URLError as e:
            logger.error(f"URL error fetching {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def store_quote(self, quote: Dict[str, Any]) -> bool:
        """Store quote data in PostgreSQL."""
        if not is_database_available():
            logger.error("Database not available")
            return False

        try:
            # Check if we have a stock_quotes table, if not use recommendations
            result = db.execute_one("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'stock_quotes'
                )
            """)

            if result and result[0]:
                # Use stock_quotes table
                db.execute("""
                    INSERT INTO stock_quotes (
                        symbol, price, open_price, high_price, low_price,
                        volume, previous_close, change_amount, change_percent,
                        trading_day, fetched_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, trading_day)
                    DO UPDATE SET
                        price = EXCLUDED.price,
                        volume = EXCLUDED.volume,
                        fetched_at = EXCLUDED.fetched_at
                """, (
                    quote['symbol'],
                    quote['price'],
                    quote['open'],
                    quote['high'],
                    quote['low'],
                    quote['volume'],
                    quote['previous_close'],
                    quote['change'],
                    float(quote['change_percent']),
                    quote['latest_trading_day'],
                    quote['fetched_at']
                ))
            else:
                # Store in recommendations table as a fallback
                db.execute("""
                    INSERT INTO recommendations (
                        symbol, recommendation, current_price,
                        confidence_score, analysis_summary
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) WHERE created_at > NOW() - INTERVAL '1 day'
                    DO UPDATE SET
                        current_price = EXCLUDED.current_price,
                        updated_at = NOW()
                """, (
                    quote['symbol'],
                    'HOLD',  # Default recommendation
                    quote['price'],
                    0.5,  # Default confidence
                    f"Price: ${quote['price']:.2f}, Change: {quote['change_percent']}%"
                ))

            logger.info(f"Stored quote for {quote['symbol']}: ${quote['price']:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error storing quote for {quote['symbol']}: {e}")
            return False

    def run_ingestion(self, symbols: List[str] = None, max_symbols: int = None) -> Dict[str, Any]:
        """Run data ingestion for specified symbols."""
        if not self._internet_reachable():
            return {
                'success': False,
                'error': 'No internet connectivity',
                'symbols_processed': 0
            }

        if symbols is None:
            symbols = POPULAR_STOCKS

        if max_symbols is None:
            max_symbols = MAX_SYMBOLS_PER_RUN

        # Rotate through symbols based on time
        hour_offset = datetime.utcnow().hour % len(symbols)
        symbols_to_process = symbols[hour_offset:hour_offset + max_symbols]
        if len(symbols_to_process) < max_symbols:
            symbols_to_process += symbols[:max_symbols - len(symbols_to_process)]

        results = {
            'success': True,
            'symbols_processed': 0,
            'symbols_failed': 0,
            'quotes': [],
            'errors': [],
            'timestamp': datetime.utcnow().isoformat()
        }

        for symbol in symbols_to_process:
            try:
                quote = self.fetch_quote(symbol)
                if quote:
                    stored = self.store_quote(quote)
                    if stored:
                        results['symbols_processed'] += 1
                        results['quotes'].append({
                            'symbol': symbol,
                            'price': quote['price']
                        })
                    else:
                        results['symbols_failed'] += 1
                        results['errors'].append(f"Failed to store {symbol}")
                else:
                    results['symbols_failed'] += 1
                    results['errors'].append(f"No data for {symbol}")

            except Exception as e:
                results['symbols_failed'] += 1
                results['errors'].append(f"Error processing {symbol}: {str(e)}")
                logger.error(f"Error processing {symbol}: {e}")

        results['success'] = results['symbols_failed'] == 0
        return results


def create_stock_quotes_table():
    """Create the stock_quotes table if it doesn't exist."""
    if not is_database_available():
        logger.error("Database not available")
        return False

    try:
        db.execute("""
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

            CREATE INDEX IF NOT EXISTS idx_stock_quotes_symbol
                ON stock_quotes(symbol);
            CREATE INDEX IF NOT EXISTS idx_stock_quotes_trading_day
                ON stock_quotes(trading_day DESC);
        """)
        logger.info("stock_quotes table created/verified")
        return True
    except Exception as e:
        logger.error(f"Error creating stock_quotes table: {e}")
        return False


# Module-level instance for easy access
_service = None

def get_service() -> DataIngestionService:
    """Get or create the data ingestion service instance."""
    global _service
    if _service is None:
        _service = DataIngestionService()
    return _service


def run_ingestion(symbols: List[str] = None, max_symbols: int = None) -> Dict[str, Any]:
    """Convenience function to run data ingestion."""
    service = get_service()
    return service.run_ingestion(symbols, max_symbols)
