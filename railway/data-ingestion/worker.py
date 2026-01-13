"""
Data Ingestion Worker for Railway deployment.
Runs scheduled data collection from Alpha Vantage API.
Uses Railway-native PostgreSQL storage - NO AWS dependencies.
"""

import os
import sys
import time
import schedule
from datetime import datetime, time as dt_time
import pytz

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Railway-native services (no AWS dependencies)
from services.data_ingestion import (
    DataIngestionService,
    create_stock_quotes_table,
    run_ingestion,
    POPULAR_STOCKS
)
from lambda_functions.shared.error_handling import StructuredLogger

logger = StructuredLogger(__name__)


def is_market_hours():
    """Check if current time is during market hours (9 AM - 4 PM EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    # Market is open Monday-Friday
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Market hours: 9:00 AM - 4:00 PM EST
    market_open = dt_time(9, 0)
    market_close = dt_time(16, 0)
    current_time = now.time()

    return market_open <= current_time <= market_close


def is_evening_hours():
    """Check if current time is evening processing hours (5 PM - 11 PM EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    # Evening processing Monday-Friday
    if now.weekday() >= 5:
        return False

    evening_start = dt_time(17, 0)
    evening_end = dt_time(23, 0)
    current_time = now.time()

    return evening_start <= current_time <= evening_end


def run_data_ingestion(max_symbols: int = 5):
    """Execute data ingestion using Railway-native service."""
    try:
        logger.log_info(f"Starting data ingestion cycle (max_symbols={max_symbols})")

        result = run_ingestion(max_symbols=max_symbols)

        if result['success']:
            logger.log_info(
                f"Data ingestion completed: {result['symbols_processed']} symbols processed"
            )
            if result.get('quotes'):
                for quote in result['quotes'][:3]:  # Log first 3
                    logger.log_info(f"  {quote['symbol']}: ${quote['price']:.2f}")
        else:
            errors = result.get('errors', [])
            if errors:
                logger.log_info(f"Data ingestion finished with {len(errors)} errors")
            else:
                logger.log_info("Data ingestion completed (no data)")

        return result

    except Exception as e:
        logger.log_error(e, context={'operation': 'data_ingestion'})
        return {'success': False, 'error': str(e)}


def run_market_hours_job():
    """Job for market hours - runs only during trading hours."""
    if is_market_hours():
        logger.log_info("Market hours active - running data ingestion")
        run_data_ingestion(max_symbols=5)
    else:
        logger.log_info("Outside market hours - skipping ingestion")


def run_evening_job():
    """Job for evening hours - extended processing."""
    if is_evening_hours():
        logger.log_info("Evening hours active - running extended data ingestion")
        run_data_ingestion(max_symbols=10)
    else:
        logger.log_info("Outside evening hours - skipping")


def run_end_of_day_job():
    """Comprehensive end-of-day data collection."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    # Only run on weekdays
    if now.weekday() < 5:
        logger.log_info("Running end-of-day comprehensive data collection")
        run_data_ingestion(max_symbols=20)


def setup_schedules():
    """Set up the job schedules."""
    # Get configuration from environment
    market_interval = int(os.environ.get('MARKET_INTERVAL_MINUTES', 5))
    evening_interval = int(os.environ.get('EVENING_INTERVAL_MINUTES', 10))

    # Market hours: Every N minutes (default: 5)
    schedule.every(market_interval).minutes.do(run_market_hours_job)
    logger.log_info(f"Scheduled market hours job: every {market_interval} minutes")

    # Evening hours: Every N minutes (default: 10)
    schedule.every(evening_interval).minutes.do(run_evening_job)
    logger.log_info(f"Scheduled evening job: every {evening_interval} minutes")

    # End of day comprehensive run at 4:30 PM EST (21:30 UTC)
    schedule.every().day.at("21:30").do(run_end_of_day_job)
    logger.log_info("Scheduled end-of-day comprehensive run at 4:30 PM EST")


def main():
    """Main worker loop."""
    logger.log_info("Starting Stock Analytics Data Ingestion Worker (Railway-native)")
    logger.log_info(f"Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'production')}")
    logger.log_info(f"Stock universe: {len(POPULAR_STOCKS)} symbols")

    # Check for API key
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
    if api_key:
        logger.log_info("Alpha Vantage API key configured")
    else:
        logger.log_info("WARNING: No ALPHA_VANTAGE_API_KEY configured")

    # Ensure database tables exist
    logger.log_info("Initializing database tables")
    create_stock_quotes_table()

    # Setup schedules
    setup_schedules()

    # Run initial ingestion on startup
    logger.log_info("Running initial data ingestion")
    run_data_ingestion(max_symbols=3)

    # Main loop
    logger.log_info("Worker running - waiting for scheduled jobs")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logger.log_info("Worker shutting down gracefully")
            break
        except Exception as e:
            logger.log_error(e, context={'operation': 'worker_loop'})
            time.sleep(60)  # Wait 1 minute before retrying


if __name__ == '__main__':
    main()
