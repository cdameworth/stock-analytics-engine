"""
Data Ingestion Worker for Railway deployment.
Runs scheduled data collection from Alpha Vantage API.
"""

import os
import sys
import time
import schedule
from datetime import datetime, time as dt_time
import pytz

# Add lambda_functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lambda_functions import stock_data_ingestion
from lambda_functions.shared.error_handling import StructuredLogger

logger = StructuredLogger(__name__)

class MockLambdaContext:
    """Mock Lambda context for compatibility."""
    def __init__(self):
        self.function_name = "railway-data-ingestion"
        self.memory_limit_in_mb = 2048
        self.invoked_function_arn = "arn:aws:lambda:railway:data-ingestion"
        self.aws_request_id = None

    def get_remaining_time_in_millis(self):
        return 300000  # 5 minutes

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

def run_data_ingestion():
    """Execute data ingestion Lambda function."""
    try:
        logger.log_info("Starting data ingestion cycle")

        event = {
            'httpMethod': 'POST',
            'path': '/ingest',
            'body': '{}',
            'headers': {},
            'requestContext': {
                'requestId': f'railway-{datetime.utcnow().isoformat()}',
                'identity': {'sourceIp': '127.0.0.1'}
            }
        }

        context = MockLambdaContext()
        context.aws_request_id = event['requestContext']['requestId']

        # Execute Lambda handler
        response = stock_data_ingestion.lambda_handler(event, context)

        if response.get('statusCode') == 200:
            logger.log_info(f"Data ingestion completed successfully: {response.get('body')}")
        else:
            logger.log_error(f"Data ingestion failed: {response.get('body')}")

    except Exception as e:
        logger.log_error(f"Error during data ingestion: {str(e)}", error=e)

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
    # Get configuration from environment
    market_interval = int(os.environ.get('MARKET_INTERVAL_MINUTES', 5))
    evening_interval = int(os.environ.get('EVENING_INTERVAL_MINUTES', 10))

    # Market hours: Every N minutes (default: 5)
    schedule.every(market_interval).minutes.do(run_market_hours_job)
    logger.log_info(f"Scheduled market hours job: every {market_interval} minutes")

    # Evening hours: Every N minutes (default: 10)
    schedule.every(evening_interval).minutes.do(run_evening_job)
    logger.log_info(f"Scheduled evening job: every {evening_interval} minutes")

    # End of day comprehensive run at 4:30 PM EST
    schedule.every().day.at("16:30").do(
        lambda: run_data_ingestion() if datetime.now(pytz.timezone('US/Eastern')).weekday() < 5 else None
    )
    logger.log_info("Scheduled end-of-day comprehensive run at 4:30 PM EST")

def main():
    """Main worker loop."""
    logger.log_info("Starting Stock Analytics Data Ingestion Worker")
    logger.log_info(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")

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
            time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logger.log_info("Worker shutting down gracefully")
            break
        except Exception as e:
            logger.log_error(f"Error in worker loop: {str(e)}", error=e)
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == '__main__':
    main()
