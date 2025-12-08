"""
Model Tuning Worker for Railway deployment.
Runs scheduled ML model optimization and hyperparameter tuning.
"""

import os
import sys
import time
import schedule
from datetime import datetime, time as dt_time
import pytz

# Add lambda_functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lambda_functions import price_model_tuning
from lambda_functions import time_model_tuning
from lambda_functions import dual_accuracy_tracker
from lambda_functions.shared.error_handling import StructuredLogger

logger = StructuredLogger(__name__)

class MockLambdaContext:
    """Mock Lambda context for compatibility."""
    def __init__(self, function_name):
        self.function_name = function_name
        self.memory_limit_in_mb = 2048
        self.invoked_function_arn = f"arn:aws:lambda:railway:{function_name}"
        self.aws_request_id = None

    def get_remaining_time_in_millis(self):
        return 600000  # 10 minutes for tuning tasks

def run_price_model_tuning():
    """Execute price model tuning."""
    try:
        logger.log_info("Starting price model tuning")

        event = {
            'httpMethod': 'POST',
            'path': '/tune/price',
            'body': '{"lookback_days": 90}',
            'headers': {},
            'requestContext': {
                'requestId': f'railway-price-tuning-{datetime.utcnow().isoformat()}',
                'identity': {'sourceIp': '127.0.0.1'}
            }
        }

        context = MockLambdaContext('price-model-tuning')
        context.aws_request_id = event['requestContext']['requestId']

        response = price_model_tuning.lambda_handler(event, context)

        if response.get('statusCode') == 200:
            logger.log_info(f"Price model tuning completed: {response.get('body')}")
        else:
            logger.log_error(f"Price model tuning failed: {response.get('body')}")

    except Exception as e:
        logger.log_error(f"Error during price model tuning: {str(e)}", error=e)

def run_time_model_tuning():
    """Execute time-to-hit model tuning."""
    try:
        logger.log_info("Starting time-to-hit model tuning")

        event = {
            'httpMethod': 'POST',
            'path': '/tune/time',
            'body': '{"lookback_days": 90}',
            'headers': {},
            'requestContext': {
                'requestId': f'railway-time-tuning-{datetime.utcnow().isoformat()}',
                'identity': {'sourceIp': '127.0.0.1'}
            }
        }

        context = MockLambdaContext('time-model-tuning')
        context.aws_request_id = event['requestContext']['requestId']

        response = time_model_tuning.lambda_handler(event, context)

        if response.get('statusCode') == 200:
            logger.log_info(f"Time model tuning completed: {response.get('body')}")
        else:
            logger.log_error(f"Time model tuning failed: {response.get('body')}")

    except Exception as e:
        logger.log_error(f"Error during time model tuning: {str(e)}", error=e)

def run_accuracy_tracking():
    """Execute accuracy tracking and validation."""
    try:
        logger.log_info("Starting accuracy tracking")

        event = {
            'httpMethod': 'POST',
            'path': '/track/accuracy',
            'body': '{}',
            'headers': {},
            'requestContext': {
                'requestId': f'railway-accuracy-{datetime.utcnow().isoformat()}',
                'identity': {'sourceIp': '127.0.0.1'}
            }
        }

        context = MockLambdaContext('dual-accuracy-tracker')
        context.aws_request_id = event['requestContext']['requestId']

        response = dual_accuracy_tracker.lambda_handler(event, context)

        if response.get('statusCode') == 200:
            logger.log_info(f"Accuracy tracking completed: {response.get('body')}")
        else:
            logger.log_error(f"Accuracy tracking failed: {response.get('body')}")

    except Exception as e:
        logger.log_error(f"Error during accuracy tracking: {str(e)}", error=e)

def run_weekly_comprehensive_tuning():
    """Run comprehensive weekly model tuning (Sunday 2 AM EST)."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)

    # Only run on Sunday
    if now.weekday() == 6:  # Sunday
        logger.log_info("Running weekly comprehensive model tuning")
        run_price_model_tuning()
        time.sleep(300)  # Wait 5 minutes between tuning jobs
        run_time_model_tuning()
        time.sleep(300)
        run_accuracy_tracking()
    else:
        logger.log_info(f"Not Sunday ({now.strftime('%A')}) - skipping weekly tuning")

def run_daily_validation():
    """Run daily model validation and accuracy tracking."""
    logger.log_info("Running daily validation and accuracy tracking")
    run_accuracy_tracking()

def setup_schedules():
    """Set up the job schedules."""
    # Daily accuracy validation at 6 AM EST (11:00 UTC)
    schedule.every().day.at("11:00").do(run_daily_validation)
    logger.log_info("Scheduled daily validation at 6:00 AM EST (11:00 UTC)")

    # Weekly comprehensive tuning on Sunday at 2 AM EST (7:00 UTC)
    schedule.every().sunday.at("07:00").do(run_weekly_comprehensive_tuning)
    logger.log_info("Scheduled weekly tuning on Sunday at 2:00 AM EST (7:00 UTC)")

    # Optional: Run accuracy check every 6 hours for continuous monitoring
    if os.environ.get('ENABLE_CONTINUOUS_MONITORING', 'false').lower() == 'true':
        schedule.every(6).hours.do(run_accuracy_tracking)
        logger.log_info("Enabled continuous monitoring: accuracy check every 6 hours")

def main():
    """Main worker loop."""
    logger.log_info("Starting Stock Analytics Model Tuning Worker")
    logger.log_info(f"Environment: {os.environ.get('ENVIRONMENT', 'production')}")

    # Setup schedules
    setup_schedules()

    # Run initial accuracy tracking on startup (after 5 minute delay)
    logger.log_info("Waiting 5 minutes before initial accuracy tracking")
    time.sleep(300)
    run_accuracy_tracking()

    # Main loop
    logger.log_info("Worker running - waiting for scheduled jobs")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.log_info("Worker shutting down gracefully")
            break
        except Exception as e:
            logger.log_error(f"Error in worker loop: {str(e)}", error=e)
            time.sleep(300)  # Wait 5 minutes before retrying

if __name__ == '__main__':
    main()
