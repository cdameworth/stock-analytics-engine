"""
Model Tuning Worker for Railway deployment.
Runs scheduled ML model optimization, hyperparameter tuning, and accuracy tracking.

Scheduled Jobs:
    - Daily validation (6 AM EST / 11:00 UTC)
    - Daily confidence calibration (7 AM EST / 12:00 UTC)
    - Daily symbol accuracy aggregation (7:30 AM EST / 12:30 UTC)
    - Daily market condition tracking (4:30 PM EST / 21:30 UTC)
    - Weekly comprehensive tuning (Sunday 2 AM EST / 07:00 UTC)
    - Weekly deployment gate evaluation (Sunday 3 AM EST / 08:00 UTC)
"""

import os
import sys
import time
import schedule
from datetime import datetime, time as dt_time
import pytz

# Set AWS region for boto3 compatibility (even though we use PostgreSQL on Railway)
# This prevents import errors from Lambda functions that still have boto3 imports
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')
os.environ.setdefault('AWS_REGION', 'us-east-1')

# Add lambda_functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Lambda functions with error handling for Railway compatibility
try:
    from lambda_functions import price_model_tuning
    from lambda_functions import time_model_tuning
    from lambda_functions import dual_accuracy_tracker
    LAMBDA_FUNCTIONS_AVAILABLE = True
except Exception as e:
    # Lambda functions may not be fully compatible with Railway yet
    LAMBDA_FUNCTIONS_AVAILABLE = False
    price_model_tuning = None
    time_model_tuning = None
    dual_accuracy_tracker = None
    print(f"Warning: Lambda functions not available: {e}")

from lambda_functions.shared.error_handling import StructuredLogger

# Import accuracy tracking modules (these use PostgreSQL, fully Railway-compatible)
from lambda_functions.shared.accuracy_tracking import (
    ConfidenceCalibrationTracker,
    SymbolAccuracyAggregator,
    DeploymentGate,
    MarketConditionTracker,
    ErrorDistributionAnalyzer
)

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
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        logger.log_info("Price model tuning skipped - Lambda functions not available on Railway")
        logger.log_info("Using PostgreSQL-based accuracy tracking instead")
        return

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
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        logger.log_info("Time model tuning skipped - Lambda functions not available on Railway")
        logger.log_info("Using PostgreSQL-based accuracy tracking instead")
        return

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
    if not LAMBDA_FUNCTIONS_AVAILABLE:
        logger.log_info("Legacy accuracy tracking skipped - using PostgreSQL tracking instead")
        # Run the PostgreSQL-based accuracy tracking
        run_confidence_calibration()
        run_symbol_accuracy_aggregation()
        return

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


# ============================================================
# NEW ACCURACY TRACKING JOBS
# ============================================================

def run_confidence_calibration():
    """Calculate Expected Calibration Error by confidence bucket."""
    try:
        logger.log_info("Starting confidence calibration analysis")

        tracker = ConfidenceCalibrationTracker()

        # Run for both model types
        for model_type in ['price_prediction', 'time_prediction']:
            result = tracker.calculate_ece(model_type=model_type, lookback_days=30)

            if result.get('ece') is not None:
                logger.log_info(
                    f"Calibration [{model_type}]: ECE={result['ece']:.4f} "
                    f"({result['calibration_quality']}), "
                    f"predictions={result['total_predictions']}"
                )
            else:
                logger.log_info(f"Calibration [{model_type}]: insufficient data")

    except Exception as e:
        logger.log_error(f"Error during confidence calibration: {str(e)}", error=e)


def run_symbol_accuracy_aggregation():
    """Aggregate accuracy by symbol and identify retraining candidates."""
    try:
        logger.log_info("Starting symbol accuracy aggregation")

        aggregator = SymbolAccuracyAggregator()
        result = aggregator.aggregate_all_symbols(lookback_days=30)

        logger.log_info(
            f"Symbol accuracy: {result['total_symbols']} symbols analyzed, "
            f"{result['retraining_count']} need retraining"
        )

        # Log top 5 and bottom 5 symbols
        if result['symbols']:
            top_5 = result['symbols'][:5]
            bottom_5 = result['symbols'][-5:] if len(result['symbols']) > 5 else []

            logger.log_info("Top 5 symbols: " + ", ".join(
                f"{s['symbol']}({s['accuracy_rate']:.1%})" for s in top_5
            ))

            if bottom_5:
                logger.log_info("Bottom 5 symbols: " + ", ".join(
                    f"{s['symbol']}({s['accuracy_rate']:.1%})" for s in bottom_5
                ))

    except Exception as e:
        logger.log_error(f"Error during symbol accuracy aggregation: {str(e)}", error=e)


def run_deployment_gate_evaluation():
    """Evaluate if current model meets deployment criteria."""
    try:
        logger.log_info("Starting deployment gate evaluation")

        gate = DeploymentGate()

        for model_type in ['price_prediction', 'time_prediction']:
            result = gate.evaluate(model_type=model_type)

            if result['passed']:
                logger.log_info(
                    f"Deployment gate [{model_type}]: PASSED "
                    f"(accuracy={result['metrics']['accuracy']:.1%})"
                )
            else:
                logger.log_info(
                    f"Deployment gate [{model_type}]: FAILED "
                    f"(reasons: {', '.join(result['failure_reasons'])})"
                )

    except Exception as e:
        logger.log_error(f"Error during deployment gate evaluation: {str(e)}", error=e)


def run_market_condition_tracking():
    """Track current market regime and store conditions."""
    try:
        logger.log_info("Starting market condition tracking")

        tracker = MarketConditionTracker()

        # Classify current regime
        regime = tracker.classify_current_regime()
        logger.log_info(f"Current market regime: {regime}")

        # Get correlation with accuracy
        correlation = tracker.correlate_accuracy_with_regime()

        if 'regimes' in correlation:
            for regime_name, stats in correlation['regimes'].items():
                logger.log_info(
                    f"Accuracy in {regime_name} market: {stats['accuracy_rate']:.1%} "
                    f"(outperformance: {stats['outperformance']:+.1%})"
                )

    except Exception as e:
        logger.log_error(f"Error during market condition tracking: {str(e)}", error=e)


def run_error_distribution_analysis():
    """Analyze error magnitude distribution."""
    try:
        logger.log_info("Starting error distribution analysis")

        analyzer = ErrorDistributionAnalyzer()

        for model_type in ['price_prediction', 'time_prediction']:
            result = analyzer.calculate_distribution(model_type=model_type, lookback_days=30)

            if 'error' not in result:
                logger.log_info(
                    f"Error distribution [{model_type}]: "
                    f"median={result['p50']:.2%}, "
                    f"mean={result['mean']:.2%}, "
                    f"outliers={result['outlier_count']}"
                )

    except Exception as e:
        logger.log_error(f"Error during error distribution analysis: {str(e)}", error=e)


def setup_schedules():
    """Set up the job schedules."""
    # ============================================================
    # DAILY SCHEDULES
    # ============================================================

    # Daily accuracy validation at 6 AM EST (11:00 UTC)
    schedule.every().day.at("11:00").do(run_daily_validation)
    logger.log_info("Scheduled daily validation at 6:00 AM EST (11:00 UTC)")

    # Daily confidence calibration at 7 AM EST (12:00 UTC)
    schedule.every().day.at("12:00").do(run_confidence_calibration)
    logger.log_info("Scheduled confidence calibration at 7:00 AM EST (12:00 UTC)")

    # Daily symbol accuracy aggregation at 7:30 AM EST (12:30 UTC)
    schedule.every().day.at("12:30").do(run_symbol_accuracy_aggregation)
    logger.log_info("Scheduled symbol accuracy aggregation at 7:30 AM EST (12:30 UTC)")

    # Daily error distribution analysis at 8 AM EST (13:00 UTC)
    schedule.every().day.at("13:00").do(run_error_distribution_analysis)
    logger.log_info("Scheduled error distribution analysis at 8:00 AM EST (13:00 UTC)")

    # Daily market condition tracking at 4:30 PM EST (21:30 UTC)
    schedule.every().day.at("21:30").do(run_market_condition_tracking)
    logger.log_info("Scheduled market condition tracking at 4:30 PM EST (21:30 UTC)")

    # ============================================================
    # WEEKLY SCHEDULES
    # ============================================================

    # Weekly comprehensive tuning on Sunday at 2 AM EST (7:00 UTC)
    schedule.every().sunday.at("07:00").do(run_weekly_comprehensive_tuning)
    logger.log_info("Scheduled weekly tuning on Sunday at 2:00 AM EST (7:00 UTC)")

    # Weekly deployment gate evaluation on Sunday at 3 AM EST (8:00 UTC)
    schedule.every().sunday.at("08:00").do(run_deployment_gate_evaluation)
    logger.log_info("Scheduled deployment gate evaluation on Sunday at 3:00 AM EST (8:00 UTC)")

    # ============================================================
    # OPTIONAL CONTINUOUS MONITORING
    # ============================================================

    if os.environ.get('ENABLE_CONTINUOUS_MONITORING', 'false').lower() == 'true':
        schedule.every(6).hours.do(run_accuracy_tracking)
        schedule.every(6).hours.do(run_confidence_calibration)
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
